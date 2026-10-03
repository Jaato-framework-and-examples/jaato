"""The jaato SELinux module grants what a runner needs and nothing it must not have.

docs/design/selinux-backend.md §11.  The module is compiled, linked into
the full ``targeted`` policy (``semodule -N -i``: no kernel needed), and
the linked policy is queried with setools.  Queries are EFFECTIVE: a rule
granted to an attribute the domain carries (``domain``, ``files_type``)
counts, because that is what the kernel would apply.

Each assertion carries its own reversion: the edit to ``jaato.te`` that
must make it fail.  ``test_each_assertion_detects_its_reversion`` rebuilds
the policy with that edit and checks the assertion now fails, so an
assertion the base policy satisfies anyway cannot pass unnoticed.  Phase 0
found exactly that shape: targeted grants ``unconfined_t`` transition into
every domain, so a check on the module's own transition line proved
nothing.  The repository's reversion meta-guard cannot run these (it runs
on Ubuntu, which has no targeted policy), so the pairing lives here.

Where it runs: the ``selinux-policy`` CI job, in a Fedora container with
``selinux-policy-devel``, ``selinux-policy-targeted`` and ``setools``, and
``JAATO_SELINUX_POLICY_TESTS=1`` set.  Anywhere else the module skips.
With the variable set, a missing tool FAILS rather than skips, so the job
cannot pass by not testing.  Locally::

    docker run --rm -v "$PWD":/w:ro -w /w fedora:44 bash -c \\
      'dnf -y -q install selinux-policy-devel selinux-policy-targeted \\
         setools-console policycoreutils make python3-pytest &&
       JAATO_SELINUX_POLICY_TESTS=1 pytest -q -p no:cacheprovider \\
         --noconftest jaato-server/selinux/tests'

Needs root in the container: ``semodule -N -i`` writes the policy store.
Never set the variable on a real SELinux host: ``-N`` skips the reload but
still installs every test build into that host's module store.
"""

from __future__ import annotations

import glob
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, FrozenSet, Optional, Tuple

import pytest

ENV = "JAATO_SELINUX_POLICY_TESTS"
MODULE_DIR = Path(__file__).resolve().parents[1]
DEVEL_MAKEFILE = "/usr/share/selinux/devel/Makefile"

ENABLED = os.environ.get(ENV) == "1"

# Skip per test, not per module: a module-level skip collects no tests and
# pytest exits 5, which check.py reads as a failing leg.
pytestmark = pytest.mark.skipif(
    not ENABLED, reason=f"{ENV}=1 not set: runs in the selinux-policy CI job")

if ENABLED:
    # With the variable set, a missing toolchain is an error, not a skip:
    # a job that skips proves nothing.
    import setools
    if not os.path.exists(DEVEL_MAKEFILE) or not shutil.which("semodule"):
        raise RuntimeError(
            f"{ENV}=1 but the policy toolchain is missing ({DEVEL_MAKEFILE}, "
            "semodule); refusing to skip")
else:
    setools = None


# ----------------------------------------------------------------------
# Building a policy with a given jaato.te
# ----------------------------------------------------------------------

_cache: dict = {}


def _policy_path() -> str:
    paths = sorted(glob.glob("/etc/selinux/targeted/policy/policy.*"))
    assert paths, "no linked targeted policy in the store"
    return paths[-1]


def build(te_text: str):
    """Compile *te_text* as the jaato module, link it, return the policy.

    The store holds one jaato module at a time, so the last build is
    cached by content and a repeat returns it without rebuilding.
    """
    key = hashlib.sha256(te_text.encode()).hexdigest()
    if _cache.get("key") == key:
        return _cache["policy"]
    work = Path(tempfile.mkdtemp(prefix="jaato-sepol-"))
    try:
        for name in ("jaato.fc", "jaato.if"):
            shutil.copy(MODULE_DIR / name, work / name)
        (work / "jaato.te").write_text(te_text)
        run = subprocess.run(["make", "-f", DEVEL_MAKEFILE, "jaato.pp"],
                             cwd=work, capture_output=True, text=True)
        assert run.returncode == 0, f"module build failed:\n{run.stdout}{run.stderr}"
        run = subprocess.run(["semodule", "-N", "-i", "jaato.pp"],
                             cwd=work, capture_output=True, text=True)
        assert run.returncode == 0, f"semodule failed:\n{run.stdout}{run.stderr}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    policy = setools.SELinuxPolicy(_policy_path())
    _cache.update(key=key, policy=policy)
    return policy


def shipped_te() -> str:
    return (MODULE_DIR / "jaato.te").read_text()


# ----------------------------------------------------------------------
# Effective queries
# ----------------------------------------------------------------------

def granted(policy, source: str, target: Optional[str], tclass: str,
            perms: FrozenSet[str], *, conditional: bool = True) -> FrozenSet[str]:
    """The subset of *perms* the policy allows *source* on *target*:*tclass*.

    Attributes are expanded on both sides, so a rule written for
    ``domain`` or ``files_type`` counts.  A rule written with ``self`` is
    stored against the source type, which *target* == *source* matches.
    ``target=None`` matches any target.

    ``conditional=False`` counts only rules no boolean can switch off.
    A presence assertion uses it: targeted grants ``domain domain:fd use``
    only under ``domain_fd_use``, so a grant this module must guarantee
    (the runner's inherited RPC socket) cannot rest on a boolean an
    operator may turn off.  An absence assertion counts every rule, since
    a forbidden grant is forbidden under any boolean.
    """
    kwargs = dict(ruletype=[setools.TERuletype.allow], source=source,
                  tclass=[tclass], source_indirect=True, target_indirect=True)
    if target is not None:
        kwargs["target"] = target
    out = set()
    for rule in setools.TERuleQuery(policy, **kwargs).results():
        if not conditional and _is_conditional(rule):
            continue
        out |= set(rule.perms) & set(perms)
    return frozenset(out)


def _granted_directly(policy, source: str, tclass: str, perms) -> FrozenSet[str]:
    """The subset of *perms* granted by rules whose source IS *source*.

    Unlike ``granted``, a rule written for an attribute the type carries
    (``domain``) does not count: it answers "what does this module give
    the type", not "what may the type do on this host".
    """
    out = set()
    query = setools.TERuleQuery(policy, ruletype=[setools.TERuletype.allow],
                                source=source, source_indirect=False,
                                tclass=[tclass])
    for rule in query.results():
        out |= set(rule.perms) & set(perms)
    return frozenset(out)


def _is_conditional(rule) -> bool:
    try:
        rule.conditional
    except Exception:  # setools raises RuleNotConditional
        return False
    return True


def type_attrs(policy, name: str) -> FrozenSet[str]:
    return frozenset(str(a) for a in policy.lookup_type(name).attributes())


def role_types(policy, role: str) -> FrozenSet[str]:
    return frozenset(str(t) for t in policy.lookup_role(role).types())


def class_perms(policy, tclass: str) -> FrozenSet[str]:
    """Every permission *tclass* defines, its common's included."""
    cls = policy.lookup_class(tclass)
    perms = set(cls.perms)
    try:
        perms |= set(cls.common.perms)
    except Exception:  # setools raises NoCommon for a class without one
        pass
    return frozenset(perms)


def type_exists(policy, name: str) -> bool:
    try:
        policy.lookup_type(name)
        return True
    except Exception:  # setools raises InvalidType
        return False


# ----------------------------------------------------------------------
# The assertions
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    """One property of the module, and the edit that must break it.

    ``holds(policy)`` is True when the property is present in the policy.
    ``find``/``replace`` edit jaato.te: ``find`` must occur exactly once;
    an ``append`` is added at the end instead (an absence is broken by
    ADDING the forbidden grant).
    """

    name: str
    holds: Callable
    because: str
    find: Optional[str] = None
    replace: str = ""
    append: Optional[str] = None


def born_as(policy, source: str, parent: str, tclass: str) -> FrozenSet[str]:
    """The types a *tclass* object *source* creates under *parent* is given."""
    return frozenset(
        str(rule.default) for rule in setools.TERuleQuery(
            policy, ruletype=[setools.TERuletype.type_transition],
            source=source, target=parent, tclass=[tclass]).results())


def silenced(policy, source: str, target: str, tclass: str,
             perms: FrozenSet[str]) -> bool:
    """*perms* are dontaudit-ed for *source* on *target*:*tclass*."""
    got = set()
    for rule in setools.TERuleQuery(
            policy, ruletype=[setools.TERuletype.dontaudit], source=source,
            target=target, tclass=[tclass], source_indirect=True,
            target_indirect=True).results():
        got |= set(rule.perms)
    return frozenset(perms) <= got


def born_as_named(policy, source: str, parent: str, tclass: str, name: str) -> FrozenSet[str]:
    """The types a *tclass* object named *name* is given when *source*
    creates it under *parent* (a filename type_transition)."""
    out = set()
    for rule in setools.TERuleQuery(
            policy, ruletype=[setools.TERuletype.type_transition],
            source=source, target=parent, tclass=[tclass]).results():
        if getattr(rule, "filename", None) == name:
            out.add(str(rule.default))
    return frozenset(out)


def born_at(policy, source: str, target: str, tclass: str) -> FrozenSet[str]:
    """The ranges a *tclass* object *source* creates in *target* is given."""
    return frozenset(
        str(rule.default) for rule in setools.MLSRuleQuery(
            policy, ruletype=[setools.MLSRuletype.range_transition],
            source=source, target=target, tclass=[tclass]).results())


def _allows(src, tgt, cls, perms):
    perms = frozenset(perms)
    return lambda p: granted(p, src, tgt, cls, perms, conditional=False) == perms


def _forbids(src, tgt, cls, perms):
    perms = frozenset(perms)
    return lambda p: not granted(p, src, tgt, cls, perms)


PTY_USE = {"read", "write", "ioctl", "open"}
ISOLATED = ("jaato_isolated_t", "jaato_isolated_ro_t")
JAATO_DOMAINS = ("jaato_runner_t", "jaato_child_t") + ISOLATED
WRITE = frozenset({"write", "append", "create", "unlink", "rename"})
LOGIN_PTYS = ("user_devpts_t", "sshd_devpts_t")

RULES: Tuple[Rule, ...] = (
    # --- ptys (phase 2a kernel run) -------------------------------------
    Rule("a pty the runner opens is born jaato_devpts_t",
         lambda p: born_as(p, "jaato_runner_t", "devpts_t", "chr_file") == {"jaato_devpts_t"},
         "without the transition the slave is devpts_t and openpty fails "
         "(the 2026-10-02 kernel run); allow rules alone cannot show this",
         find="term_create_pty(jaato_runner_t, jaato_devpts_t)\n"),
    Rule("a pty a child opens is born jaato_devpts_t",
         lambda p: born_as(p, "jaato_child_t", "devpts_t", "chr_file") == {"jaato_devpts_t"},
         "a child running script, expect or ssh -t opens its own",
         find="term_create_pty(jaato_child_t, jaato_devpts_t)\n"),
    Rule("the runner may use a jaato pty",
         _allows("jaato_runner_t", "jaato_devpts_t", "chr_file", PTY_USE),
         "interactive_shell holds the master and talks to the slave",
         find="allow { jaato_runner_t jaato_child_t } jaato_devpts_t:chr_file { rw_term_perms setattr };\n",
         replace="allow jaato_child_t jaato_devpts_t:chr_file { rw_term_perms setattr };\n"),
    Rule("a child may use a jaato pty",
         _allows("jaato_child_t", "jaato_devpts_t", "chr_file", PTY_USE),
         "the shell interactive_shell starts has the slave as its stdio",
         find="allow { jaato_runner_t jaato_child_t } jaato_devpts_t:chr_file { rw_term_perms setattr };\n",
         replace="allow jaato_runner_t jaato_devpts_t:chr_file { rw_term_perms setattr };\n"),
    Rule("neither domain may touch a login terminal",
         lambda p: all(not granted(p, d, t, "chr_file", frozenset(PTY_USE))
                       for d in ("jaato_runner_t", "jaato_child_t") for t in LOGIN_PTYS),
         "login ptys sit at s0, which a runner's level dominates: "
         "term_use_all_ptys would let it write an admin's terminal",
         append="term_use_all_ptys(jaato_runner_t)\n"),
    Rule("neither domain may use other domains' unrelabelled ptys",
         lambda p: all(not granted(p, d, "devpts_t", "chr_file", frozenset(PTY_USE))
                       for d in ("jaato_runner_t", "jaato_child_t")),
         "devpts_t is the pty of every domain without its own type",
         append="term_use_generic_ptys(jaato_child_t)\n"),

    # --- refused quietly (phase 2a kernel run, traced on the import) ----
    # --- binding (parity with AppArmor's `network inet stream`) --------
    Rule("the runner may bind any address",
         _allows("jaato_runner_t", "node_t", "tcp_socket", {"node_bind"}),
         "the webhook plugin listens on its configured host",
         find="corenet_tcp_bind_generic_node(jaato_runner_t)\n"),
    Rule("the runner may bind the webhook's default port 9100",
         _allows("jaato_runner_t", "hplip_port_t", "tcp_socket", {"name_bind"}),
         "9100 is hplip_port_t, a defined port the unreserved set covers",
         find="corenet_tcp_bind_all_unreserved_ports(jaato_runner_t)\n"),
    Rule("a child may bind an unreserved port on any address",
         lambda p: (granted(p, "jaato_child_t", "node_t", "tcp_socket",
                            frozenset({"node_bind"}), conditional=False)
                    and granted(p, "jaato_child_t", "unreserved_port_t", "tcp_socket",
                                frozenset({"name_bind"}), conditional=False)),
         "a test server, `npm run dev`",
         find="corenet_tcp_bind_all_unreserved_ports(jaato_child_t)\n"),
    Rule("neither domain may bind a port below 1024",
         lambda p: all(not granted(p, d, t, "tcp_socket", frozenset({"name_bind"}))
                       for d in ("jaato_runner_t", "jaato_child_t")
                       for t in ("http_port_t", "ssh_port_t", "reserved_port_t")),
         "AppArmor's runner holds no net_bind_service either",
         append="corenet_tcp_bind_http_port(jaato_child_t)\n"),

    # --- the user-global tier, ~/.jaato ---------------------------------
    Rule("both may read the user tier's config",
         lambda p: all(granted(p, d, "jaato_user_config_t", "file",
                               frozenset({"read", "open"}), conditional=False)
                       == {"read", "open"} for d in ("jaato_runner_t", "jaato_child_t")),
         "agents/, profiles/, references/, services/, gc.json (AppArmor plugin grants)",
         find="read_files_pattern({ jaato_runner_t jaato_child_t }, jaato_user_config_t, jaato_user_config_t)\n"),
    Rule("nobody writes the user tier's config",
         lambda p: all(not granted(p, d, "jaato_user_config_t", c,
                                   frozenset({"write", "append", "add_name", "remove_name", "unlink", "rename"}))
                       for d in ("jaato_runner_t", "jaato_child_t") for c in ("file", "dir")),
         "a profile or persona every workspace loads is not a session's to change",
         append="manage_files_pattern(jaato_child_t, jaato_user_config_t, jaato_user_config_t)\n"),
    Rule("both may write the user tier's memories, prompts and skills",
         lambda p: all(granted(p, d, "jaato_user_data_t", c, frozenset(perms), conditional=False)
                       == frozenset(perms)
                       for d in ("jaato_runner_t", "jaato_child_t")
                       for c, perms in (("dir", {"add_name", "remove_name", "write"}),
                                        ("file", {"create", "write", "rename", "unlink"}))),
         "the memory plugin's global tier, prompt_library",
         find="manage_files_pattern({ jaato_runner_t jaato_child_t }, jaato_user_data_t, jaato_user_data_t)\n",
         replace="read_files_pattern({ jaato_runner_t jaato_child_t }, jaato_user_data_t, jaato_user_data_t)\n"),
    Rule("what a runner writes to the user tier is born at s0",
         lambda p: born_at(p, "jaato_runner_t", "jaato_user_data_t", "file") == {"s0"},
         "at the workspace's level another workspace could not read it, and "
         "the global memory tier would split per workspace",
         find="range_transition { jaato_runner_t jaato_child_t } jaato_user_data_t:{ file dir lnk_file } s0;\n"),
    Rule("both may traverse ~/.jaato itself",
         lambda p: all(granted(p, d, "jaato_user_dir_t", "dir",
                               frozenset({"search"}), conditional=False) == {"search"}
                       for d in ("jaato_runner_t", "jaato_child_t")),
         "the user tier's subtrees sit under it",
         find="allow { jaato_runner_t jaato_child_t } jaato_user_dir_t:dir search_dir_perms;\n"),
    Rule("no other directory in a user's home can be traversed",
         lambda p: all(not granted(p, d, "user_home_t", "dir", frozenset({"search"}))
                       for d in ("jaato_runner_t", "jaato_child_t")),
         "a root runner's child walked into /home/<user>/... with only DAC "
         "stopping it (phase 2b kernel run)",
         append="userdom_search_user_home_content(jaato_child_t)\n"),
    Rule("~/.jaato itself cannot be listed",
         lambda p: all(not granted(p, d, "jaato_user_dir_t", "dir", frozenset({"read"}))
                       for d in ("jaato_runner_t", "jaato_child_t")),
         "listing it would enumerate the stored credentials' names",
         append="allow jaato_runner_t jaato_user_dir_t:dir list_dir_perms;\n"),
    Rule("nothing else in a home is readable",
         lambda p: all(not granted(p, d, t, "file", frozenset({"read", "open"}))
                       for d in ("jaato_runner_t", "jaato_child_t")
                       for t in ("user_home_t", "admin_home_t")),
         "~/.jaato/*_auth.json, instructions/, permissions.json keep the home's type",
         append="userdom_read_user_home_content_files(jaato_runner_t)\n"),
    Rule("no home directory can be listed",
         lambda p: all(not granted(p, d, t, "dir", frozenset({"read"}))
                       for d in ("jaato_runner_t", "jaato_child_t")
                       for t in ("user_home_dir_t", "user_home_t", "admin_home_t")),
         "search reaches the granted subtrees; read would enumerate the home",
         append="userdom_list_user_home_dirs(jaato_child_t)\n"),
    Rule("cryptography's cgroup read is refused without an AVC",
         lambda p: all(silenced(p, d, "cgroup_t", "dir", {"search"})
                       and not granted(p, d, "cgroup_t", "dir", frozenset({"search"}))
                       for d in ("jaato_runner_t", "jaato_child_t")),
         "its thread pool falls back to the affinity mask",
         find="fs_dontaudit_search_cgroup_dirs(jaato_runner_t)\n"),

    # --- the version marker the readiness check probes -----------------
    Rule("marker type jaato_policy_v4_t exists",
         lambda p: type_exists(p, "jaato_policy_v4_t"),
         "SELinuxBackend refuses a host whose module lacks the marker",
         find="type jaato_policy_v4_t;\nfiles_type(jaato_policy_v4_t)",
         replace="type jaato_policy_v3_t;\nfiles_type(jaato_policy_v3_t)"),

    # --- a pool slot enters the runner at fork (phase 4) ---------------
    Rule("a daemon in unconfined_t may move a forked slot into the runner",
         _allows("unconfined_t", "jaato_runner_t", "process", {"dyntransition"}),
         "targeted grants it only under unconfined_dyntrans_all",
         find="allow unconfined_t jaato_runner_t:process dyntransition;\n"),
    Rule("a daemon in unconfined_service_t may move a forked slot into the runner",
         _allows("unconfined_service_t", "jaato_runner_t", "process", {"dyntransition"}),
         "targeted does not grant it; a systemd daemon's slots could not enter",
         find="allow unconfined_service_t jaato_runner_t:process dyntransition;\n"),
    Rule("no jaato domain may dyntransition into the runner",
         lambda p: all(not granted(p, d, "jaato_runner_t", "process",
                                   frozenset({"dyntransition"}))
                       for d in ("jaato_child_t",) + ISOLATED),
         "a child or an isolated sub-runner could otherwise become the runner",
         append="allow jaato_child_t jaato_runner_t:process dyntransition;\n"),

    # --- entering the runner and //child -------------------------------
    Rule("the runner may exec-transition its children into jaato_child_t",
         _allows("jaato_runner_t", "jaato_child_t", "process", {"transition"}),
         "without it every cli / shell / kernel spawn fails closed",
         find="allow jaato_runner_t jaato_child_t:process transition;\n"),
    Rule("the runner may set its children's exec context",
         _allows("jaato_runner_t", "jaato_runner_t", "process", {"setexec"}),
         "setexeccon() in the preexec_fn needs it",
         find="allow jaato_runner_t self:process setexec;\n"),
    Rule("bin_t is an entrypoint of jaato_runner_t",
         _allows("jaato_runner_t", "bin_t", "file", {"entrypoint"}),
         "the daemon execs the interpreter, which is bin_t",
         find="corecmd_bin_entry_type(jaato_runner_t)\n"),
    Rule("shell_exec_t is an entrypoint of jaato_child_t",
         _allows("jaato_child_t", "shell_exec_t", "file", {"entrypoint"}),
         "cli runs /bin/sh -c, the first exec after setexeccon",
         find="corecmd_shell_entry_type(jaato_child_t)\n"),
    Rule("a managed workspace's binaries are an entrypoint of jaato_child_t",
         _allows("jaato_child_t", "jaato_managed_ws_t", "file", {"entrypoint"}),
         "node_modules/.bin and the tool venv (#1273/#1274)",
         find="allow jaato_child_t jaato_managed_ws_t:file entrypoint;\n"),
    Rule("system_r is authorized for jaato_runner_t",
         lambda p: "jaato_runner_t" in role_types(p, "system_r"),
         "a daemon under systemd keeps system_r (phase 0)",
         find="role system_r types { jaato_runner_t jaato_child_t jaato_isolated_t jaato_isolated_ro_t };\n"),
    Rule("unconfined_r is authorized for jaato_runner_t",
         lambda p: "jaato_runner_t" in role_types(p, "unconfined_r"),
         "a daemon started from a login shell keeps unconfined_r",
         find="role unconfined_r types { jaato_runner_t jaato_child_t jaato_isolated_t jaato_isolated_ro_t };\n"),
    Rule("jaato_runner_t is MCS-constrained",
         lambda p: "mcs_constrained_type" in type_attrs(p, "jaato_runner_t"),
         "without it the level separates nothing on targeted",
         find="mcs_constrained(jaato_runner_t)\n"),
    Rule("jaato_child_t is MCS-constrained",
         lambda p: "mcs_constrained_type" in type_attrs(p, "jaato_child_t"),
         "a child must not reach another workspace either",
         find="mcs_constrained(jaato_child_t)\n"),

    # --- inherited descriptors (phase 0: the silent fd drop) -----------
    Rule("the runner may use descriptors from a systemd daemon",
         _allows("jaato_runner_t", "unconfined_service_t", "fd", {"use"}),
         "the RPC socketpair and stdio come from the daemon",
         find="allow jaato_runner_t { unconfined_t unconfined_service_t }:fd use;\n"),
    Rule("the runner may use the daemon's socketpair",
         _allows("jaato_runner_t", "unconfined_service_t", "unix_stream_socket",
                 {"read", "write"}),
         "the RPC channel is a unix stream socket the daemon created",
         find="allow jaato_runner_t { unconfined_t unconfined_service_t }:"
              "unix_stream_socket { getattr getopt setopt read write shutdown };\n"),
    Rule("a child may use the runner's descriptors",
         _allows("jaato_child_t", "jaato_runner_t", "fd", {"use"}),
         "a child's stdout is a pipe the runner created",
         find="allow jaato_child_t jaato_runner_t:fd use;\n"),

    # --- file types ------------------------------------------------------
    Rule("a child may write the workspace",
         _allows("jaato_child_t", "jaato_workspace_t", "file",
                 {"write", "create", "unlink", "rename"}),
         "the workspace is the session's to change",
         find="manage_files_pattern(jaato_child_t, jaato_workspace_t, jaato_workspace_t)\n"),
    Rule("a child may execute from a managed workspace",
         _allows("jaato_child_t", "jaato_managed_ws_t", "file",
                 {"execute", "execute_no_trans"}),
         "what a managed project builds runs (#1273)",
         find="allow { jaato_runner_t jaato_child_t } jaato_managed_ws_t:file "
              "{ lock map execute execute_no_trans };\n"),
    Rule("a child may read agent config",
         _allows("jaato_child_t", "jaato_agent_config_t", "file", {"read", "open"}),
         "personas, profiles and scripts are read by the session",
         find="read_files_pattern({ jaato_runner_t jaato_child_t }, "
              "jaato_agent_config_t, jaato_agent_config_t)\n"),
    Rule("the runner and its children may write the prompt library",
         _allows("jaato_child_t", "jaato_prompts_t", "file", {"write", "create", "unlink"}),
         "prompt_library's savePrompt/deletePrompt (AppArmor: write-allowed)",
         find="manage_files_pattern({ jaato_runner_t jaato_child_t }, "
              "jaato_prompts_t, jaato_prompts_t)\n"),
    Rule("a child may read authored config",
         _allows("jaato_child_t", "jaato_authored_t", "file", {"read", "open"}),
         "profiles, agents and scripts are read by the session",
         find="read_files_pattern({ jaato_runner_t jaato_child_t }, "
              "jaato_authored_t, jaato_authored_t)\n"),
    Rule("the runner may write reference claims",
         _allows("jaato_runner_t", "jaato_claims_t", "file", {"write", "create"}),
         "proposeReference runs in the runner",
         find="manage_files_pattern(jaato_runner_t, jaato_claims_t, jaato_claims_t)\n"),
    Rule("a child may write the session tmpdir",
         _allows("jaato_child_t", "jaato_tmp_t", "file", {"write", "create"}),
         "TMPDIR and the private /tmp",
         find="manage_files_pattern({ jaato_runner_t jaato_child_t }, "
              "jaato_tmp_t, jaato_tmp_t)\n"),

    # --- the isolated sub-runner (phase 3, design §5.3) -----------------
    # A transition rule from the daemon is not asserted: targeted lets
    # unconfined_t transition into every domain (phase 0), so it would be
    # decorative. The role and the entrypoint are the halves only this
    # module grants, as the readiness check says.
    Rule("system_r is authorized for the isolated domains",
         lambda p: set(ISOLATED) <= role_types(p, "system_r"),
         "a daemon under systemd keeps system_r",
         find="role system_r types { jaato_runner_t jaato_child_t jaato_isolated_t "
              "jaato_isolated_ro_t };\n",
         replace="role system_r types { jaato_runner_t jaato_child_t };\n"),
    Rule("bin_t is an entrypoint of jaato_isolated_t",
         _allows("jaato_isolated_t", "bin_t", "file", {"entrypoint"}),
         "the daemon execs the interpreter into the isolated domain",
         find="corecmd_bin_entry_type(jaato_isolated_t)\n"),
    Rule("bin_t is an entrypoint of jaato_isolated_ro_t",
         _allows("jaato_isolated_ro_t", "bin_t", "file", {"entrypoint"}),
         "the read-only variant is entered the same way",
         find="corecmd_bin_entry_type(jaato_isolated_ro_t)\n"),
    Rule("jaato_isolated_t is MCS-constrained",
         lambda p: "mcs_constrained_type" in type_attrs(p, "jaato_isolated_t"),
         "its parent's level must be the only workspace it reaches",
         find="mcs_constrained(jaato_isolated_t)\n"),
    Rule("jaato_isolated_ro_t is MCS-constrained",
         lambda p: "mcs_constrained_type" in type_attrs(p, "jaato_isolated_ro_t"),
         "the read-only variant too",
         find="mcs_constrained(jaato_isolated_ro_t)\n"),
    Rule("the isolated domains may use the daemon's descriptors",
         lambda p: all(granted(p, d, "unconfined_service_t", "fd", frozenset({"use"}),
                               conditional=False) == {"use"} for d in ISOLATED),
         "the RPC socketpair comes from the daemon (phase 0: silent fd drop)",
         find="allow { jaato_isolated_t jaato_isolated_ro_t } "
              "{ unconfined_t unconfined_service_t }:fd use;\n"),
    Rule("jaato_isolated_t may write its parent's workspace",
         _allows("jaato_isolated_t", "jaato_workspace_t", "file", {"write", "create", "unlink"}),
         "AppArmor: \"<ws>/**\" rwkl in the sub-profile",
         find="manage_files_pattern(jaato_isolated_t, { jaato_workspace_t "
              "jaato_managed_ws_t }, { jaato_workspace_t jaato_managed_ws_t })\n"),
    Rule("jaato_isolated_ro_t may read the workspace",
         _allows("jaato_isolated_ro_t", "jaato_managed_ws_t", "file", {"read", "open"}),
         "isolated_read_only_workspace downgrades rwkl to r, not to nothing",
         find="read_files_pattern(jaato_isolated_ro_t, { jaato_workspace_t "
              "jaato_managed_ws_t }, { jaato_workspace_t jaato_managed_ws_t })\n"),
    Rule("jaato_isolated_ro_t cannot write the workspace",
         lambda p: all(not granted(p, "jaato_isolated_ro_t", t, "file", WRITE)
                       and not granted(p, "jaato_isolated_ro_t", t, "dir",
                                       frozenset({"add_name", "remove_name", "write"}))
                       for t in ("jaato_workspace_t", "jaato_managed_ws_t")),
         "the read-only tightening",
         append="allow jaato_isolated_ro_t jaato_workspace_t:file write;\n"),
    Rule("the isolated domains may read the shared authored config",
         lambda p: all(granted(p, d, "jaato_authored_t", "file",
                               frozenset({"read", "open"}), conditional=False)
                       == {"read", "open"} for d in ISOLATED),
         "AppArmor leaves references, templates, services and plans readable",
         find="read_files_pattern({ jaato_isolated_t jaato_isolated_ro_t }, "
              "jaato_authored_t, jaato_authored_t)\n"),
    Rule("the isolated domains cannot read agent config",
         lambda p: all(not granted(p, d, "jaato_agent_config_t", c,
                                   frozenset({"read", "open", "getattr", "search"}))
                       for d in ISOLATED for c in ("file", "dir")),
         "AppArmor read-denies personas, profiles, scripts, schemas, instructions",
         append="allow jaato_isolated_t jaato_agent_config_t:file { read open };\n"),
    Rule("the isolated domains cannot read the prompt library",
         lambda p: all(not granted(p, d, "jaato_prompts_t", c,
                                   frozenset({"read", "open", "getattr", "search"}))
                       for d in ISOLATED for c in ("file", "dir")),
         "AppArmor read-denies .jaato/prompts/ to the sub-profile",
         append="allow jaato_isolated_ro_t jaato_prompts_t:file { read open };\n"),
    Rule("the isolated domains reach no user-tier files",
         lambda p: all(not granted(p, d, t, c, frozenset({"read", "open", "search", "write"}))
                       for d in ISOLATED
                       for t in ("jaato_user_dir_t", "jaato_user_config_t",
                                 "jaato_user_data_t")
                       for c in ("file", "dir")),
         "the sub-profile drops ~/.jaato; its user tier rides the envelope",
         append="allow jaato_isolated_t jaato_user_data_t:file { read open };\n"),
    # ``execute`` + ``map`` on bin_t stay: after the exec transition the
    # kernel maps the entry binary (the interpreter) under the NEW domain,
    # so a domain that cannot execute its own entrypoint cannot start.
    # What it must not have is a way to run anything else: no
    # execute_no_trans on any type, no execute beyond the entry type, and
    # no transition into another domain.  Counted over the rules this
    # module writes for the domain itself: targeted gives EVERY ``domain``
    # prelink_exec_t execute (only under fips_mode) and a transition to
    # abrt_helper_t (inert without execute on abrt_helper_exec_t), which
    # the runner and the child carry too and no module can take away.
    Rule("the isolated domains execute nothing but their own entrypoint",
         lambda p: all(not _granted_directly(p, d, "file", {"execute_no_trans"})
                       and not _granted_directly(p, d, "process", {"transition"})
                       and all(not granted(p, d, t, "file", frozenset({"execute"}))
                               for t in ("shell_exec_t", "jaato_managed_ws_t",
                                         "jaato_workspace_t", "jaato_tmp_t"))
                       for d in ISOLATED),
         "the flat sub-profile grants no exec outside the venv's bin/",
         append="allow jaato_isolated_t shell_exec_t:file { execute execute_no_trans };\n"),
    Rule("the isolated domains cannot set an exec context or change domain",
         lambda p: all(not granted(p, d, d, "process",
                                   frozenset({"setexec", "dyntransition", "setcurrent"}))
                       for d in ISOLATED),
         "the sub-profile: DROP change_profile transitions",
         append="allow jaato_isolated_t self:process dyntransition;\n"),
    Rule("jaato_isolated_t may write reference claims",
         _allows("jaato_isolated_t", "jaato_claims_t", "file", {"write", "create"}),
         "claims sit inside the workspace the sub-profile grants rwkl",
         find="manage_files_pattern(jaato_isolated_t, jaato_claims_t, jaato_claims_t)\n"),
    Rule("jaato_isolated_ro_t cannot write reference claims",
         _forbids("jaato_isolated_ro_t", "jaato_claims_t", "file", WRITE),
         "the read-only tightening covers the whole workspace",
         append="allow jaato_isolated_ro_t jaato_claims_t:file create;\n"),
    Rule("both isolated domains may append to their log",
         lambda p: all(granted(p, d, "jaato_runner_log_t", "file",
                               frozenset({"open", "append"}), conditional=False)
                       == {"open", "append"} for d in ISOLATED),
         "a read-only sub-runner that cannot write its log runs blind",
         find="allow { jaato_isolated_t jaato_isolated_ro_t } jaato_runner_log_t:file "
              "{ getattr open append ioctl lock };\n"),
    Rule("the isolated domains cannot rewrite or remove their log",
         lambda p: all(not granted(p, d, "jaato_runner_log_t", "file",
                                   frozenset({"write", "create", "unlink", "rename",
                                              "setattr", "relabelfrom"}))
                       for d in ISOLATED),
         "append only: a sub-runner may add to its log, not erase what it said",
         append="allow jaato_isolated_ro_t jaato_runner_log_t:file write;\n"),
    Rule("both isolated domains may write the session tmpdir",
         lambda p: all(granted(p, d, "jaato_tmp_t", "file", frozenset({"write", "create"}),
                               conditional=False) == {"write", "create"} for d in ISOLATED),
         "AppArmor keeps /tmp/jaato-*/** rwkl in the read-only variant too",
         find="manage_files_pattern({ jaato_isolated_t jaato_isolated_ro_t }, "
              "jaato_tmp_t, jaato_tmp_t)\n"),
    Rule("no domain may transition into the isolated domains",
         lambda p: all(not granted(p, d, i, "process",
                                   frozenset({"transition", "dyntransition"}))
                       for d in ("jaato_runner_t", "jaato_child_t") for i in ISOLATED),
         "only the daemon starts a sub-runner",
         append="allow jaato_runner_t jaato_isolated_t:process transition;\n"),

    # --- what must stay absent -----------------------------------------
    Rule("a child cannot set an exec context or change its domain",
         _forbids("jaato_child_t", "jaato_child_t", "process",
                  {"setexec", "dyntransition", "setcurrent"}),
         "a child could otherwise leave //child (#1323)",
         append="allow jaato_child_t self:process setexec;\n"),
    Rule("the runner cannot change its own domain",
         _forbids("jaato_runner_t", "jaato_runner_t", "process",
                  {"dyntransition", "setcurrent"}),
         "nothing in the runner may leave jaato_runner_t (§4.4)",
         append="allow jaato_runner_t self:process setcurrent;\n"),
    Rule("neither domain may transition to unconfined_t",
         lambda p: not granted(p, "jaato_runner_t", "unconfined_t", "process",
                               frozenset({"transition", "dyntransition"}))
         and not granted(p, "jaato_child_t", "unconfined_t", "process",
                         frozenset({"transition", "dyntransition"})),
         "the AppArmor base keeps change_profile -> unconfined; this must not",
         append="allow jaato_child_t unconfined_t:process transition;\n"),
    Rule("a child cannot read the runner's /proc entries",
         _forbids("jaato_child_t", "jaato_runner_t", "file", {"read", "open"}),
         "the runner's environ holds the provider credential",
         append="allow jaato_child_t jaato_runner_t:file { read open };\n"),
    Rule("a missing .jaato/reactors.json is born agent config, so it cannot be created",
         lambda p: all(born_as_named(p, d, ws, "file", "reactors.json") == {"jaato_agent_config_t"}
                       for d in ("jaato_runner_t", "jaato_child_t", "jaato_isolated_t")
                       for ws in ("jaato_workspace_t", "jaato_managed_ws_t")),
         "AppArmor denies the path whether or not the file exists",
         find='type_transition { jaato_runner_t jaato_child_t jaato_isolated_t } '
              '{ jaato_workspace_t jaato_managed_ws_t }:file jaato_agent_config_t '
              '"reactors.json";\n'),
    Rule("a missing .jaato/template_routing.yaml is born authored, so it cannot be created",
         lambda p: all(born_as_named(p, d, ws, "file", "template_routing.yaml") == {"jaato_authored_t"}
                       for d in ("jaato_runner_t", "jaato_child_t", "jaato_isolated_t")
                       for ws in ("jaato_workspace_t", "jaato_managed_ws_t")),
         "where a rendered template lands is authored config",
         find='type_transition { jaato_runner_t jaato_child_t jaato_isolated_t } '
              '{ jaato_workspace_t jaato_managed_ws_t }:file jaato_authored_t '
              '"template_routing.yaml";\n'),
    Rule("no jaato domain may create an authored or agent-config file",
         lambda p: all(not granted(p, d, t, "file", frozenset({"create"}))
                       for d in JAATO_DOMAINS
                       for t in ("jaato_authored_t", "jaato_agent_config_t")),
         "the filename transitions refuse a creation only because of this",
         append="allow jaato_child_t jaato_authored_t:file create;\n"),
    Rule("nobody writes, renames or removes authored files",
         lambda p: all(not granted(p, d, "jaato_authored_t", "file",
                                   frozenset({"write", "append", "create", "unlink",
                                              "rename", "setattr", "link"}))
                       for d in ("jaato_runner_t", "jaato_child_t")),
         "authored config is write-denied (AppArmor: audit deny ... wlk)",
         append="allow jaato_runner_t jaato_authored_t:file write;\n"),
    Rule("nobody adds to, renames or removes authored directories",
         lambda p: all(not granted(p, d, "jaato_authored_t", "dir",
                                   frozenset({"write", "add_name", "remove_name",
                                              "rename", "reparent", "rmdir",
                                              "setattr", "create"}))
                       for d in ("jaato_runner_t", "jaato_child_t")),
         "moving .jaato/agents away and making a fresh one must fail (§6)",
         append="allow jaato_child_t jaato_authored_t:dir rename;\n"),
    Rule("a child cannot write reference claims",
         _forbids("jaato_child_t", "jaato_claims_t", "file",
                  {"write", "append", "create", "unlink", "rename"}),
         "a shell command could forge witnessed_by (template v43)",
         append="allow jaato_child_t jaato_claims_t:file create;\n"),
    Rule("nothing executes from a user's own checkout",
         lambda p: all(not granted(p, d, "jaato_workspace_t", "file",
                                   frozenset({"execute", "execute_no_trans",
                                              "entrypoint"}))
                       for d in ("jaato_runner_t", "jaato_child_t")),
         "only a managed workspace is executable",
         append="allow jaato_child_t jaato_workspace_t:file execute;\n"),
    Rule("neither domain holds any capability",
         lambda p: all(not granted(p, d, d, c, class_perms(p, c))
                       for d in ("jaato_runner_t", "jaato_child_t")
                       for c in ("capability", "cap_userns", "capability2")),
         "no sys_admin, net_admin, sys_ptrace, dac_override, ...",
         append="allow jaato_runner_t self:capability sys_admin;\n"),
    Rule("neither domain may mount",
         lambda p: all(not granted(p, d, None, "filesystem",
                                   frozenset({"mount", "remount", "unmount"}))
                       and not granted(p, d, None, "dir", frozenset({"mounton"}))
                       for d in JAATO_DOMAINS),
         "AppArmor: deny mount",
         append="allow jaato_child_t jaato_tmp_t:filesystem mount;\n"),
    Rule("neither domain may open raw or packet sockets",
         lambda p: all(not granted(p, d, d, c, frozenset({"create"}))
                       for d in JAATO_DOMAINS
                       for c in ("rawip_socket", "packet_socket")),
         "AppArmor: deny network raw",
         append="allow jaato_child_t self:rawip_socket create;\n"),
    Rule("neither domain may ptrace",
         lambda p: all(not granted(p, d, t, "process", frozenset({"ptrace"}))
                       for d in JAATO_DOMAINS
                       for t in ("jaato_runner_t", "jaato_child_t",
                                 "unconfined_t", "unconfined_service_t")),
         "AppArmor: deny ptrace",
         append="allow jaato_child_t jaato_runner_t:process ptrace;\n"),
    Rule("neither domain may create files in the host's /tmp",
         lambda p: all(not granted(p, d, "tmp_t", "dir",
                                   frozenset({"add_name", "write"}))
                       for d in ("jaato_runner_t", "jaato_child_t")),
         "scratch goes to the session tmpdir or the private /tmp (#1381)",
         append="allow jaato_child_t tmp_t:dir { write add_name };\n"),
)


def _edited(rule: Rule, te: str) -> str:
    if rule.append is not None:
        return te + "\n" + rule.append
    assert te.count(rule.find) == 1, (
        f"reversion anchor for {rule.name!r} occurs {te.count(rule.find)} "
        "times in jaato.te (need exactly 1): the module moved and this "
        "reversion is stale")
    return te.replace(rule.find, rule.replace)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

@pytest.mark.parametrize("rule", RULES, ids=[r.name for r in RULES])
def test_the_shipped_module_has_the_property(rule: Rule):
    policy = build(shipped_te())
    assert rule.holds(policy), f"{rule.name}: {rule.because}"


@pytest.mark.parametrize("rule", RULES, ids=[r.name for r in RULES])
def test_each_assertion_detects_its_reversion(rule: Rule):
    """The edit that removes the property must make its assertion fail.

    An assertion that still holds after its own line is removed is
    satisfied by the base policy, and proves nothing about this module.
    """
    policy = build(_edited(rule, shipped_te()))
    assert not rule.holds(policy), (
        f"DECORATIVE: {rule.name!r} still holds after its reversion; the "
        "base policy satisfies it regardless of jaato.te")
