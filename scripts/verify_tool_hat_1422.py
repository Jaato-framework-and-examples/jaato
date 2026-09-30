#!/usr/bin/env python3
"""Verify #1422 (tool bodies in ``tool_hat``) on an ENFORCING AppArmor host.

CI has no AppArmor LSM, so the #1422 tests check rendered text and
injected ``attr/current`` writes.  This script checks what the kernel
does.  It renders and loads a real jaato session profile with
``AppArmorManager``, then runs a probe process that confines itself with
the runner's own ``confine_to_profile`` (bootstrap step 1c) and drives
the production code: ``apparmor_hat.tool_hat`` and a ``ToolExecutor``
carrying ``make_tool_hat_context``.

Run as root from a jaato venv, on the host you want to accept:

    sudo .venv/bin/python scripts/verify_tool_hat_1422.py

Exit 0 when every check passes.  Each check prints PASS/FAIL.  The
profile and the temp tree are removed on exit.

What it checks (the acceptance list of #1422):

  base (the runner's own work, a user command):
    - the thread label is the session profile
    - it CAN write ``.jaato/references/**`` (v44 drops that deny)
    - it cannot write ``.jaato/agents/**`` (integrity deny kept)
  tool_hat (a model-called tool body):
    - the thread label is ``<profile>//tool_hat`` while the body runs
    - it cannot write ``.jaato/references/**``
    - it cannot read ``.jaato/agents/**`` / ``profiles`` / ``scripts``
    - it can write an ordinary workspace file
    - a subprocess it spawns with the //child callback reports ``//child``
    - the thread is back in the base profile afterwards
  ToolExecutor:
    - 8 parallel tool bodies each see the hat and each thread returns
    - a ``base_profile_calls`` user command writes the catalog
  #1023:
    - ``scan_thread_profiles`` finds no divergence after all of the above,
      including a thread created inside the hat
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _probe(profile: str, ws: str) -> int:
    """Runs in the child process: confine, then check."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from jaato_server.server.apparmor import make_child_transition_callback
    from jaato_server.server.runner.bootstrap import (
        confine_to_profile, scan_thread_profiles,
    )
    from jaato_server.shared import apparmor_hat as H
    from jaato_server.shared.ai_tool_runner import ToolExecutor

    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    def label():
        with open(H.thread_attr_path()) as f:
            return f.read().replace("\x00", "").strip()

    def can_write(path):
        try:
            with open(path, "w") as f:
                f.write("x")
            return True
        except OSError:
            return False

    def can_read(path):
        try:
            with open(path) as f:
                f.read()
            return True
        except OSError:
            return False

    j = os.path.join(ws, ".jaato")
    confine_to_profile(profile, require_enforce=True)
    check("base: label is the session profile", label().startswith(profile + " "), label())
    check("base: can write .jaato/references/**", can_write(f"{j}/references/base.json"))
    check("base: cannot write .jaato/agents/**", not can_write(f"{j}/agents/a.md"))

    def in_hat():
        with H.tool_hat(profile):
            check("hat: label is //tool_hat", label().startswith(H.hat_profile(profile)), label())
            check("hat: cannot write .jaato/references/**",
                  not can_write(f"{j}/references/hat.json"))
            for sub in ("agents/a.md", "profiles/p.yaml", "scripts/s.py"):
                check(f"hat: cannot read .jaato/{sub}", not can_read(f"{j}/{sub}"))
            check("hat: can write a workspace file", can_write(f"{ws}/out.txt"))
            out = subprocess.run(
                ["/bin/cat", "/proc/self/attr/current"], capture_output=True,
                text=True, preexec_fn=make_child_transition_callback(profile),
            ).stdout
            check("hat: a spawned subprocess is in //child", "//child" in out, out)
            born = []
            t = threading.Thread(target=lambda: born.append(label()))
            t.start()
            t.join()
            check("hat: a thread created in the hat inherits it",
                  born and "//tool_hat" in born[0], str(born))
        check("hat: thread is back in base afterwards",
              label().startswith(profile + " "), label())

    t = threading.Thread(target=in_hat)
    t.start()
    t.join()

    e = ToolExecutor()
    e.set_apparmor_context(H.make_tool_hat_context(profile))
    seen = []
    e.register("probe", lambda a: seen.append((label(), label())) or {"ok": True})
    e.register("references", lambda a: {"ok": can_write(f"{j}/references/cmd.json")})
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: e.execute("probe", {}), range(8)))
        after = list(pool.map(lambda _: label(), range(8)))
    check("executor: 8 parallel bodies ran in the hat",
          len(seen) == 8 and all("//tool_hat" in s[0] for s in seen), str(seen))
    check("executor: every worker returned to base",
          all(a.startswith(profile + " ") for a in after), str(after))
    with e.base_profile_calls():
        ok, res = e.execute("references", {})
    check("executor: a user command writes the catalog in base", ok and res["ok"], str(res))

    scan = scan_thread_profiles(profile, stuck_hat_tids=H.stuck_hat_tids())
    check("#1023: no divergence", not scan.divergent, scan.summary())

    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if not ok and detail else ""))
    return 0 if all(ok for _, ok, _ in results) else 1


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--probe":
        return _probe(sys.argv[2], sys.argv[3])
    if os.geteuid() != 0:
        print("must run as root (apparmor_parser)", file=sys.stderr)
        return 2
    from jaato_server.server.apparmor import AppArmorManager

    tmp = Path(tempfile.mkdtemp(prefix="jaato-1422-"))
    root = tmp / "workspaces"
    ws = root / "sessions" / "verify"
    for sub in ("agents", "profiles", "scripts", "references", "sessions"):
        (ws / ".jaato" / sub).mkdir(parents=True)
    (ws / ".jaato/agents/a.md").write_text("persona")
    (ws / ".jaato/profiles/p.yaml").write_text("name: p")
    (ws / ".jaato/scripts/s.py").write_text("pass")
    os.chmod(tmp, 0o755)
    manager = AppArmorManager(workspace_root=str(root), profile_dir=str(tmp / "profiles"))
    session = "verify1422"
    if not manager.provision_profile(session, str(ws)):
        print("could not load the profile", file=sys.stderr)
        return 2
    profile = manager.get_profile_name(session)
    try:
        print(f"template v{manager._TEMPLATE_VERSION}, profile {profile}")
        return subprocess.run(
            [sys.executable, __file__, "--probe", profile, str(ws)],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        ).returncode
    finally:
        manager.teardown_profile(session)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
