"""A provider that names its stored-credential command must have one.

THE DEFECT (#888).  ``nebius``, ``ovhcloud`` and ``doubleword`` each
declared::

    AuthSource("stored", "nebius-auth",
               "nebius_auth.json (config_root -> workspace -> ~/.jaato)")

and no plugin registered ``nebius-auth``.  The declaration is not inert
prose: ``shared/scaffold/introspect.py`` reads
``PROVIDER_AUTH_RESOLUTION`` into the model ``explain provider <name>``
renders, so the framework's own derived surface advertised a command that
did not exist.  The providers' ``provider.py`` went further and told a
user holding a rejected key to *run* it::

    "Run 'ovhcloud-auth key <your_api_key>' to re-authenticate, "

A builder following that instruction reached a command the daemon had
never heard of.

WHY IT WAS INVISIBLE.  The *reader* half of the tier works: each
provider's ``env.resolve_api_key`` falls through to
``auth.get_stored_api_key``, and each ``auth.py`` is complete
(``validate_api_key`` / ``login_with_key`` / ``clear_credentials``).  So
the stored tier resolved correctly for anyone who wrote the JSON by
hand, and nothing anywhere failed.  Only the *writer* was missing.

WHAT THIS GUARD ASSERTS, AND WHY IT IS DERIVED RATHER THAN LISTED.  The
check walks every provider's declaration and requires a live plugin to
provide each named ``stored`` command.  Both sides are read from the
tree — the providers' own ``PROVIDER_AUTH_RESOLUTION`` and the
``*_auth`` plugins' own ``get_user_commands()`` — so a provider added
next year is covered without anyone remembering to extend a list here.
A hardcoded roster of three names would have passed on the day #888 was
filed, because the three names were exactly what nobody had written
down.

Deliberately one-directional: it requires a DECLARED command to exist,
never that every auth plugin be declared by some provider.  An auth
plugin serving a provider that resolves credentials another way is not
a defect, and asserting the converse would invent one.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from shared.tests.reversion import Reversion

#: The defect, put back: the plugin exists but stops answering to the
#: name its provider advertises.  That is #888 exactly — a declaration
#: naming a command nothing provides — and the guard must notice without
#: the directory being deleted, which a find/replace cannot do.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/plugins/nebius_auth/plugin.py",
        find='COMMAND = "nebius-auth"',
        replace='COMMAND = "nebius-auth-unregistered"',
        test="TestDeclaredCommandsExist::test_every_declared_stored_command_is_provided",
        because="a provider naming a stored-credential command nothing registers",
    ),
]

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins"
PROVIDER_DIR = PLUGIN_DIR / "model_provider"

#: Same exclusions the sibling provider guards use — test stubs and
#: non-providers.
_EXCLUDE = {"tests", "__pycache__", "bundle_common", "echo"}


def _provider_dirs() -> List[str]:
    out = []
    for entry in sorted(os.listdir(PROVIDER_DIR)):
        d = PROVIDER_DIR / entry
        if not d.is_dir() or entry in _EXCLUDE or entry.startswith("_"):
            continue
        if (d / "__init__.py").exists():
            out.append(entry)
    return out


def _assign_value(provider: str, name: str) -> Optional[ast.expr]:
    """The AST value node assigned to module-level ``name``, or None.

    AST rather than import, for the reason the sibling provider guards
    give: a provider whose vendor SDK is not installed must not break a
    contract check that never needed the SDK.
    """
    src = (PROVIDER_DIR / provider / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if name in targets:
                return node.value
    return None


def _names_a_command(value: str) -> bool:
    """Whether a ``stored`` AuthSource value names a command to RUN.

    The ``stored`` kind carries two different things, and conflating
    them would make this guard invent defects.  Nine declarations name a
    command a user types (``nebius-auth``); four name the LOCATION a
    credential is read from and have no command at all:

        ``openai_auth.json``   ``azure_openai_auth.json``
        ``~/.aws/credentials`` ``GOOGLE_APPLICATION_CREDENTIALS``

    Those four are correct as they stand — Bedrock resolves through
    botocore's own chain and Google through ADC, so there is nothing for
    a ``*-auth`` command to write.  A command is therefore recognised
    structurally: a bare token, no path separator and no filename
    extension, that ends in the ``-auth`` suffix every such command in
    this tree uses.
    """
    if "/" in value or "." in value or value != value.lower():
        return False
    return value.endswith("-auth")


def declared_stored_commands() -> Dict[str, str]:
    """Every ``AuthSource("stored", "<command>")`` in the tree.

    Returns:
        ``{command: provider}`` — the command a provider tells its users
        to run, mapped to the provider that says so.  Location-shaped
        declarations are excluded; see :func:`_names_a_command`.
    """
    found: Dict[str, str] = {}
    for provider in _provider_dirs():
        node = _assign_value(provider, "PROVIDER_AUTH_RESOLUTION")
        if node is None or not isinstance(node, (ast.Tuple, ast.List)):
            continue
        for element in node.elts:
            if not isinstance(element, ast.Call):
                continue
            args = element.args
            if len(args) < 2:
                continue
            kind, value = args[0], args[1]
            if not (isinstance(kind, ast.Constant) and kind.value == "stored"):
                continue
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                if value.value and _names_a_command(value.value):
                    found[value.value] = provider
    return found


def provided_commands() -> Set[str]:
    """Every user command the ``*_auth`` plugins actually register.

    Read from the plugins themselves — instantiate and ask — rather
    than from a list in this file, so the guard measures what a daemon
    would load.
    """
    import importlib

    commands: Set[str] = set()
    for entry in sorted(os.listdir(PLUGIN_DIR)):
        if not entry.endswith("_auth") or not (PLUGIN_DIR / entry).is_dir():
            continue
        module = importlib.import_module(f"shared.plugins.{entry}")
        factory = getattr(module, "create_plugin", None)
        if factory is None:
            continue
        plugin = factory()
        for command in plugin.get_user_commands():
            name = getattr(command, "name", None)
            if name:
                commands.add(name)
    return commands


def _missing() -> List[Tuple[str, str]]:
    provided = provided_commands()
    return sorted(
        (command, provider)
        for command, provider in declared_stored_commands().items()
        if command not in provided
    )


class TestDeclaredCommandsExist:
    def test_the_declaration_is_actually_being_read(self):
        """Guard the guard: an empty scan would pass the check below.

        If the AST walk stopped finding declarations -- a renamed
        constant, a moved file -- ``_missing()`` would be empty and the
        real assertion would report success having examined nothing.
        """
        declared = declared_stored_commands()
        assert len(declared) >= 8, (
            f"only {len(declared)} stored-credential commands found across "
            f"{len(_provider_dirs())} providers; the PROVIDER_AUTH_RESOLUTION "
            f"scan has probably stopped matching. Found: {sorted(declared)}"
        )
        assert provided_commands(), (
            "no *_auth plugin registered any user command; the plugin scan "
            "has stopped matching and the check below proves nothing"
        )

    def test_every_declared_stored_command_is_provided(self):
        """#888: a named command a user is told to run must exist."""
        missing = _missing()
        assert missing == [], (
            "provider(s) declare a stored-credential command that no plugin "
            "registers, so `explain provider` advertises it and the "
            "provider's own 401 message tells a user to run it:\n"
            + "\n".join(
                f"  {command!r} declared by model_provider/{provider}/"
                f"__init__.py -- expected a shared/plugins/*_auth/ plugin "
                f"whose get_user_commands() returns it"
                for command, provider in missing
            )
        )

    def test_a_location_is_not_read_as_a_command(self):
        """The four ``stored`` values that name a file, not a command.

        Bedrock resolves through botocore's chain and Google through
        ADC, so neither has -- or should have -- a ``*-auth`` command.
        Reading these as commands would make the guard demand plugins
        that must not exist.
        """
        for location in ("openai_auth.json", "azure_openai_auth.json",
                         "~/.aws/credentials",
                         "GOOGLE_APPLICATION_CREDENTIALS"):
            assert not _names_a_command(location), (
                f"{location!r} names where a credential is READ FROM, not a "
                f"command to run"
            )
        declared = declared_stored_commands()
        assert "openai_auth.json" not in declared
        assert "~/.aws/credentials" not in declared

    def test_the_three_providers_from_the_issue_are_covered(self):
        """The specific regression #888 reported, pinned by name."""
        provided = provided_commands()
        for command in ("nebius-auth", "ovhcloud-auth", "doubleword-auth"):
            assert command in provided, (
                f"{command} is declared by its provider and documented in "
                f"CLAUDE.md, but no plugin registers it (#888)"
            )
