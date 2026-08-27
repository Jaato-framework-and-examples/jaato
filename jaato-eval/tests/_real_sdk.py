"""Load the REAL ``jaato_sdk.completion_processors`` from the repo checkout.

``jaato_sdk/__init__.py`` pulls in pydantic, which unit-test environments
need not have. But ``completion_processors`` itself is stdlib-only, so it
can be loaded by path and registered under its real name.

This matters: it means ``tests/test_ledger.py`` exercises the pairing rule
the SDK actually ships (jaato #640) rather than a stub that agrees with
whatever this package expects. A stub would keep passing if the real rule
changed underneath — precisely the rot the SDK builder exists to prevent.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

#: jaato-eval/tests/_real_sdk.py -> repo root -> jaato-sdk/...
_MODULE = (Path(__file__).resolve().parents[2]
           / "jaato-sdk" / "jaato_sdk" / "completion_processors.py")


def install() -> bool:
    """Register the real module as ``jaato_sdk.completion_processors``.

    Returns:
        ``True`` when it was loaded, ``False`` when the checkout is not
        laid out as expected — callers skip rather than silently testing
        nothing.
    """
    if "jaato_sdk.completion_processors" in sys.modules:
        return True
    if not _MODULE.is_file():
        return False

    spec = importlib.util.spec_from_file_location(
        "jaato_sdk.completion_processors", _MODULE)
    if spec is None or spec.loader is None:
        return False
    module = importlib.util.module_from_spec(spec)

    parent = sys.modules.get("jaato_sdk")
    if parent is None:
        parent = types.ModuleType("jaato_sdk")
        parent.__path__ = [str(_MODULE.parent)]
        sys.modules["jaato_sdk"] = parent

    sys.modules["jaato_sdk.completion_processors"] = module
    spec.loader.exec_module(module)
    parent.completion_processors = module
    return True
