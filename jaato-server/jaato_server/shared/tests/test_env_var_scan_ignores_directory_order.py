"""The env-var scan must not depend on the order the filesystem lists files.

``_scan_env_vars`` gives a variable read in several files the category,
default and description of the FIRST file it reaches.  It walked
``rglob`` unsorted, so the answer was the filesystem's directory order,
which differs between machines.  The authoring snapshot is projected from
this scan and a guard compares it against the live projection, so a snapshot
written on one machine could fail the guard on another: regenerating it for
the jaato-server 1.3.0 bump flipped ``ZHIPUAI_API_KEY``,
``ZHIPUAI_ENABLE_THINKING`` and ``ZHIPUAI_THINKING_BUDGET`` from
``provider:zhipuai`` to ``provider:zhipuai_openai`` with no provider change.
"""

from __future__ import annotations

import pathlib

from jaato_server.shared.scaffold import introspect
from jaato_server.shared.tests.reversion import Reversion

_INTRO = "jaato-server/jaato_server/shared/scaffold/introspect.py"

REVERSIONS = [
    Reversion(
        target=_INTRO,
        find="        # provider:zhipuai_openai).\n        for py in sorted(root.rglob(\"*.py\")):\n",
        replace="        # provider:zhipuai_openai).\n        for py in root.rglob(\"*.py\"):\n",
        test="test_the_scan_gives_the_same_answer_whatever_the_directory_order",
        because="the scan follows the filesystem's order, so a reversed listing changes a shared var's category",
    ),
]


def _facts(found):
    return {name: (ev.category, ev.default, ev.description, list(ev.sources))
            for name, ev in found.items()}


def test_the_scan_gives_the_same_answer_whatever_the_directory_order(monkeypatch):
    forward = _facts(introspect._scan_env_vars())

    real = pathlib.Path.rglob

    def reversed_rglob(self, pattern, *args, **kwargs):
        return iter(sorted(real(self, pattern, *args, **kwargs), reverse=True))

    monkeypatch.setattr(pathlib.Path, "rglob", reversed_rglob)
    backward = _facts(introspect._scan_env_vars())

    assert backward == forward


def test_a_var_two_providers_read_is_filed_under_the_first_in_path_order():
    found = introspect._scan_env_vars()
    assert found["ZHIPUAI_API_KEY"].category == "provider:zhipuai"
