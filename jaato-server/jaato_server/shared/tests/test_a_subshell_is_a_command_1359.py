"""A subshell before ``&&`` / ``||`` / ``|`` is a command (#1359).

``cli`` refuses a command its pre-flight analyzer cannot parse, and the
analyzer refused ``(cd dir && make) && echo OK`` as "chain operator '&&'
without a preceding command": the ``)`` closed the segment, and the operator
after it saw an empty one.  bash accepts all of these, so the refusal told the
model its shell was broken.  The shapes bash itself rejects stay refused.
"""

from __future__ import annotations

import pytest

from jaato_server.shared.command_analysis import UnanalyzableCommand, analyze_command
from jaato_server.shared.plugins.command_containment import first_denied_path
from jaato_server.shared.tests.reversion import Reversion

_CA = "jaato-server/jaato_server/shared/command_analysis.py"

REVERSIONS = [
    Reversion(
        target=_CA,
        find="                and not self._word_started and not self._after_subshell):\n",
        replace="                and not self._word_started):\n",
        test="test_bash_accepts_it_so_the_analyzer_does",
        because="a subshell before a chain operator is refused as no command at all",
    ),
    Reversion(
        target=_CA,
        find="            self._after_subshell = start is not None and len(self.segments) > start\n",
        replace="            self._after_subshell = True\n",
        test="test_bash_rejects_it_so_the_analyzer_does",
        because="an empty subshell `()` is counted as a command",
    ),
    Reversion(
        target=_CA,
        find="        else:\n            self._after_subshell = False\n",
        replace="",
        test="test_bash_rejects_it_so_the_analyzer_does",
        because="a separator after the subshell leaves the next operator a command it no longer has",
    ),
]


@pytest.mark.parametrize("command", [
    "(true) && echo OK",
    "(true) || echo OK",
    "(true) | cat",
    "( true )&& echo OK",
    "(cd x && make) && echo OK",
    "(a) > out.txt && b",
    "true && (echo a) && echo b",
    "((1)) && echo OK",
])
def test_bash_accepts_it_so_the_analyzer_does(command):
    analyze_command(command)


@pytest.mark.parametrize("command", [
    "() && echo OK",
    "(true); && echo OK",
    "(true)\n&& echo OK",
    "(true) &&",
    "&& echo OK",
])
def test_bash_rejects_it_so_the_analyzer_does(command):
    with pytest.raises(UnanalyzableCommand):
        analyze_command(command)


def test_the_subshell_body_is_still_segmented():
    words = [seg.words for seg in analyze_command("(cd x && make) && echo OK")]
    assert words == [["cd", "x"], ["make"], ["echo", "OK"]]


def test_cli_preflight_no_longer_refuses_it(tmp_path):
    assert first_denied_path(
        "(cd . && ls) && echo OK", str(tmp_path), on_parse_error="deny"
    ) is None
