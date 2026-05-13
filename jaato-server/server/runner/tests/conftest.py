"""Shared pytest fixtures for runner-side tests.

PR 5a introduced an AppArmor self-confine step (``_maybe_self_confine``)
to ``bootstrap_session``.  Most bootstrap tests use placeholder
profile names like ``"cli_test"`` that aren't actually loaded in the
kernel — without a mock, those tests would now fail with
``aa_change_profile errno=2``.

This conftest auto-mocks ``_maybe_self_confine`` so existing
bootstrap tests pass unchanged.  Tests that specifically exercise
self-confine logic (``test_runner_session_self_confine_5a.py``)
patch the underlying ``confine_to_profile`` / ``read_current_profile``
helpers directly and so don't go through this mock.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _disable_runner_self_confine(request):
    """Auto-mock ``server.runner.session._maybe_self_confine`` for
    every test in this directory EXCEPT the dedicated self-confine
    test file.

    Most bootstrap tests use placeholder profile names that aren't
    AppArmor profiles loaded in the kernel.  Pre-PR-5a, bootstrap_session
    didn't call confine at all (cold-spawn handled it in
    ``__main__.py``).  Post-PR-5a, bootstrap_session WILL call confine
    when ``envelope.profile_name`` is set — so the placeholder names
    cause ``aa_change_profile errno=2 (profile not loaded)``.

    This fixture is a NO-OP for tests in
    ``test_runner_session_self_confine_5a.py`` which inject their
    own mocks targeting the underlying ``confine_to_profile`` /
    ``read_current_profile`` helpers — letting them exercise the
    real ``_maybe_self_confine`` orchestration logic.
    """
    # Skip the mock for the self-confine test file (it does its own
    # injection at the helper level).
    if request.node.fspath.basename == "test_runner_session_self_confine_5a.py":
        yield
        return

    with patch(
        "server.runner.session._maybe_self_confine",
        return_value=None,
    ):
        yield
