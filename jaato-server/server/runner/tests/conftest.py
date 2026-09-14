"""Shared pytest fixtures for runner-side tests.

Two AppArmor-related helpers in ``bootstrap_session`` need to be
auto-mocked for tests that use placeholder profile names like
``"cli_test"`` (which aren't AppArmor profiles loaded in the
kernel):

  1. ``_maybe_self_confine`` (PR 5a) — would call
     ``aa_change_profile`` and fail with errno=2 (profile not
     loaded) for placeholder names.
  2. ``_maybe_install_child_callback`` (PR 102 extraction; was
     inline pre-102) — would attempt the //child transition
     callback install which raises ``BootstrapError`` when the
     stub session's executor lacks
     ``set_apparmor_child_transition_callback``.

Tests that specifically exercise these helpers
(``test_runner_session_self_confine_5a.py`` and
``test_runner_session_apparmor_child_install.py``) opt out of the
auto-mock so they can exercise the real orchestration logic with
controlled inputs.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# Per-helper opt-out: each dedicated test file declares which helper
# it wants to exercise without auto-mock.  The OTHER helper stays
# mocked so the test's envelope.profile_name placeholder doesn't
# trip the unrelated kernel-side machinery.
_SELF_CONFINE_TESTERS = frozenset({
    "test_runner_session_self_confine_5a.py",
})
_CHILD_CALLBACK_TESTERS = frozenset({
    "test_runner_session_apparmor_child_install.py",
})


@pytest.fixture(autouse=True)
def _disable_runner_apparmor_helpers(request):
    """Auto-mock ``_maybe_self_confine`` + ``_maybe_install_child_callback``
    for every test in this directory.

    Most bootstrap tests use placeholder profile names that aren't
    AppArmor profiles loaded in the kernel.  Without the mock both
    helpers fail:

    - ``_maybe_self_confine`` calls ``aa_change_profile`` and the
      kernel returns errno=2 (profile not loaded) → ``BootstrapError("confine", ...)``.
    - ``_maybe_install_child_callback`` hits the case-3 path when
      ``envelope.profile_name`` is set + lacks ``"//"``, tries to
      install the callback on the stub session's executor, and
      raises ``BootstrapError("configure", ...)`` because the
      stub lacks ``set_apparmor_child_transition_callback``.

    Dedicated tests opt out of ONE specific helper (the one they
    exercise) via the per-helper sets above.  The OTHER helper
    stays mocked so the test's envelope placeholder doesn't trip
    the unrelated kernel-side machinery.
    """
    basename = request.node.fspath.basename
    skip_self_confine_mock = basename in _SELF_CONFINE_TESTERS
    skip_child_callback_mock = basename in _CHILD_CALLBACK_TESTERS

    patchers = []
    if not skip_self_confine_mock:
        patchers.append(patch(
            "server.runner.session._maybe_self_confine",
            return_value=None,
        ))
    if not skip_child_callback_mock:
        patchers.append(patch(
            "server.runner.session._maybe_install_child_callback",
            return_value=None,
        ))

    for p in patchers:
        p.start()
    try:
        yield
    finally:
        for p in reversed(patchers):
            p.stop()


class StubSession:
    """Minimal stand-in for ``JaatoSession``, satisfying every stamp
    ``bootstrap_session`` makes on the session it has just created.

    Lives in conftest ON PURPOSE.  Four separate test files each carried
    a private ``class _StubSession: pass`` nested inside their stub
    runtime's ``create_session``.  When production grew
    ``_stamp_daemon_identity`` (``set_daemon_session_id``, and #859's
    ``set_client_user_id`` beside it) two files were updated and four
    were not -- invisibly, because no commit-triggered workflow ran this
    directory (#736).  One shared stub means the next stamp breaks once,
    loudly, in one place.

    ``set_client_user_id`` is implemented even though no envelope in
    this directory currently carries ``created_by``: production calls it
    under ``if created_by:``, so today it is latent, and the next test
    that sets ``created_by`` would otherwise rediscover this whole
    cluster.

    Construction kwargs are recorded on ``create_session_kwargs`` so a
    stub runtime can hand the object straight back and tests can assert
    what flowed through.
    """

    def __init__(self, **kwargs: object) -> None:
        self.create_session_kwargs = dict(kwargs)
        self._daemon_session_id = None
        self._client_user_id = None

    def set_daemon_session_id(self, session_id: str) -> None:
        """Bootstrap step 3b stamps ``envelope.session_id`` here."""
        self._daemon_session_id = session_id

    def set_client_user_id(self, user_id: object) -> None:
        """Bootstrap step 3c stamps ``envelope.created_by`` here (#859)."""
        self._client_user_id = user_id
