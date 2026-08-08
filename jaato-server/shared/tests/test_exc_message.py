"""exc_message: the message slot is never blank, for any exception.

Regression cover for the truthy-but-empty shape that appeared twice on the
cascade degrade-push path: ``exc or "default"`` never fires, because an
exception instance is truthy even when it stringifies to "" (a no-arg
``TimeoutError`` is exactly that).  The log rendered as ``push failed ()``.
"""
import pytest

from shared.utils.errors import exc_message


def test_no_arg_exception_falls_back_to_class_name():
    assert exc_message(TimeoutError()) == "TimeoutError"


def test_message_is_preserved_when_present():
    assert exc_message(ValueError("boom")) == "boom"


@pytest.mark.parametrize("exc", [
    TimeoutError(), RuntimeError(), ValueError(), OSError(),
    KeyboardInterrupt(), Exception(),
])
def test_never_empty_for_no_arg_exceptions(exc):
    assert exc_message(exc) != ""


def test_the_broken_idiom_is_what_this_replaces():
    # Documents WHY the helper exists: the natural form silently picks the
    # truthy-but-empty instance, so the default is dead code.
    exc = TimeoutError()
    assert bool(exc) is True          # truthy...
    assert str(exc) == ""             # ...and empty at the same time
    assert (exc or "no message") is exc          # the `or` never fires
    assert str(exc or "no message") == ""        # -> renders as ()
    assert exc_message(exc) == "TimeoutError"    # the fix


def test_accepts_base_exception_not_just_exception():
    # Signature is BaseException: KeyboardInterrupt/SystemExit are logged too.
    assert exc_message(SystemExit()) == "SystemExit"
