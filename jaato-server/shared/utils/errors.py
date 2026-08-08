"""Exception-message helpers.

Exists to make one specific mistake hard to write.  See :func:`exc_message`.
"""

__all__ = ["exc_message"]


def exc_message(exc: BaseException) -> str:
    """Return ``exc``'s message, falling back to its class name when empty.

    Use this anywhere an exception is rendered into a log line, an error
    field, or a user-facing string.

    **Why this exists.**  The natural-looking idiom is wrong:

    .. code-block:: python

        logger.warning("push failed (%s)", exc or "no message")   # BROKEN

    An exception *instance* is truthy regardless of its message, so the
    ``or`` never fires and the truthy-but-empty value is selected.  A
    ``TimeoutError()`` raised with no args is truthy and stringifies to
    ``""`` at the same time, so the line renders as ``push failed ()``.
    The fallback is dead code that looks like a guard.

    Testing the string instead of the object is what actually works::

        logger.warning("push failed (%s)", exc_message(exc))      # -> "(TimeoutError)"

    This shape appeared twice on the cascade degrade-push path in
    ``server/session_manager.py``, the second time seven lines below a
    comment describing it — which is why the rule lives in a function you
    have to call rather than in prose you have to find.

    Args:
        exc: The exception to render.  Any ``BaseException``.

    Returns:
        ``str(exc)`` when it is non-empty, otherwise the exception's class
        name (e.g. ``"TimeoutError"``).  Never returns an empty string for
        an exception whose class has a name.
    """
    return str(exc) or type(exc).__name__
