"""``pool`` commands parse as daemon commands, never as a model message.

Before protocol 1.35 nothing parsed ``pool``, so ``pool resize 6`` typed at
the TUI prompt (or sent with ``--cmd``) reached the MODEL as a request.
"""

from client_commands import CommandAction, parse_user_input


def test_pool_resize_is_a_server_command():
    parsed = parse_user_input("pool resize 6 12")
    assert parsed.action == CommandAction.SERVER_COMMAND
    assert (parsed.command, parsed.args) == ("pool.resize", ["6", "12"])


def test_bare_pool_reads_the_status():
    parsed = parse_user_input("pool")
    assert (parsed.action, parsed.command) == (
        CommandAction.SERVER_COMMAND, "pool.status")
