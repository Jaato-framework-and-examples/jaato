"""CONFIRMED: a REUSED pool slot carries session A's env into session B.

Result on jaato d64dae7d, 2026-08-27:

    [1-declares] canary present in output: True
    [2-does-not] canary present in output: True      <- LEAK
    daemon log:  acquire_slot: cascade reuse HIT — slot pid=2184931

Session 2 has its OWN workspace and its OWN .env declaring no such key,
and still reads it. session_env is the field carrying DECODED pass:// /
vault:// secrets, so what survives is session A's resolved credentials,
readable by any tool session B runs.

Slot reuse is gated on cascade_driver_id ALONE (runner_pool.acquire_slot),
and _apply_envelope_session_env only SETS keys — it never clears ones the
new session does not define.  Its no-reset is justified by "a runner
process serves exactly one session for its whole lifetime", which pool
reuse contradicts.

Session 1 declares LEAK_CANARY in its .env.  Session 2 shares the cid,
declares no such key, and is asked to print it through a tool.  If the
canary comes back, session A's resolved env — which is where decoded
pass:// secrets live — survived into session B.
"""
import asyncio
import sys as _sys

# Line-buffer stdout: redirected to a file, Python block-buffers and
# the log stays EMPTY until exit — which reads as 'hung' for a script
# whose scenarios take minutes each.
try:
    _sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover - Python < 3.7
    pass
, pathlib, sys, uuid
from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.events import ClientType, EventType

SOCK, BASE, CFG = sys.argv[1], sys.argv[2], sys.argv[3]
CID = f"leak-{uuid.uuid4().hex[:8]}"
CANARY = "CANARY_FROM_SESSION_ONE_SHOULD_NOT_SURVIVE"


async def run(label, ws, extra_env, prompt):
    p = pathlib.Path(ws); p.mkdir(parents=True, exist_ok=True)
    lines = ["JAATO_PROFILE_SET=openrouter_gpt5mini"] + extra_env
    (p / ".env").write_text("\n".join(lines) + "\n")
    out = []
    async with IPCClient.session(
            profile="probe", workspace_path=str(p), config_root=CFG,
            env_file=".env", socket_path=SOCK,
            cascade_driver_id=CID) as sess:
        sess.client.subscribe(
            EventType.AGENT_OUTPUT,
            lambda e: out.append(str(getattr(e, "text", "") or "")))
        await sess.complete(prompt)
    text = "".join(out)
    print(f"[{label}] canary present in output: {CANARY in text}")
    return text


async def main():
    print(f"cid={CID} (both sessions share it, so the slot is reuse-eligible)")
    await run("1-declares", f"{BASE}/leak1", [f"LEAK_CANARY={CANARY}"],
              "Run the shell command: printenv LEAK_CANARY || echo ABSENT")
    # The slot is returned to the pool ASYNCHRONOUSLY after session end.
    # Without this wait session 2 acquires before the return lands, gets a
    # FRESH slot, and a clean result would say nothing about reuse — which
    # is the only path this test exists to exercise.  Measured: the return
    # landed 50ms after session 2 had already acquired elsewhere.
    print("waiting for slot return before session 2 ...")
    await asyncio.sleep(20)
    text = await run("2-does-not", f"{BASE}/leak2", [],
                     "Run the shell command: printenv LEAK_CANARY || echo ABSENT")
    print("=" * 60)
    print("VERDICT:", "LEAK — session 1's env reached session 2"
          if CANARY in text else "clean — session 2 did not see it")

asyncio.run(main())
