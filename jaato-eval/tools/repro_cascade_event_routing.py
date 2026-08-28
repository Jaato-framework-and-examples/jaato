"""Repro: what a client receives from a session stamped with a cascade id.

Four scenarios against one live daemon, each printing the events that
actually reached the client:

  A  no cascade id                       -> baseline
  B  cascade id, NOT registered          -> the defect
  C  cascade id, registered as observer  -> partial fix
  D  pool declared with limits={"tokens": 0}, spawn attempted

Usage:
    python repro_cascade_event_routing.py <socket> <workspace> \
        <config_root> [profile] [profile_set]

<config_root> must contain profiles/<profile>.yaml (default profile:
"worker").  jaato-eval's tasks/example-echo/.jaato works as-is.

A/B/C each send one short prompt, so the run costs three model turns.
D sends nothing and costs none.
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

import os
import pathlib
import sys
import uuid

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.events import (ClientType, ErrorEvent, EventType, HistoryEvent,
                              SessionTerminatedEvent, TurnCompletedEvent)

SOCK, WS, CFG = sys.argv[1], sys.argv[2], sys.argv[3]
PROFILE = sys.argv[4] if len(sys.argv) > 4 else "worker"
PROFILE_SET = sys.argv[5] if len(sys.argv) > 5 else os.environ.get(
    "JAATO_PROFILE_SET", "")

# The daemon reads JAATO_PROFILE_SET from the WORKSPACE's .env, never from
# this process's environment.  Writing it here is what makes the script
# self-contained: without it a tier-2 profile is invisible and every
# create_session waits out its full timeout on a COLD daemon -- while
# appearing to work against one that happened to discover the set during
# some earlier run.  A repro whose result depends on daemon history is not
# a repro; this one did, until it wrote its own .env.
pathlib.Path(WS).mkdir(parents=True, exist_ok=True)
pathlib.Path(WS, ".env").write_text(
    f"JAATO_PROFILE_SET={PROFILE_SET}\n" if PROFILE_SET else "")
# MUST drive the profile to its declared terminus.  A prompt the model can
# answer in prose ends the turn with finish_reason="stop" and never calls
# signal_completion — which is the very path under test, so a chatty prompt
# silently tests nothing.
PROMPT = ("Create a file called answer.txt in the workspace root whose "
          "entire contents are the single word READY. Then call "
          "signal_completion, reporting the path you created in "
          "'file_written' and the exact contents in 'content'.")

WATCH = [EventType.TURN_COMPLETED, EventType.HISTORY,
         EventType.SESSION_TERMINATED, EventType.ERROR,
         EventType.AGENT_ERROR, EventType.AGENT_OUTPUT]


async def _client():
    c = IPCClient(socket_path=SOCK, client_type=ClientType.API,
                  workspace_path=WS, config_root=CFG)
    await c.connect(timeout=120)
    return c


async def scenario(label, *, cid=None, register=False, send=True):
    seen = []
    settled = asyncio.Event()
    got_history = asyncio.Event()
    settled_by = [None]
    loop = asyncio.get_running_loop()

    def note(name):
        seen.append(name)
        if name in ("TURN_COMPLETED", "SESSION_TERMINATED"):
            settled_by[0] = settled_by[0] or name
            loop.call_soon_threadsafe(settled.set)
        if name == "HISTORY":
            loop.call_soon_threadsafe(got_history.set)

    errors = []

    def on_err(e):
        errors.append(f"{getattr(e, 'error_type', None)}: "
                      f"{str(getattr(e, 'error', ''))[:110]}")

    c = await _client()
    for et in WATCH:
        c.subscribe(et, lambda e, et=et: note(str(et).split(".")[-1]))
    for et in (EventType.ERROR, EventType.AGENT_ERROR):
        c.subscribe(et, on_err)
    if cid and register:
        await c.cascade_register(cid, "observer",
                                 [TurnCompletedEvent, HistoryEvent,
                                  SessionTerminatedEvent, ErrorEvent])
    kwargs = {"name": f"repro-{label}", "profile": PROFILE}
    if cid:
        kwargs["cascade_driver_id"] = cid
    try:
        sid = await c.create_session(**kwargs)
    except Exception as exc:
        print(f"[{label}] create_session RAISED {type(exc).__name__}: {exc}")
        print(f"[{label}]   error_type={getattr(exc, 'error_type', None)!r}")
        await c.disconnect()
        return
    print(f"[{label}] create_session -> sid={sid!r}")
    if send:
        # Event-driven, NOT a fixed sleep.  A sleep short enough to keep the
        # repro quick is also short enough to cut off a slow turn, and a
        # missing event then means "not finished yet" rather than "never
        # delivered" — which is exactly the confusion this script exists to
        # remove.  Waiting on the settle signal makes an absent event mean
        # absent.  AGENT_OUTPUT is the fallback liveness signal for the
        # routing case where neither terminal event reaches this client.
        await c.send_message(PROMPT)
        try:
            await asyncio.wait_for(settled.wait(), timeout=180)
            print(f"[{label}] settled on: {settled_by[0]}")
        except asyncio.TimeoutError:
            print(f"[{label}] NEVER SETTLED within 180s "
                  f"(events so far: {sorted(set(seen))})")
        await asyncio.sleep(2)
        await c.request_history()
        try:
            await asyncio.wait_for(got_history.wait(), timeout=30)
        except asyncio.TimeoutError:
            print(f"[{label}] no HistoryEvent within 30s of request_history()")
    else:
        await asyncio.sleep(5)
    counts = {}
    for name in seen:
        counts[name] = counts.get(name, 0) + 1
    print(f"[{label}] events received: {counts or '(none)'}")
    for e in errors:
        print(f"[{label}]   error -> {e}")
    await c.disconnect()


async def via_facade(label, cid):
    """Same session, driven through IPCClient.session().complete().

    The only remaining difference from scenario B: the facade's
    send-and-wait recipe (first-of TURN_COMPLETED / SESSION_TERMINATED)
    instead of send_message + an explicit wait.
    """
    seen = []
    kwargs = {"profile": PROFILE, "workspace_path": WS, "config_root": CFG,
              "socket_path": SOCK}
    if cid:
        kwargs["cascade_driver_id"] = cid
    async with IPCClient.session(**kwargs) as sess:
        c = sess.client
        for et in WATCH:
            c.subscribe(et, lambda e, et=et: seen.append(str(et).split(".")[-1]))
        await sess.complete(PROMPT)
        await c.request_history()
        await asyncio.sleep(8)
    counts = {}
    for name in seen:
        counts[name] = counts.get(name, 0) + 1
    print(f"[{label}] events received: {counts or '(none)'}")


async def main():
    print("=" * 62)
    await scenario("A no-cid")

    cid = f"repro-{uuid.uuid4().hex[:8]}"
    owner = await _client()
    await owner.cascade_budget_set(cid, limits={"tokens": 200000})
    print("=" * 62)
    print(f"declared cid={cid} limits={{'tokens': 200000}}")
    await scenario("B cid-unregistered", cid=cid)
    print("=" * 62)
    await scenario("C cid-observer", cid=cid, register=True)

    print("=" * 62)
    await via_facade("E cid-facade-complete", cid)
    print("=" * 62)
    await via_facade("F nocid-facade-complete", None)

    zero = f"repro-zero-{uuid.uuid4().hex[:8]}"
    owner_errors = []
    for et in (EventType.ERROR, EventType.AGENT_ERROR):
        owner.subscribe(et, lambda e: owner_errors.append(
            f"{getattr(e, 'error_type', None)}: {str(getattr(e, 'error', ''))[:140]}"))
    print("=" * 62)
    print(f"declared cid={zero} limits={{'tokens': 0}}  (no degrade)")
    await owner.cascade_budget_set(zero, limits={"tokens": 0})
    await asyncio.sleep(5)
    print(f"[D declare] owner errors: {owner_errors or '(none — silent)'}")
    await scenario("D zero-pool", cid=zero, register=True, send=False)
    await owner.disconnect()

asyncio.run(main())
