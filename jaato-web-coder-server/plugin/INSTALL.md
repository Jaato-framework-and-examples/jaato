# Installing the `toolchain_offer` plugin (deployment hand-off)

**Audience:** whoever deploys `jaato-server` and `@jaato/web-coder-server`.
Paths and unit names follow [`../deploy/README.md`](../deploy/README.md):
the daemon runs as `jaato` from `/srv/jaato/.venv` under the
`jaato-server` systemd unit. Adjust them if your host differs.

## What it is, and why the daemon needs it

The web coder can bind toolchains to a workspace (Node, Go, Java, ...). When
a command in a session fails because its toolchain is missing, this plugin:

- tells the model which toolchain provides the command and that the user
  binds it from the web coder, so the model doesn't try to install it some
  other way;
- sends the page a notice that shows up as the **Bind** chip in the session
  rail's *Toolchains* section.

Nothing else detects this. **Without the plugin there is no chip and no hint.**
Binding toolchains by hand from the rail still works either way.

It is a Python **enrichment plugin**. It runs inside the jaato daemon's
runner processes, so it must be installed into the **daemon's** Python
environment, not the web coder's. The web coder backend (Node) cannot install
it at runtime.

It is off in every workspace where the backend hasn't written
`<workspace>/.jaato/toolchain-offer.json`. So installing it on a daemon that
also serves non-web-coder workspaces is safe: it does nothing there.

## Prerequisites

| | Needed | How to check |
|---|---|---|
| `jaato-server` | a release whose protocol is **1.31** or later (it carries `tool.result_enriched`) | `/srv/jaato/.venv/bin/python -c "import jaato_sdk.events as e; print(e.PROTOCOL_VERSION)"` |
| `@jaato/web-coder-server` | a release that writes `toolchain-offer.json` and ships `plugin/` (this one) | `ls "$(npm root -g)/@jaato/web-coder-server/plugin/pyproject.toml"` |
| `@jaato/web-coder-ui` | a release that reads `tool.result_enriched` (the same release train) | served by the web coder server; no separate step |
| backend config | an `environment:` block in the web coder server's config; without it the offer file is never written and the plugin stays silent | `grep -n '^environment:' /etc/jaato-web-coder/jaato_server.server.yaml` |
| Python build backend | `hatchling`. It is fetched from PyPI at install time. On a host without PyPI access, build the wheel elsewhere (step 2, offline variant) | — |

With an older daemon (protocol below 1.31), the model still gets the hint but
the page gets no chip. Nothing breaks either way.

## Rollout order

1. Upgrade `jaato-server` in `/srv/jaato/.venv` (if it is below 1.31).
2. Install this plugin into the same venv (below).
3. Restart the daemon (below).
4. Upgrade `@jaato/web-coder-server` (which brings the matching UI) and
   restart `jaato-web-coder-server`.

Steps 1 to 3 can be done in one maintenance window. Step 4 can come before
or after; the pieces tolerate each other's absence.

## 1. Locate the plugin source

It ships inside the web coder server's npm package:

```bash
PLUGIN_DIR="$(npm root -g)/@jaato/web-coder-server/plugin"
test -f "$PLUGIN_DIR/pyproject.toml" && echo ok
```

Or, from a git checkout of the jaato repository:
`<checkout>/jaato-web-coder-server/plugin`.

Install the plugin from the **same release** as the web coder server you
run. The two share the file format of `toolchain-offer.json`, and a mismatch
turns the hints off (the plugin ignores a schema it does not know and logs
it once).

## 2. Install it into the daemon's venv

As the account that owns the venv (here `jaato`), with that venv's own pip:

```bash
sudo -u jaato /srv/jaato/.venv/bin/python -m pip install "$PLUGIN_DIR"
```

If the venv was made with uv:

```bash
sudo -u jaato uv pip install --python /srv/jaato/.venv/bin/python "$PLUGIN_DIR"
```

Offline host: build the wheel on a machine with PyPI access, copy it over
and install it with no network:

```bash
python -m pip wheel --no-deps -w dist "$PLUGIN_DIR"      # on the build machine
sudo -u jaato /srv/jaato/.venv/bin/python -m pip install --no-deps --no-index dist/jaato_web_coder_toolchain_offer-*.whl
```

The distribution is named `jaato-web-coder-toolchain-offer` and the module
`jaato_toolchain_offer`. It depends only on `jaato-sdk`, which `jaato-server`
already brings, and it pulls in nothing else.

**If the daemon restricts plugin sources.** When the daemon's environment
sets `JAATO_PLUGIN_ENTRY_POINT_ALLOWLIST` (check the unit and any
`EnvironmentFile`), add the distribution to it, or the plugin is refused
before it loads:

```ini
Environment=JAATO_PLUGIN_ENTRY_POINT_ALLOWLIST=...,jaato-web-coder-toolchain-offer
```

If the variable is unset, every installed distribution may contribute
plugins, and nothing needs to change.

## 3. Restart the daemon

Plugins are imported when the daemon starts its runner template, and every
pooled runner is forked from that template, so a running daemon never picks
the plugin up:

```bash
sudo systemctl restart jaato-server
journalctl -u jaato-server -n 50 --no-pager
```

A restart unloads live sessions. They are saved and come back when users
reattach, but a turn in progress is cut. Schedule it accordingly.

## 4. Verify

**The daemon's environment sees it**, loaded the way a runner loads it:

```bash
sudo -u jaato /srv/jaato/.venv/bin/jaato-scaffold explain plugins | grep toolchain_offer
# toolchain_offer        enrichment runner   dynamic   <- jaato-web-coder-toolchain-offer (jaato_toolchain_offer)
```

If the line is missing, look in `journalctl -u jaato-server` for an entry
naming `toolchain_offer`. It says why: refused by the allow-list, or a
protocol problem.

**The whole path**, with a signed-in user and an `environment:` block that
allows at least one toolchain (Java, say) that is not bound in the test
workspace:

1. Open a workspace in the web coder and open the rail's *Toolchains*
   section. The backend writes the offer file:
   ```bash
   sudo cat /srv/jaato/workspaces/<workspace>/.jaato/toolchain-offer.json
   # {"_jaato_managed":"toolchain-offer v1","schema":1,"toolchains":[...]}
   ```
   (The workspace root is whatever `environment.workspace_root` names.)
2. In a session in that workspace, ask the agent to run `javac -version`.
3. Expect:
   - the *Toolchains* badge in the rail shows `!`, and the section offers
     *Bind Java*;
   - the agent tells the user to bind Java from the Toolchains section,
     rather than trying to install it.
4. Bind it from the chip, then ask the agent to run the command again. It
   works in the same session, with no restart.

## Rollback

```bash
sudo -u jaato /srv/jaato/.venv/bin/python -m pip uninstall -y jaato-web-coder-toolchain-offer
sudo systemctl restart jaato-server
```

Nothing else depends on the plugin. Without it the chip and the hint stop,
and binding by hand from the rail keeps working. The offer files the backend
wrote stay in the workspaces and are harmless.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `explain plugins` has no `toolchain_offer` line | installed into another venv (check the `ExecStart` path in the unit), or refused by `JAATO_PLUGIN_ENTRY_POINT_ALLOWLIST` |
| listed, but no chip and no hint | the daemon wasn't restarted; or the workspace has no `.jaato/toolchain-offer.json` (no `environment:` block, or the backend never opened that workspace); or the command's toolchain isn't in the allow-list, or the command isn't one the catalog maps to it |
| the hint reaches the agent but no chip appears | the daemon is below protocol 1.31, or the page is an older UI |
| `toolchain_offer: ... is not a schema-1 offer` in the daemon log | the plugin and the web coder server come from different releases; install the plugin from the running server's package |
| the offer file exists but the runner cannot read it | uid layout: the file is written by the web coder server's account and read by the daemon's. See "The uid layout matters" in [`../README.md`](../README.md) |

What the plugin reads and what it sends are documented in
[`README.md`](README.md) and in the `jaato_toolchain_offer/plugin.py`
docstring.
