# Installing the `web_coder_toolchains` plugin (deployment hand-off)

**Audience:** whoever deploys `jaato-server` and `@jaato/web-coder-server`.
Paths and unit names follow [`../deploy/README.md`](../deploy/README.md):
the daemon runs as `jaato` from `/srv/jaato/.venv` under the
`jaato-server` systemd unit. Adjust them if your host differs.

## What it is, and why the daemon needs it

The web coder lets a workspace owner bind toolchains to a workspace (Node,
Go, Bun, Java, Maven, Gradle, and Python's language server). **This plugin
does all of the work, inside each session's runner**:

- installs a toolchain with `mise` into `<workspace>/.home` when the page
  asks (the `toolchain` user command), links its binaries onto every
  command's `PATH`, installs the pinned language server, and writes
  `.lsp.json` (and `.home/.mavenrc` while Java or Maven is bound);
- keeps the record the page reads: `.jaato/environment.json`;
- proposes toolchains from the repositories' files, and names their
  `AGENTS.md` / `CONTRIBUTING.md` in the session's instructions;
- when a command fails because its toolchain is missing, tells the model to
  ask the user to bind it, and sends the page the notice behind the **Bind**
  chip.

The web coder server only holds the operator's policy (the `environment:`
block). It never writes a workspace, so it may run as any account, including
an ordinary one beside a root daemon. The page stages the policy into each
workspace as `.jaato/toolchain-offer.json`, through the daemon.

**Without the plugin, the Toolchains section can show the allow-list but
nothing binds, and there is no chip and no hint.** A workspace the web coder
never opened has no offer file, so the plugin does nothing there: installing
it on a daemon that also serves other workspaces is safe.

It is a Python **tool plugin** with no model-facing tools, runner tier. It
must be installed into the **daemon's** Python environment; the web coder
server (Node) cannot install it.

## Prerequisites

| | Needed | How to check |
|---|---|---|
| `jaato-server` | protocol **1.31** or later (`tool.result_enriched`) | `/srv/jaato/.venv/bin/python -c "import jaato_sdk.events as e; print(e.PROTOCOL_VERSION)"` |
| `@jaato/web-coder-server` | the release that ships `plugin/` with `jaato_web_coder_toolchains/` (this one) | `ls "$(npm root -g)/@jaato/web-coder-server/plugin/jaato_web_coder_toolchains/plugin.py"` |
| `@jaato/web-coder-ui` | the same release train (it stages the offer and sends the `toolchain` command) | served by the web coder server; no separate step |
| backend config | an `environment:` block; without it no offer is staged and the plugin stays silent | `grep -n '^environment:' /etc/jaato-web-coder/jaato_server.server.yaml` |
| `mise` on the daemon host | on the runner's `PATH`, in a system directory (see step 4) | `sudo -u jaato sh -c 'command -v mise'` |
| egress from sessions | the download hosts mise and the language servers use (below), if sessions' network is restricted | — |
| Python build backend | `hatchling`, fetched from PyPI at install time; on a host without PyPI, build the wheel elsewhere (step 2) | — |

The hosts installs typically fetch from, for an egress allow-list (mise's own backends may change them; check a failed install's log): `nodejs.org`, `go.dev` and
`dl.google.com`, `github.com` and `objects.githubusercontent.com` (Bun,
Java, Maven and Gradle releases through mise), `api.adoptium.net`,
`repo.maven.apache.org`, `services.gradle.org`, `download.eclipse.org`
(jdtls), `pypi.org` and `files.pythonhosted.org` (basedpyright),
`registry.npmjs.org` (typescript-language-server), `proxy.golang.org`
(gopls).

## Upgrading from `jaato-web-coder-toolchain-offer` (0.1)

The previous release installed `jaato-web-coder-toolchain-offer` and the
fragment `jaato-toolchain-offer.rules`, and the web coder server did the
installs itself. Before the steps below:

```bash
sudo -u jaato /srv/jaato/.venv/bin/python -m pip uninstall -y jaato-web-coder-toolchain-offer
DAEMON_HOME="$(getent passwd jaato | cut -d: -f6)"
sudo rm -f "$DAEMON_HOME/.jaato/apparmor-fragments/jaato-toolchain-offer.rules"
```

and remove `workspace_root`, `mise` and `python` from the web coder
server's `environment:` block: it refuses them now, by name. Toolchains
already bound keep working: the plugin reads their record
(`.jaato/environment.json`), `.lsp.json` and mise config as the server wrote
them.

The server also wrote three files the plugin now provides another way, and
nothing removes them for you. Delete them from existing workspaces, or the
session prompt names the toolchains and the repository guidance twice:

```bash
sudo find /srv/jaato/workspaces -maxdepth 4 \( \
     -path '*/.jaato/instructions/45-environment.md' \
  -o -path '*/.jaato/instructions/30-repo-guidance.md' \
  -o -path '*/.jaato/apparmor-fragments/jaato-environment.rules' \) \
  -exec grep -l 'jaato-managed' {} + | sudo xargs -r rm -v
```

(only copies that still carry the `jaato-managed` marker; a copy the user
made their own is kept).

## Rollout order

1. Upgrade `jaato-server` in `/srv/jaato/.venv` (if it is below 1.31).
2. Install this plugin into the same venv.
3. Install its AppArmor fragment.
4. Install `mise`.
5. Restart the daemon.
6. Upgrade `@jaato/web-coder-server` (which brings the matching UI), fix its
   `environment:` block if you are upgrading, and restart it.

Steps 1 to 5 fit one maintenance window. Step 6 can come before or after.

## 1. Locate the plugin source

It ships inside the web coder server's npm package:

```bash
PLUGIN_DIR="$(npm root -g)/@jaato/web-coder-server/plugin"
test -f "$PLUGIN_DIR/pyproject.toml" && echo ok
```

Or, from a git checkout of the jaato repository:
`<checkout>/jaato-web-coder-server/plugin`.

Install the plugin from the **same release** as the web coder server you
run: the two share the format of `toolchain-offer.json`, and the plugin
ignores an offer whose schema it does not know (and logs it once).

## 2. Install it into the daemon's venv

As the account that owns the venv (here `jaato`), with that venv's own pip:

```bash
sudo -u jaato /srv/jaato/.venv/bin/python -m pip install "$PLUGIN_DIR"
```

If the venv was made with uv:

```bash
sudo -u jaato uv pip install --python /srv/jaato/.venv/bin/python "$PLUGIN_DIR"
```

Offline host: build the wheel where PyPI is reachable, copy it over, and
install it with no network:

```bash
python -m pip wheel --no-deps -w dist "$PLUGIN_DIR"      # on the build machine
sudo -u jaato /srv/jaato/.venv/bin/python -m pip install --no-deps --no-index dist/jaato_web_coder_toolchains-*.whl
```

The distribution is `jaato-web-coder-toolchains`, the module
`jaato_web_coder_toolchains`, the plugin `web_coder_toolchains`. It depends
only on `jaato-sdk`, which `jaato-server` already brings.

**If the daemon restricts plugin sources.** When its environment sets
`JAATO_PLUGIN_ENTRY_POINT_ALLOWLIST` (check the unit and any
`EnvironmentFile`), add the distribution, or it is refused before it loads:

```ini
Environment=JAATO_PLUGIN_ENTRY_POINT_ALLOWLIST=...,jaato-web-coder-toolchains
```

## 3. Install the AppArmor fragment

The fragment does two things for confined sessions:

- denies writing, linking or locking any `.jaato/toolchain-offer.json`
  (reading stays allowed), so the agent cannot widen the operator's
  allow-list. The daemon, which stages the file, is not confined;
- grants what a JDK that mise installed under a workspace's `.home` needs:
  `m` on its `.so` files and exec of its `lib/jspawnhelper`; and exec of a
  mise Go's `pkg/tool/<os_arch>/*` (`compile`, `link`, `asm`, `cgo`, `vet`),
  without which every `go build` is refused. **Reinstall the fragment when
  upgrading the plugin**: it gained the Go rule in 0.2.1.

Install it into the **daemon account's** user tier, which applies to every
workspace that daemon serves:

```bash
DAEMON_HOME="$(getent passwd jaato | cut -d: -f6)"
sudo install -d -o jaato -g jaato -m 0755 "$DAEMON_HOME/.jaato/apparmor-fragments"
sudo install -o jaato -g jaato -m 0644 "$PLUGIN_DIR/apparmor/jaato-web-coder-toolchains.rules" \
    "$DAEMON_HOME/.jaato/apparmor-fragments/jaato-web-coder-toolchains.rules"
```

Use the home of the account in the unit's `User=` (for a root daemon,
`/root`). Keep the file name as shipped.

- It applies to sessions provisioned **after** it is installed (the restart
  in step 5 takes care of the running ones).
- Web coder sessions declare no `apparmor_fragments:`, so they compose it
  automatically. A profile that declares that list must add
  `jaato-web-coder-toolchains`.
- On a host without AppArmor, or for an unconfined session, nothing enforces
  it. The plugin validates every field of the offer and never quotes free
  text from it, and an unconfined agent could run mise itself anyway.

## 4. Install `mise`

The plugin runs `mise` in the session's runner, through the session's
`//child` AppArmor profile, which may exec what is in the system `PATH`
directories. Install mise **system-wide**, not into a user's home:

```bash
curl -fsSL https://mise.run | sudo MISE_INSTALL_PATH=/usr/local/bin/mise sh
/usr/local/bin/mise --version
```

(or your distribution's package). The plugin finds `mise` on the runner's
`PATH`; to name it explicitly, set it in the daemon unit:

```ini
Environment=JAATO_TOOLCHAINS_MISE=/usr/local/bin/mise
```

mise keeps nothing outside the workspace: every install lands in
`<workspace>/.home`, with a clean environment and `MISE_CEILING_PATHS`
so a repository's own `mise.toml` cannot choose what is downloaded.

## 5. Restart the daemon

Plugins are imported when the daemon starts its runner template, and every
pooled runner is forked from it, so a running daemon never picks the plugin
up:

```bash
sudo systemctl restart jaato-server
journalctl -u jaato-server -n 50 --no-pager
```

A restart unloads live sessions. They are saved and come back when users
reattach, but a turn in progress is cut, and so is an install in progress
(the next session can bind again). Schedule it accordingly.

## 6. Verify

**The daemon's environment sees it**, loaded the way a runner loads it:

```bash
sudo -u jaato /srv/jaato/.venv/bin/jaato-scaffold explain plugins | grep web_coder_toolchains
#   web_coder_toolchains   tool       runner   0 (0 core/0 disc)   <- jaato-web-coder-toolchains (jaato_web_coder_toolchains)
```

If the line is missing, look in `journalctl -u jaato-server` for an entry
naming `web_coder_toolchains`: it says why (refused by the allow-list, or a
protocol problem).

**The fragment is composed** into new sessions' profiles: after a session
starts, the daemon log line `AppArmor profile … composing N fragments: [...]`
lists `jaato-web-coder-toolchains`.

**The whole path**, with a signed-in user and an `environment:` block that
allows Java (say):

1. Open a session in a workspace in the web coder. The page stages the offer:
   ```bash
   sudo cat /srv/jaato/workspaces/<workspace>/.jaato/toolchain-offer.json
   # {"_jaato_managed": "toolchain-offer v2", "schema": 2, "toolchains": [...], ...}
   ```
2. Open the rail's *Toolchains* section and bind Java. The transcript shows
   `toolchain: binding Java 21 (job …)`, the section shows the install's
   output, then `java 21 installed.` The record is:
   ```bash
   sudo cat /srv/jaato/workspaces/<workspace>/.jaato/environment.json
   ```
3. Ask the agent to run `javac -version`: it runs, from `~/.local/bin`.
4. For the hint: unbind Java, ask the agent to run `javac -version` again.
   Expect the rail's *Toolchains* badge to show `!` with *Bind Java*, and
   the agent to tell the user to bind Java rather than install it.

## Rollback

```bash
sudo -u jaato /srv/jaato/.venv/bin/python -m pip uninstall -y jaato-web-coder-toolchains
sudo rm -f "$DAEMON_HOME/.jaato/apparmor-fragments/jaato-web-coder-toolchains.rules"
sudo systemctl restart jaato-server
```

Without the plugin, nothing binds and the chip and hint stop; toolchains
already installed stay on `PATH` in their workspaces (the links are in
`.home`), but a JDK loses the fragment's grants and may no longer start
under confinement. The files the page staged and the plugin wrote stay in
the workspaces and are harmless.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `explain plugins` has no `web_coder_toolchains` line | installed into another venv (check the `ExecStart` path in the unit), or refused by `JAATO_PLUGIN_ENTRY_POINT_ALLOWLIST` (it must name `jaato-web-coder-toolchains`) |
| Bind answers `toolchain: this workspace has no toolchain offer` | the page has not staged the offer into this workspace: the web coder server has no `environment:` block, the user does not own the workspace, or staging failed (the browser console says why) |
| Bind answers `mise is not installed where this session runs` | step 4: `mise` is not on the runner's `PATH`, or `JAATO_TOOLCHAINS_MISE` names a file that is not executable |
| the install fails with `Permission denied` running mise | mise is in a directory the session's `//child` profile does not exec (a user's home); install it into `/usr/local/bin` |
| the install fails on a download | sessions' egress does not allow the host (see the list above), or a proxy needs `HTTPS_PROXY` / `SSL_CERT_FILE` in the daemon's environment (both are passed to installs) |
| Bind answers `another session is installing a toolchain in this workspace` | two sessions in one workspace; the second binds once the first install ends |
| a bound JDK fails with `mmap` or `jspawnhelper` denials in `journalctl -k` | the fragment is missing, misnamed, in the wrong account's home, or the session's profile declares `apparmor_fragments:` without it |
| the log line for a new session does not list `jaato-web-coder-toolchains` | as above |
| the Toolchains section shows the allow-list but no *Bound* list or proposals | the daemon is below protocol 1.20, so the page cannot read `.jaato/environment.json` |
| `web_coder_toolchains: ... is not a schema-2 offer` in the daemon log | the plugin and the web coder server come from different releases; install the plugin from the running server's package |

What the plugin reads, writes and sends is documented in
[`README.md`](README.md) and in the module docstrings under
`jaato_web_coder_toolchains/`.
