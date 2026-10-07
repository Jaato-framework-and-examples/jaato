# Deploying jaato-web-coder on one host

Everything here is a native install under systemd: Keycloak, the jaato
daemon (`pip`), this server (`npm`), and the host's reverse proxy. No
containers. The topology and the reasons for it are in
[`docs/design/web-server-bff.md`](../../docs/design/web-server-bff.md) §11.

```
https://jaato.example.org
  /auth/*  → Keycloak :8180        realm jaato-web-coder-shell, client jaato-web-coder
  /daemon  → jaato daemon :8080    --web-socket 127.0.0.1:8080 --ws-app-credentials …
  /*       → this server :8443     sign-in, /api/ticket, the bundle
```

## 1. Keycloak

Native install, started with its public hostname so tokens carry it as
`iss`, and with dynamic back-channel hostnames so this server may reach it
over loopback:

```bash
kc.sh start --hostname https://jaato.example.org/auth --http-port 8180 \
            --proxy-headers xforwarded --hostname-backchannel-dynamic=true
```

In realm **`jaato-web-coder-shell`**, one confidential client:

| Setting | Value |
|---|---|
| Client ID | `jaato-web-coder` |
| Client authentication | on; copy the secret into `/etc/jaato-web-coder/oidc.secret` (mode 0600) |
| Standard flow | on; direct access grants and implicit flow off |
| PKCE code challenge method | `S256` |
| Valid redirect URIs | `https://jaato.example.org/auth/callback` |
| Valid post-logout redirect URIs | `https://jaato.example.org/` |
| Back-channel logout URL | `https://jaato.example.org/auth/backchannel-logout` |

Optionally create a realm role (e.g. `jaato-user`) and set
`auth.oidc.required_role` so only members may sign in.

## 2. The shared app credential

The application's workspaces live under a directory owned by an OS account
of their own; create both first:

```bash
sudo useradd --system --create-home webcoder
sudo -u webcoder mkdir -m 0700 /home/webcoder/workspaces
sudo jaato-web-coder-server init --dir /etc/jaato-web-coder \
    --account webcoder --workspace-root /home/webcoder/workspaces
```

writes `app.credential`, `session.secret`, `credentials.key` (the key the
per-user API-key store is encrypted with) and a `jaato_server.server.yaml` template
(all 0600) and prints the daemon-side entry. Put that entry in
`/etc/jaato/ws-apps.json` (mode 0600, owned by the daemon's user):

```json
{"jaato-web-coder": {"credential": "<the printed credential>",
                     "account": "webcoder",
                     "workspace_root": "/home/webcoder/workspaces"}}
```

The daemon refuses to start when the root does not exist, is not owned by
the account, or overlaps the daemon's own workspace root or another
application's. A root daemon hands every file it writes in these
workspaces to `webcoder`; start it with `--runner-uid-policy
workspace-owner` so the sessions' runners run as `webcoder` too.

Edit `jaato_server.server.yaml`: `public_url`, `daemon.url`, `auth.oidc.issuer`, and
`backchannel_url: http://127.0.0.1:8180` if Keycloak is reached over
loopback.

## 3. Services

```bash
sudo cp jaato-server.service jaato-web-coder-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now jaato-server jaato-web-coder-server
journalctl -u jaato-web-coder-server -f     # "issuer … discovered", "bind channel open … as app jaato-web-coder", "listening …"
```

Both units run in the foreground (`Type=simple`); the daemon is started
**without** `--daemon`, which would double-fork and fight systemd.

The daemon's unit sets `KillMode=mixed`. Keep it if you write your own unit:
with systemd's default, a stop sends SIGTERM to the daemon and its runner
processes together, the runners exit first, and every session still loaded
loses its final save (`Failed to save session …: RunnerRPCClient is closed`).

With an `environment:` block (toolchains), also install the
`web_coder_toolchains` plugin, its AppArmor fragment and `mise` for the
daemon before starting it:
[`../plugin/INSTALL.md`](../plugin/INSTALL.md).

## 4. The proxy

`Caddyfile` (automatic TLS) or `nginx.conf` (bring your certificates). The
one thing to get right is the WebSocket upgrade on `/daemon`; both files
carry it.

## 5. Check

```bash
curl -s https://jaato.example.org/config.json
# {"daemon":"wss://jaato.example.org/daemon","ticketUrl":"./api/ticket","loginUrl":"./auth/login","autoConnect":true}
```

Open `https://jaato.example.org/`, sign in, and the status bar should show
the daemon's version; `jaato-web-coder-server`'s log shows
`ticket for jaato-web-coder:<you>` on every connect.
