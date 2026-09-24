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

```bash
sudo jaato-web-coder-server init --dir /etc/jaato-web-coder
```

writes `app.credential`, `session.secret`, `credentials.key` (the key the
per-user API-key store is encrypted with) and a `jaato_server.server.yaml` template
(all 0600) and prints the daemon-side entry. Put that entry in
`/etc/jaato/ws-apps.json` (mode 0600, owned by the daemon's user):

```json
{"jaato-web-coder": "<the printed credential>"}
```

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
