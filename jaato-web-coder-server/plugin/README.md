# jaato-web-coder-toolchains

The web coder's toolchains plugin (#1344). It runs in each session's runner,
as the session's account and under its confinement, and does everything that
touches a workspace's toolchains. The web coder's backend only holds the
operator's policy, and may run as an account that cannot write the workspaces.

| What | How |
|---|---|
| **Binding** | the `toolchain` user command, which the web coder page sends (never the model): `bind <tool> <version>`, `unbind <tool>`, `scan`, `cancel`, `status`. A bind starts an install job and returns at once; the job runs `mise install` into `<ws>/.home`, links the binaries mise reports (`mise bin-paths`) into `.home/.local/bin`, installs the pinned language server, and writes `.lsp.json`, the mise config and, while Java or Maven is bound, `.home/.mavenrc` (Java reads `user.home` from the account rather than `$HOME`, so without it Maven would use the account's `~/.m2`). Gradle needs no file: the framework sets `GRADLE_USER_HOME=<home>/.gradle` wherever it applies the workspace HOME, which covers `gradle` and a repository's `./gradlew`. Every install step execs through the session's `//child` AppArmor transition |
| **The record** | `.jaato/environment.json`: what is bound, the current or last job with its log, the proposals and the repositories' guidance files. The page reads it through the daemon (`workspace.file.fetch`) |
| **Proposals** | a scan of the repositories' markers (`.nvmrc`, `go.mod`, `pom.xml`, …) at session start and on `toolchain scan`, limited to what the operator allows |
| **Instructions** | a system-prompt section naming what is bound and the repositories' `AGENTS.md` / `CONTRIBUTING.md` / … (never their contents) |
| **The hint** | when a `cli`, `interactive_shell` or `notebook` result shows a command was not found and an allowed toolchain provides it: one line for the model (ask the user to bind it; do not install it another way), and a `tool.result_enriched` notice (`kind: "toolchain_offer"`, daemon protocol 1.31) that becomes the Bind chip in the rail |

Its policy is `<workspace>/.jaato/toolchain-offer.json`, which the web coder
page stages into the workspace through the daemon. A workspace without that
file gets nothing: no scan, no record, no hints, no instructions.

`apparmor/jaato-web-coder-toolchains.rules` is installed into the daemon
account's `~/.jaato/apparmor-fragments/`. It denies confined sessions
writing the offer (so the agent cannot widen the allow-list) and grants what
a bound JDK needs (its `.so` files and `lib/jspawnhelper`) and what a bound
Go needs (`pkg/tool/<os_arch>/*`: `compile`, `link`, …).

Installation: see [INSTALL.md](INSTALL.md).
