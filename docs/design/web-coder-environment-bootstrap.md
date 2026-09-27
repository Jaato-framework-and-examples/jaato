# Web coder environment bootstrap: toolchains, LSP and repo knowledge

Status: **proposed**. Nothing here is built.

A workspace the web coder creates starts with Python and little else. There
is no Node, Go, Rust or JVM, no language server attached, and no knowledge of
the repository it was cloned from. An agent-driven assessment of a live
deployment (Ubuntu 26.04, server 1.2.0rc1) graded it "D" for polyglot work,
and found 24 LSP tools doing nothing.

This document proposes that the **web coder application**, not the
framework, bootstraps a workspace's environment:

1. which toolchains a workspace has;
2. which language servers attach;
3. which of the repository's own guidance documents the agent is pointed at.

The user chooses, or accepts a proposal, and the application installs and
writes the result. The framework gains three small, client-neutral pieces
(§8). Everything else lives in `jaato-web-coder-server` (the BFF) and
`jaato-web-coder-ui` (the page).

## 1. Why the application, not the framework

Installing a toolchain is a **policy decision**: it costs disk and network,
it is a supply-chain choice, and a workspace owner should consent to it. The
web coder already owns every other per-workspace decision of that kind:

| Precedent | Where |
|---|---|
| binding a GitHub account, writing `GH_TOKEN=app://github` to `.env` | `src/github.ts` (#1227) |
| seeding `.home/.gitconfig` | `src/github.ts` |
| shipping working guidance as a versioned, user-overridable file | `src/managed-files.ts`, `docs/design/github-workspace-guidance.md` |
| reloading the owner's live sessions after a change | `GitHubService._reload` → `secret.reload` |
| cloning repositories into a new workspace | the #1332 picker, `workspace.clone` |

A TUI or SDK user working in their own checkout installs their own
toolchains. What they need from the framework is only the client-neutral part
(§8).

## 2. What already works, without any change

Three framework facts make an application-side bootstrap possible today:

- **A per-workspace HOME** (#1225). Model-driven subprocesses (`cli`,
  `interactive_shell`, the notebook kernel) run with `HOME=<ws>/.home` and
  the four `XDG_*` variables under it. The directory is ignored by git
  (`.home/.gitignore` is `*`).
- **Programs under that HOME may run.** The AppArmor rules grant `ix` on
  `<ws>/.home/.local/bin/*` and `<ws>/.home/.local/share/**/bin/*`
  (#1273/#1274), and `<ws>/.home/.local/bin` is appended to the subprocess
  `PATH`.
- **The confinement template runs what `PATH` exposes** (v38, #1342, which
  includes Rust coreutils on Ubuntu 26.04).

A version manager whose data lives under `.home` therefore installs binaries
the confined session can already execute. With `HOME=<ws>/.home`, mise's
defaults are:

| mise path | Resolves to | Granted |
|---|---|---|
| data, `~/.local/share/mise/installs/<tool>/<ver>/bin/` | `<ws>/.home/.local/share/mise/installs/...` | `ix` (`.local/share/**/bin/*`) |
| shims, `~/.local/share/mise/shims/` | `<ws>/.home/.local/share/mise/shims/` | not on `PATH`; see §4.2 |
| global config, `~/.config/mise/config.toml` | `<ws>/.home/.config/mise/config.toml` | inside the workspace |

**This has not been tried on a confined host.** It is the first thing phase 0
(§10) verifies.

## 3. Who does what

| Piece | Owner | Why there |
|---|---|---|
| deciding a toolchain is wanted, and asking the user | page | the page sees the event stream; the BFF does not, in `direct` mode (`web-server-bff.md` §1) |
| scanning a workspace's files for markers | BFF | it has filesystem access under its configured `workspace_root` |
| installing a toolchain | BFF | outside the confined runner, with the user's consent, as the same process that already writes `.env` and `.gitconfig` |
| writing `.lsp.json`, instruction files, the environment manifest | BFF | managed files (§6) |
| telling the agent what is installed | BFF (the `45-environment.md` managed file) | §6 |
| answering what can run *now* | framework (`get_environment(aspect="runtime")`) | §8 |
| hiding idle LSP tools, the `runtime` aspect, `AGENTS.md` | **framework** | every client benefits (§8) |

## 4. Toolchain binding

### 4.1 The binding

A binding is `(workspace, tool, version)`, for example `(ws, node, 22)`. The
BFF records it per signed-in user, beside the GitHub bindings, and writes it
to the workspace as the mise global config under that workspace's HOME:

```toml
# <ws>/.home/.config/mise/config.toml   (managed; see §6)
[tools]
node = "22"
go = "1.23"
```

It is written under `.home`, not as a `.tool-versions` or `mise.toml` in the
repository. The binding belongs to this workspace, not to the project, and
must not appear as an uncommitted change in the user's repo.

A version the repository pins (`.nvmrc`, `engines.node`, the `go` directive,
`rust-toolchain.toml`, `.tool-versions`) pre-fills the choice. The user can
change it.

### 4.2 Installation

The BFF runs the install. The confined runner never does:

```
MISE_DATA_DIR=<ws>/.home/.local/share/mise \
MISE_CONFIG_DIR=<ws>/.home/.config/mise \
MISE_CACHE_DIR=<ws>/.home/.cache/mise \
  mise install
```

Then it links each installed tool's binaries into `<ws>/.home/.local/bin/`,
which is already on the command `PATH`. mise's shims directory is not on that
`PATH`. Adding it would take a framework change, and a shim is a
mise-dispatching wrapper when a plain symlink to the version-pinned binary
does the same job.

Requirements:

- **Pinned, verified versions.** A tool allow-list the operator configures
  (`node`, `go`, `rust`, `java`, `bun`, …) and mise's checksum verification.
  No arbitrary plugin URLs.
- **Progress, cancel, retry,** shaped like the #1332 clone progress.
- **The same uid constraint as `.gitconfig` seeding.** The BFF writes into a
  `.home` the daemon created. If the two run as different users, the
  directory must be writable by the BFF, and the installed files readable
  and executable by the runner. The runner's profile grants no
  `dac_override`, so even a root runner is bound by file permissions. Phase 0
  confirms the deployment's layout.
- **Egress.** The install needs to reach the tools' download hosts. A
  deployment behind an egress policy lists them, or points mise at a mirror.

### 4.3 Per-workspace first, a shared cache later

Per-workspace installs duplicate data: Node is about 100 MB per workspace.
A daemon-wide, read-only cache (`/var/cache/jaato/toolchains`) removes that,
but it lives outside the workspace, so the runner needs `r` and `ix` on it.
That is a one-line **user-tier AppArmor fragment** the operator installs once
(`/root/.jaato/apparmor-fragments/toolchains.rules`), the mechanism
`premium-reactor` already uses. It needs no framework code, but it is a
daemon-wide grant, which is why it comes second.

## 5. Language servers

A small catalog in the BFF maps a bound toolchain to a server and its install
route:

| Toolchain | Server | Installed by |
|---|---|---|
| python | basedpyright | `pip` into the workspace tool-venv (`.jaato/tool-venv`) |
| node / TypeScript | typescript-language-server | `npm i -g --prefix <ws>/.home/.local` |
| go | gopls | `go install` with `GOBIN=<ws>/.home/.local/bin` |
| rust | rust-analyzer | `rustup component add` |
| java | jdtls | mise or a download; heavy (§11) |

On bind, the BFF installs the server and writes `<ws>/.lsp.json`. That is the
file the `lsp` plugin reads for a profile-less workspace, at the workspace
root, then `~/.lsp.json`. With #1332's clone flow repositories land in
subdirectories, so the workspace root is not anyone's repo.

Python needs no toolchain binding at all: the tool-venv exists in every
managed workspace, so basedpyright can be the first catalog entry.

**Two cautions:**

- **JSON cannot carry the managed-file marker.** `.lsp.json` needs its own
  ownership signal, for example a `"_jaato_managed": "lsp v1"` key the `lsp`
  plugin ignores (§6).
- **`.lsp.json` is model-writable,** which the `lsp` plugin already notes as
  its reason to prefer profile configuration. That is not new here: a model
  able to edit it can already run the same binary through `cli`.

## 6. Managed files

Everything the bootstrap writes goes through `managed-files.ts`, which
decides the write and refresh rules (write when absent, refresh when the
version changes, **never** clobber a copy whose marker the user deleted).

| File | Content | Marker |
|---|---|---|
| `<ws>/.home/.config/mise/config.toml` | the bound tools | `# jaato-managed: toolchains vN` (TOML comment) |
| `<ws>/.lsp.json` | the servers for the bound toolchains | `"_jaato_managed"` key; the module gains a JSON variant |
| `<ws>/.jaato/instructions/45-environment.md` | what is installed, how to add more, and a pointer to `get_environment(aspect="runtime")` for what is runnable now | first-line HTML comment, as today |
| `<ws>/.jaato/instructions/30-repo-guidance.md` | a pointer to the repo's own guidance (§7) | first-line HTML comment |
| `<ws>/.jaato/environment.json` | the machine-readable manifest §8 reads | `"_jaato_managed"` key |

`managed-files.ts` needs one change: the marker format per file type
(HTML comment, `#` comment, JSON key). Its ownership and version rules stay
as they are.

## 7. Repository knowledge

**A pointer, not a copy.** A copied `AGENTS.md` goes stale on the first
`git pull`. The instruction file says only where the guidance is:

```markdown
<!-- jaato-managed: repo-guidance v1 — delete this line to keep your own edits -->
This workspace contains repositories with their own agent guidance. Read the
relevant file before working in each:

- `api-server/AGENTS.md`
- `web/CONTRIBUTING.md`
```

The BFF writes it after a clone, from the files it finds:
- `AGENTS.md`;
- `CLAUDE.md`;
- `CONTRIBUTING.md`;
- `.github/copilot-instructions.md`;
- `.cursor/rules`.

It refreshes the file when the page reports a clone or a pull.

**Trust.** A repository's own guidance is fairly trusted when it is the
user's own, and a prompt-injection route when it is a third party's. The
pointer file delegates the reading to the agent, and the agent reads through
`readFile`, whose result is ordinary tool output. The pointer adds no text
from the repository to the trusted system prompt, which is the reason to
prefer it over inlining.

**Later.** Register each repository's `docs/` as a local `references`
source. Add an onboarding profile that writes seed memories into the raw
tier on a workspace's first session.

## 8. Framework pieces (separate issues)

Three pieces help every client, not only the web coder:

1. **Hide LSP tools when no server can attach.** Today the tools are
   exposed and answer "No LSP servers connected". That costs tokens on every
   request and misleads the model. The plugin should expose them only when
   its resolved server table is non-empty.
2. **A `runtime` aspect on `get_environment`.** The managed
   `45-environment.md` file already tells the agent what the web coder
   installed. It cannot say three things:
   - **what confinement lets run.** The BFF knows what it installed, not the
     exec scope or which fragments grant what. #1342 was coreutils that were
     installed and not runnable;
   - **what changed since the prompt was rendered.** A session's system
     prompt is rendered once and persisted, and a revived session keeps it,
     so a binding made mid-session is not in the text the model reads
     (#1291 is the same staleness);
   - **anything, for a client that is not the web coder.** A TUI or SDK user
     has no BFF writing files.

   All three are answered better by something the model asks for than by
   something it is always told. `get_environment` already has aspects (`os`,
   `shell`, `arch`, `cwd`, `network`, …) and nothing about what can run. A
   `runtime` aspect reports, live:
   - the exec scope and granted binaries (#1326's record);
   - the effective subprocess `PATH`, the tool-venv and HOME;
   - the toolchains in `<ws>/.jaato/environment.json` when present.

   It costs nothing per request, is always current (a revived session
   included), and serves every client. The managed instruction file ends
   with one line pointing at it.

   A standing block in the system prompt was the earlier idea. It is not
   worth its per-request cost when an instruction file plus a pull-based
   aspect covers the same facts.
3. **`AGENTS.md` in a checkout the user opened themselves.** The same pointer
   as §7, produced by the framework for any workspace whose root holds one.
   The web coder's version covers cloned subdirectories; this covers the TUI
   user in their own repo. If the framework does it, the BFF skips the root.

**A related idea, not part of this design:** explain a denial when it
happens. When a command fails with `Permission denied`, neither an
instruction file nor the `runtime` aspect connects that failure to the
environment. `cli` could, for example by adding "`/usr/bin/ls` resolves to
`/usr/lib/cargo/bin/coreutils/ls`, which this profile does not allow" to the
failed result. That is what would have stopped the assessing agent from
inventing a "command-name blocklist" for what was #1342. It is a framework
change of its own, independent of the bootstrap.

## 9. Detection and consent

Nothing is installed without the user's confirmation. Detection only
proposes.

| Signal | Seen by | When | Strength |
|---|---|---|---|
| markers after a clone (`package.json`, `go.mod`, `Cargo.toml`, `pom.xml`, `.tool-versions`, `.nvmrc`) | BFF scan, asked by the page when #1332's clone completes | before the first session | best moment to ask |
| a `cli` result with exit 127 or `<name>: not found` | page (tool events) | mid-session | unambiguous: the agent tried and failed |
| `WorkspaceFilesChangedEvent` naming a new manifest | page | mid-session | weak: writing a manifest does not mean running it |

The page shows a chip: *"Node 22 detected (from `.nvmrc`). Bind it?"*.
- **Accepting** calls `POST /api/toolchains/bind`.
- **Declining** is remembered per user and workspace, so the chip does not
  keep asking.

A detection never installs anything by itself.

## 10. Rollout

| Phase | What | Proves |
|---|---|---|
| 0 | by hand, on the 26.04 host: `mise install node@22` into a workspace `.home` as the BFF user, link it, run `node -v` from a confined session | §2, the uid and egress layout, v38 |
| 1 | framework: hide idle LSP tools; `AGENTS.md` pointer (§8.1, §8.3) | independent of everything below |
| 2 | BFF: basedpyright into the tool-venv, a managed `.lsp.json`, the repo-guidance pointer | Python LSP with no toolchain work |
| 3 | BFF + page: toolchain binding, install with progress, the environment manifest, the clone-time chip | the §4 flow |
| 4 | framework: the `runtime` aspect reading the manifest (§8.2); page: the mid-session chip | mid-session bindings |
| 5 | the shared cache and operator fragment (§4.3) | disk use |

## 11. Risks and open questions

- **Ownership, not only containment.** `GitHubService._resolveWorkspace`
  checks that a workspace path is under the BFF's `workspace_root`. It does
  not check that the signed-in user owns that workspace on the daemon
  (`WorkspaceInfo.owner`). The toolchain routes install software, so they
  must verify ownership, for example by asking the daemon through
  `workspace.list` over the user's own identity. Whether the GitHub bind
  route has the same gap is worth checking separately.
- **jdtls and memory.** A Java language server uses gigabytes, and #806
  (servers never reaped) is still open. Java should wait for that.
- **`proxy` mode and a BFF on another host.** Everything here assumes the
  BFF shares a filesystem with the daemon, as today's `.env` writes already
  do. A split deployment would need a daemon verb instead.
- **Scoped sessions.** A profile that declares `apparmor_fragments` gets no
  broad exec. Its fragments must grant the toolchain paths, and a `validate`
  warning could say so.
- **Removal.** Unbinding removes the links and the managed files. Whether it
  also deletes the installed data or keeps it for a rebind is open.

## 12. Not in scope

- **General-purpose profiles** (`new profile-set --archetype dev`). That is
  a scaffold archetype, useful to every client, and stays framework-side. The
  web picker lists whatever profiles a workspace has.
- **Containers or Nix.** They are heavier, and do not fit a host-level daemon
  with per-session AppArmor confinement.
