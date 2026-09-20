# Extending `jaato-scaffold`

Two seams: a **verb** (a new subcommand) and a **topic** (a new, or extended,
`explain` answer).  Pick the topic seam when your package has something to
SAY — a daemon extension, a subsystem with its own config files — and the verb
seam when it has something to RUN.

## A verb

`jaato-scaffold` ships three built-in verbs — `explain`, `validate`, `new`.
External packages (the premium `compile` verb, or your own) add verbs **without
modifying this repo** by registering a `ScaffoldVerb` under the
`jaato.scaffold_verbs` entry-point group. The CLI discovers and mounts them at
startup; a verb whose package is not installed simply does not appear. This is
the same convention as `jaato.premium` and `jaato.extensions` elsewhere in the
framework.  (There is no `jaato.premium_reactors` group, as this line used to
claim: reactors mount as the `reactors` entry in `jaato.extensions`, and their
rules load from directories, not entry points.)

## The contract

Import everything from the one stable surface, `shared.scaffold.api`
(`SCAFFOLD_EXTENSION_API` is its version — gate on it if you need a minimum):

```python
from shared.scaffold.api import (
    ScaffoldVerb, GeneratedFile, write_files, emit_then_validate,
    validate_workspace, Diagnostic, introspect,
)

class MyVerb:
    name = "myverb"
    help = "one-line help shown in `jaato-scaffold --help`"

    def configure(self, parser):        # register your args on a fresh subparser
        parser.add_argument("target")
        parser.add_argument("--workspace", required=True)

    def run(self, args) -> int:         # return a process exit code
        files = [GeneratedFile(path="…", content="…")]
        written, diags = emit_then_validate(files, args.workspace)
        errors = [d for d in diags if d.severity == "error"]
        return 1 if errors else 0
```

`MyVerb` satisfies the `ScaffoldVerb` protocol (name / help / `configure` /
`run`). It may be an instance or a zero-arg class/factory — the loader handles
both.

## Register it

In your package's `pyproject.toml`:

```toml
[project.entry-points."jaato.scaffold_verbs"]
myverb = "my_package.my_module:MyVerb"
```

Install your package into the same environment as `jaato-server`, and
`jaato-scaffold myverb` appears. Built-in verb names (`explain` / `validate` /
`new`) always win on a name collision; a verb that fails to import is skipped
with a warning rather than breaking the CLI.

## What you get to reuse

- `GeneratedFile` + `write_files` + `emit_then_validate` — the generic
  emit-then-validate plumbing (write a tree, run it back through the framework
  validator — the discipline the built-in `new` verb uses).
- `validate_workspace(workspace, *, profile_set=None, only=None) -> List[Diagnostic]`
  — profile / provider / plugin / knob validation.
- `introspect` — `providers()`, `plugins()`, `resolve_provider()`, `gc_strategies()`,
  `profile_schema()` — for fail-loud checks against the installed framework.

Anything your verb needs beyond structural validation (e.g. asset-*contract*
checks that `validate_workspace` doesn't perform) it does itself — that's the
point of a verb: the generic host validates structure, the verb owns its domain.

---

## A topic

`jaato-scaffold explain` is the surface an agent reads before it writes a file.
A package that contributes a daemon extension — premium's reactor engine is the
worked case — is invisible there: `explain plugins` will never show it, because
it is not a plugin, and there is no other place to look.  The measured cost is
that a session asked to add a reactor could not learn where the declaration
file goes, what keys it takes, or what the script must define, and read the
engine's source instead.

Register an `ExplainTopic` under `jaato.scaffold_topics`:

```toml
[project.entry-points."jaato.scaffold_topics"]
reactors = "my_package.topics:ReactorsTopic"
```

```python
from shared.scaffold.api import ExplainTopic, TopicRequest   # Rendered = (dict, str)

class ReactorsTopic:
    name = "reactors"                 # `jaato-scaffold explain reactors`
    help = "event -> script rules"    # the `# ...` blurb in the banner
    arg = ""                          # "<name>" / "[<filter>]" when it takes one
    extends = ""                      # "" = a topic of its own
    reads_workspace = True            # the banner shows [--workspace DIR]

    def render(self, request: TopicRequest):
        return {"tiers": [...]}, "reactors — ...\n"
```

Only `name` and `render` are required; the rest are read with defaults.

**One signature.**  `render` always takes a `TopicRequest` (`topic`, `name`,
`workspace`) whatever shape the topic is, so a contributor never has to learn
the built-in table's five calling conventions — or update when a sixth is
added.  A topic that REQUIRES a name says so by returning a data dict carrying
an `error` key; the CLI prints the text on stderr and exits 2, exactly as a
built-in `named` scope does.

### Extending a built-in topic

Set `extends` to a built-in topic's name and your renderer's output is appended
to it as an attributed section:

```python
class PremiumPathsSection:
    name = "premium-paths"
    extends = "paths"                 # appended to `explain paths`
    ...
```

This is usually the half that matters.  "Where does this file go" is asked at
`explain paths`, not at a topic the reader has not heard of yet.

Three rules the CLI enforces, so a contributor cannot get them wrong:

- **the built-in renders first and unchanged** — its text is yours to follow,
  never to replace;
- **your data lands at `data["extensions"][<name>]`** — never merged into the
  built-in's own keys, so you cannot silently redefine what a documented key
  means for a `--json` consumer branching on it;
- **a section that raises is reported in place and the built-in still prints** —
  installing a package must not be able to delete the framework's own docs.

### What the CLI does for you

The topic reaches the dispatch, the overview banner, `explain --help` and the
unknown-scope error from one merged table, so it cannot be advertised without
being served or served without being advertised.  Contributed topics are marked
with their distribution (`<- jaato-premium`) wherever they are listed.  A
built-in name always wins a collision; a topic that fails to load is skipped
with a warning rather than breaking the CLI; and an `extends` naming a topic
that does not exist is inert rather than an error.

### Compute it, do not write it down

The reason a topic beats a README is that it can be *read off the installed
code*, so it cannot describe a version you are not running.  Prefer
`inspect`/`dataclasses.fields` over a hand-typed table, hoist a constant the
engine uses so the topic and the engine share one declaration, and where a
fact is a consequence of two behaviours, **probe it** and render the result.
Anything you genuinely cannot compute should say so rather than guess — a
reader told something the engine does not do loses more time than one told
nothing.
