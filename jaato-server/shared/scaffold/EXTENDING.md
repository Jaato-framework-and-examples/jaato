# Extending `jaato-scaffold` with a verb

`jaato-scaffold` ships three built-in verbs — `explain`, `validate`, `new`.
External packages (the premium `compile` verb, or your own) add verbs **without
modifying this repo** by registering a `ScaffoldVerb` under the
`jaato.scaffold_verbs` entry-point group. The CLI discovers and mounts them at
startup; a verb whose package is not installed simply does not appear. This is
the same convention as `jaato.premium` / `jaato.premium_reactors` elsewhere in
the framework.

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
