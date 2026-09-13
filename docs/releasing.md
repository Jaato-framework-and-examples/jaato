# Releasing

Four distributions ship from this repository — `jaato-sdk`, `jaato-server`,
`jaato-tui` and `jaato-eval` — each with its own `workflow_dispatch` workflow
under `.github/workflows/publish-pypi-<pkg>.yml`.

Every workflow takes a **`target`**: `testpypi` (the default) or `pypi`. The
intended shape is to stage first, let someone install and exercise the real
artifact, and only then publish:

```
bump + merge  ->  run with target: testpypi  ->  someone tests  ->  run the SAME
                                                                    workflow on the
                                                                    SAME commit with
                                                                    target: pypi
```

Re-running the same commit is what makes this worth doing. The staged build
carries a pre-release suffix (see below) and the PyPI build does not, so the two
artifacts differ in exactly one respect — the version string. Everything else is
the same file: the same tree, the same dependency pins, and the same generated
changelog, because the suffix is applied *after* `build_readme.py` runs and the
changelog is anchored on the version `pyproject.toml` declares.

`target: pypi` publishes that declared version and refuses a `suffix`, so an
`rcN` never reaches PyPI.

## Which packages need a release

Only the ones the merges since the last release actually touched:

```bash
last=$(git log --format=%H -1 --grep='^Bump jaato-')      # or the known release sha
for p in jaato-sdk jaato-server jaato-tui jaato-eval; do
  echo "=== $p: $(git log --oneline $last..origin/main -- $p/ | wc -l) commit(s)"
  git log --oneline $last..origin/main -- $p/
done
```

A package with an empty diff is not republished. Level follows the commits: any
`feat` in the range is a minor, fixes alone are a patch.

Preview what each changelog will say before committing the bump — it becomes the
package's PyPI long description, and it is not editable afterwards:

```bash
(cd jaato-sdk && python3 ../scripts/build_readme.py) && \
  sed -n '1,/^---$/p' jaato-sdk/PKG_README.md
rm -f jaato-*/PKG_README.md          # generated, gitignored
```

## Staging on TestPyPI

Run the workflow with `target: testpypi`. Publish **`jaato-sdk` first** when it
is part of the release: `jaato-server` and `jaato-tui` depend on it without a
version pin, so pip resolves whichever index offers the highest version.

The tester then needs both indexes, because the third-party dependencies
(`mcp`, `openai`, `anthropic`, ...) exist only on real PyPI:

```bash
python3 -m venv /tmp/stage && . /tmp/stage/bin/activate
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  'jaato-sdk==0.21.0rc1' 'jaato-server[all]==0.14.0rc1'
```

Pin the exact versions rather than relying on cross-index resolution picking the
staged build. The pin is also what makes pip accept a pre-release at all: an
exact `==` specifier naming one is honoured, while `jaato-server`'s own unpinned
dependency on `jaato-sdk` would otherwise skip every `rcN`. Then check the
artifact is what you think it is:

```bash
python -c "import importlib.metadata as m; print(m.version('jaato-server'))"
python -m server --status            # or whatever the change under test needs
```

## Publishing to PyPI

Re-run the same workflow, same commit, `target: pypi`. Then verify against what
the index actually serves, not against what the workflow said it uploaded:

```bash
curl -s https://pypi.org/pypi/jaato-server/0.14.0/json | \
  python3 -c 'import sys,json; d=json.load(sys.stdin); \
    print([u["packagetype"] for u in d["urls"]]); \
    print(d["info"]["description"].split("\n---\n")[0])'
```

## A version can be uploaded to an index once, ever

Deleting a release does not free its version number. The two indexes are
independent, so staging `0.20.0` on TestPyPI leaves `0.20.0` free on PyPI — but
a **second** staging attempt of the same release cannot reuse it.

That is what the **`suffix`** input is for: `rc1` builds and uploads `0.21.0rc1`
without touching the version the repository declares. Iterate `rc1`, `rc2`, ...
on TestPyPI, then publish the clean `0.21.0` to PyPI. The workflow refuses a
suffix when `target: pypi`, so the released version is always the one in
`pyproject.toml`.

## Every TestPyPI build is a pre-release

Leaving `suffix` blank on `target: testpypi` does **not** stage a plain version
— it stages `rc1`. There is no way to ask for an unsuffixed build on TestPyPI,
and that is deliberate.

PyPI pins the **THIS RELEASE** badge to the newest *stable* version whenever one
exists, and only falls through to the newest pre-release when there is none. So
a plain `0.21.0` staged on Monday outranks a `0.21.0rc1` staged on Friday: the
project page presents the older build as the current one, and a tester reading
it installs the wrong artifact. With nothing but pre-releases on the index the
newest upload is always the one shown.

It also stops a staging round from spending the number the release needs.
`0.21.0` staged plainly is `0.21.0` gone from TestPyPI forever, so the next
attempt at that same release has to be suffixed anyway — the rule just makes the
first attempt behave like the second.

`target: pypi` is unaffected: `suffix` stays blank there and is refused if set,
so what ships is the version `pyproject.toml` declares.

The suffix is applied **after** `build_readme.py` runs, deliberately: that
script anchors the changelog by walking git history for the commits declaring
the version in `pyproject.toml`, and a suffixed version matches no commit — the
walk then anchors on the bump commit itself and emits an empty changelog.

## Trusted publishers

Both indexes use PyPI trusted publishing (OIDC), so there are no tokens. A
publisher is keyed on **repository + workflow filename + environment name**, and
the workflow picks the environment from `target`:

| `target` | environment | index |
|----------|-------------|-------|
| `testpypi` | `testpypi-<pkg>` | test.pypi.org |
| `pypi` | `pypi-<pkg>` | pypi.org |

So each package needs a publisher registered on *each* index, both naming
`publish-pypi-<pkg>.yml` and the matching environment. Note that PyPI caps
**pending** publishers (for projects that do not exist yet) at three per
account; publishing converts a pending publisher into an ordinary one and frees
the slot.
