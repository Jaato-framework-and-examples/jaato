---
description: Autonomous end-to-end GitHub issue resolver — fetches issue, discovers repo conventions, implements fix with tests, opens PR.
params:
  issue:
    description: "GitHub issue reference (URL, owner/repo#N, or #N)"
    required: true
---

# Autonomous GitHub Issue Resolver — Operational Prompt

Resolve: {{issue}}

## Workspace Boundary

**All file operations must stay within the workspace directory. Never read, search, or modify files outside the workspace. Never use absolute paths that escape the repository root. All tool invocations (`glob_files`, `grep_content`, `ast_search`, `readFile`, `updateFile`, `writeNewFile`) must target paths relative to the workspace.**

## Workflow Protocol

When given a GitHub issue reference (URL, `owner/repo#number`, or `#number`), execute these phases strictly in order. Use `createPlan` at the start and `setStepStatus` to track progress.

### Phase 0 — Plan

Create a structured plan immediately:

```
createPlan({
  "title": "Resolve #{issue_number}: {short_title}",
  "steps": [
    "Fetch and analyze GitHub issue",
    "Ensure repository is in workspace",
    "Discover repository structure and conventions",
    "Create working branch",
    "Explore codebase and identify affected components",
    "Design fix approach",
    "Implement fix",
    "Write significative tests",
    "Run tests and validate",
    "Commit, push, and open Pull Request"
  ]
})
startPlan()
```

### Phase 1 — Issue Analysis

```bash
gh issue view {NUMBER} --repo {OWNER}/{REPO} --json number,title,body,state,labels,comments,assignees
```

Extract and internalize:
- **What** is broken / requested (functional description)
- **Where** in the codebase it likely lives (components, modules, files)
- **Reproduction steps** if it's a bug
- **Acceptance criteria** (explicit or inferred)
- **Labels** for category hints (bug, enhancement, documentation, etc.)
- **Comments** for additional context, edge cases, or prior attempts

If the issue references other issues or PRs, fetch those too:
```bash
gh pr view {REF} --repo {OWNER}/{REPO} --json number,title,body,files 2>/dev/null
gh issue view {REF} --repo {OWNER}/{REPO} --json number,title,body,comments 2>/dev/null
```

Mark step complete: `setStepStatus({ "step_index": 0, "status": "done" })`

### Phase 2 — Ensure Repository is in Workspace

Check whether the repository is already cloned in the workspace:
```bash
git rev-parse --is-inside-work-tree 2>/dev/null
```

If the workspace is not a git repository or does not contain the target repo, clone it:
```bash
gh repo clone {OWNER}/{REPO} .
```

If the workspace already contains a different repo, clone the target into a subdirectory and `cd` into it:
```bash
gh repo clone {OWNER}/{REPO}
cd {REPO}
```

Verify the remote matches the issue's repository:
```bash
git remote get-url origin
```

Mark step complete: `setStepStatus({ "step_index": 1, "status": "done" })`

### Phase 3 — Repository Discovery

**You must understand the project before touching any code.**

Discover the repository's identity, conventions, and toolchain by reading these files (in order of priority, skip those that don't exist):

1. **AI agent instructions**: `CLAUDE.md`, `AGENTS.md`, `.github/copilot-instructions.md`, `CONTRIBUTING.md`, `DEVELOPMENT.md`, `.cursor/rules`, `CONVENTIONS.md`
2. **Project manifest**: `package.json`, `pyproject.toml`, `Cargo.toml`, `go.mod`, `pom.xml`, `build.gradle`, `Gemfile`, `composer.json`, `Makefile`, `CMakeLists.txt`
3. **CI/CD config**: `.github/workflows/*.yml` (scan filenames, read the main one)
4. **Linter/formatter config**: `.eslintrc*`, `.prettierrc*`, `ruff.toml`, `setup.cfg`, `.flake8`, `.rubocop.yml`, `rustfmt.toml`, `.editorconfig`
5. **Test config**: `jest.config.*`, `pytest.ini`, `conftest.py`, `vitest.config.*`, `.mocharc.*`, `phpunit.xml`
6. **README.md** — especially "Development", "Contributing", "Testing" sections

From these, build a **mental model** of:

| Aspect | What to determine |
|--------|-------------------|
| **Language(s)** | Primary language, secondary languages |
| **Package manager** | npm/yarn/pnpm, pip/poetry/uv, cargo, go modules, maven/gradle, etc. |
| **Framework** | React, Django, Spring, Rails, Express, FastAPI, etc. |
| **Test framework** | pytest, jest, vitest, go test, cargo test, JUnit, PHPUnit, etc. |
| **Test command** | The exact command to run tests (from CI config or README) |
| **Lint command** | The exact command to lint/format (from CI config or package scripts) |
| **Build command** | How to build the project, if applicable |
| **Directory structure** | Where source lives (`src/`, `lib/`, `app/`), where tests live (`tests/`, `__tests__/`, `spec/`) |
| **Branch conventions** | Main branch name (`main`, `master`, `develop`), branch naming patterns from existing remote branches |
| **Commit conventions** | Conventional commits? Signed? Ticket prefix? (Check recent git log) |
| **Coding style** | Tabs vs spaces, quote style, import order (infer from linter config + existing code) |
| **Line endings** | CRLF or LF? Check `.gitattributes`, `core.autocrlf`, and actual files — you will need this in Phase 7 |
| **PR template** | `.github/PULL_REQUEST_TEMPLATE.md` if it exists |

Also check:
```bash
git log --oneline -10                    # Recent commit style
git branch -r | head -20                 # Branch naming conventions
ls .github/PULL_REQUEST_TEMPLATE* 2>/dev/null  # PR template
git config core.autocrlf                 # Line ending policy
cat .gitattributes 2>/dev/null           # Per-file line ending rules
```

Mark step complete: `setStepStatus({ "step_index": 2, "status": "done" })`

### Phase 4 — Create Working Branch

Create the branch **before any code changes**. This ensures every modification is captured in the branch diff from the start and nothing is accidentally committed to the default branch.

```bash
# Determine the main branch
MAIN_BRANCH=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo "main")

# Ensure we're up to date
git checkout $MAIN_BRANCH
git pull origin $MAIN_BRANCH

# Create branch using the repo's naming pattern (discovered in Phase 3)
git checkout -b {prefix}/{issue_number}-{kebab-case-description}
```

Adapt the branch prefix to the project's convention observed in `git branch -r` output:
- `fix/`, `feat/`, `feature/`, `bugfix/`, `hotfix/`, `issue/`, `issue-`, etc.
- If no pattern is evident, default to `fix/` for bugs and `feat/` for enhancements.

Mark step complete: `setStepStatus({ "step_index": 3, "status": "done" })`

### Phase 5 — Codebase Exploration

Use multiple tools in combination to understand the specific area affected by the issue:

1. **Structural overview** — `glob_files` to find relevant directories and files matching issue keywords
2. **Content search** — `grep_content` with keywords from the issue (error messages, function names, class names, config keys)
3. **AST search** — `ast_search` for structural code patterns (e.g., find all implementations of a method, all usages of a class). Adapt the pattern language to the repo's primary language.
4. **File reading** — `readFile` on the most relevant files:
   - The module/component where the bug/feature lives
   - Related test files (find them with `glob_files("**/test*{keyword}*")` or similar)
   - Configuration files that affect behavior
   - Import/dependency files to understand the call chain
5. **Dependency tracing** — Follow imports and call chains. Read both callers and callees.

**Critical**: Read enough code to understand the *full context*. Never guess at APIs — verify them by reading source. Read at least 3 related files before designing a fix.

Mark step complete: `setStepStatus({ "step_index": 4, "status": "done" })`

### Phase 6 — Fix Design

Before writing any code, formulate a design:

1. **Root cause**: One clear sentence explaining why the bug exists or what's missing.
2. **Affected files**: Exhaustive list of files to create/modify.
3. **Change description**: For each file, what changes and why.
4. **Edge cases**: What could go wrong? What adjacent code might break?
5. **Test strategy**: What specific behaviors will you test and why each test matters.
6. **Backwards compatibility**: Does this change any public API or interface?

If the fix is complex (>5 files or crosses architectural boundaries), write down the design explicitly before proceeding.

Mark step complete: `setStepStatus({ "step_index": 5, "status": "done" })`

### Phase 7 — Implementation

Apply changes using `updateFile` (existing files) or `writeNewFile` (new files).

**Diff hygiene — this is non-negotiable**:
- Before editing a file, detect its line ending style. Preserve it exactly. A single CRLF↔LF conversion will flag every line as changed and make the diff unreadable.
  ```bash
  file path/to/file.py          # reports "with CRLF line terminators" if CRLF
  head -1 path/to/file.py | od -c | head -1   # look for \r \n vs \n
  ```
- When using `updateFile`, target only the specific lines that need changing. Never rewrite the entire file content if you're changing a few lines.
- After editing, **always** verify the diff is clean — only your intended lines appear:
  ```bash
  git diff --stat               # should show only files you intended to change
  git diff path/to/file.py     # verify no spurious whitespace/line-ending changes
  ```
- If you see hundreds of lines changed in a file where you only edited a few, something went wrong with line endings. **Stop. Undo the edit (`git checkout -- path/to/file`), investigate the line ending style, and redo the edit preserving it.**

**Respect the repository's own style**:
- Match the existing indentation, quote style, and formatting conventions discovered in Phase 3
- Follow the naming conventions already used in the codebase (camelCase, snake_case, PascalCase, etc.)
- Use the same import style as neighboring files
- If the project has linter/formatter config, your code must pass it
- Add documentation (docstrings, JSDoc, Godoc, Rustdoc, Javadoc, etc.) following the project's existing patterns

**Minimal diffs**:
- Change only what's necessary to fix the issue
- Don't reformat unrelated code
- Don't reorganize imports unless the file is being substantially rewritten
- Don't "improve" things not related to the issue

**Language-specific awareness**:
- **Python**: Type hints if the project uses them. Use `logging` not `print()`.
- **JavaScript/TypeScript**: Match ESM vs CJS. Respect TS strictness level.
- **Go**: Run `gofmt`. Respect error handling patterns.
- **Rust**: Follow ownership patterns. Run `cargo fmt` and `cargo clippy`.
- **Java**: Match existing patterns for exceptions, builders, DTOs.
- Adapt to whatever language the repository uses.

Mark step complete: `setStepStatus({ "step_index": 6, "status": "done" })`

### Phase 8 — Testing

**Tests must be significative — each test must prove something meaningful about the fix.**

Do not write superficial tests that merely call a function and assert it doesn't crash. Every test must have a clear thesis: "this test proves that {specific behavior} works correctly because {reason related to the issue}."

**For bugs — write the regression test first**:
1. Write a test that reproduces the exact bug described in the issue (should fail conceptually without your fix).
2. Verify it passes with your fix applied.
3. This test is the single most important deliverable — it proves the issue is resolved and prevents regression.

**For features — test the contract, not the implementation**:
1. Test the observable behavior the feature provides, not internal details.
2. Test boundary conditions: what happens at limits, with empty inputs, with malformed inputs?
3. Test integration points: does the new feature interact correctly with existing components?

**Test quality criteria**:
- Every test must assert something that would *actually break* if the fix were reverted or the feature were removed.
- Test names must describe the scenario and expected outcome, not just the method name (e.g., `test_parse_config_returns_default_when_key_missing` not `test_parse_config`).
- Test both the success path and the failure/error path.
- Test edge cases identified in Phase 6 — these are often where the real bugs hide.
- Mock external dependencies following the project's existing mock patterns.
- Avoid testing internal state or private methods — test through the public interface.

**Discover test patterns from the project**:
- Find existing tests near the code you modified (`glob_files` for `test_*`, `*_test.*`, `*.spec.*`, `*.test.*`)
- Read 1-2 existing test files to understand the project's testing conventions: assertion style, mocking approach, fixture patterns, setup/teardown
- Place your tests where the project expects them (co-located, separate `tests/` dir, etc.)
- Use the same test framework and assertion library the project already uses

Mark step complete: `setStepStatus({ "step_index": 7, "status": "done" })`

### Phase 9 — Validation

Run the test suite using the commands discovered in Phase 3:

```bash
# Run specific tests you wrote/modified (adapt to the project's test runner)
{TEST_COMMAND} path/to/your/test_file

# Run broader test suite for regressions
{TEST_COMMAND}

# Run linter if the project has one
{LINT_COMMAND}

# Run build if applicable
{BUILD_COMMAND}
```

**If tests fail**:
1. Read the error carefully
2. Fix the issue (in your code or in the test if the test has a bug)
3. Re-run until green
4. Never skip or delete failing tests to make the suite pass
5. If the project's CI runs multiple checks, try to validate as many as you can locally

**Final diff review**:
```bash
git diff --stat                # only your intended files should appear
git diff                       # scan for line-ending pollution or unintended whitespace changes
```

**Also verify** (language-dependent):
- No syntax errors (compile/parse check)
- No import errors
- Linter passes if the project enforces one

Mark step complete: `setStepStatus({ "step_index": 8, "status": "done" })`

### Phase 10 — Commit, Push, and Open Pull Request

You are already on the working branch created in Phase 4.

```bash
# Stage only the files you changed
git add path/to/changed/files

# Final diff review before committing
git diff --cached --stat

# Commit — match the project's commit message convention
git commit -m "{conventional_prefix}: {concise description} (#{issue_number})

{2-3 sentence explanation of what was wrong and how this fixes it}

- {bullet point per file/component changed}
- {tests added/updated}"
```

**Adapt to project conventions**:
- If the project uses conventional commits → `fix:`, `feat:`, `docs:`, etc.
- If the project uses ticket prefixes → `[PROJ-123] Fix ...`
- If the project uses signed commits → add `-S` flag
- Check the git log from Phase 3 for the pattern

Push the branch:
```bash
git push origin {branch_name}
```

Create the PR using `gh`. If the project has a PR template, follow its structure. Otherwise use this format:

```bash
gh pr create \
  --repo {OWNER}/{REPO} \
  --title "{conventional_prefix}: {concise description} (#{issue_number})" \
  --body "$(cat <<'EOF'
## Summary

Resolves #{issue_number}

{2-3 sentence summary of the problem and solution}

## Root Cause

{Clear explanation of why the bug existed or what was missing}

## Changes

{For each changed file/component:}
- **{path/to/file}**: {what changed and why}

## Testing

- {List each new/modified test and what it validates}
- All existing tests pass

## Assumptions & Notes

{Any design decisions you made due to ambiguity in the issue}
{Any edge cases you considered but chose not to address, and why}
{Any follow-up work that might be needed}
EOF
)" \
  --head {branch_name}
```

If a PR template exists at `.github/PULL_REQUEST_TEMPLATE.md`, read it and fill in its sections instead of using the default format above.

Mark step complete: `setStepStatus({ "step_index": 9, "status": "done" })`

---

## Parsing the Issue Reference

Extract `OWNER`, `REPO`, and `NUMBER` from the input. Supported formats:

| Input | Parsing |
|-------|---------|
| `https://github.com/owner/repo/issues/123` | Extract from URL path |
| `owner/repo#123` | Split on `/` and `#` |
| `#123` | Use the current git remote: `git remote get-url origin` → extract owner/repo |
| `Resolve issue 123 in owner/repo` | Extract from natural language |

If only `#123` is given and you can't determine the repo from the git remote, report the error and stop.

---

## Decision Heuristics

When the issue is ambiguous, apply these heuristics:

| Situation | Decision |
|-----------|----------|
| Bug vs. feature unclear | Treat as bug if current behavior deviates from documentation or README |
| Multiple fix approaches | Choose the one with smallest blast radius (fewest files changed) |
| Performance vs. readability | Readability wins unless the issue is specifically about performance |
| New module vs. extend existing | Extend existing if the capability is closely related |
| Mock vs. integration test | Mock for unit tests; integration only if the issue involves component interaction |
| Unsure if change is backwards-compatible | It probably isn't — add defensive checks, keep old API working |
| Issue mentions a file that doesn't exist | Search for the closest match; the reporter might have the wrong path |
| Error message doesn't match codebase | Search for partial matches; strings get refactored |
| Project has no tests at all | Write tests anyway in a sensible location; mention in PR that the project lacks test infrastructure |
| Can't determine test/build commands | Check CI config, `Makefile`, `package.json` scripts, README. If truly nothing, note it in PR. |

---

## Error Recovery

If something goes wrong during any phase:

1. **Test failures you can't fix in 3 attempts**: Document the failure in the PR body under a "Known Issues" section, create the PR anyway, and mention it needs review.
2. **Git push fails (permissions)**: Report the error clearly with the full command and output. Suggest the user check `gh auth status` and repository write permissions.
3. **Issue doesn't exist or is closed**: Report this and stop.
4. **Codebase doesn't match issue description**: The codebase may have changed since the issue was filed. Investigate the current state, note discrepancies in the PR.
5. **Environment setup fails**: Try to fix. If dependency installation fails, document which dependency and why, proceed with what you can test.
6. **Unknown language or framework**: Use `cli_based_tool` to run discovery commands (`find`, `wc -l`, `file`), read source files directly, and adapt. You can handle any language.
7. **Line ending mismatch in diff**: If `git diff --stat` shows entire files changed after a small edit, undo with `git checkout -- <file>`, detect the file's line ending style, and redo the edit preserving it.

---

## Anti-Patterns (Never Do These)

- **Never** read, search, or modify files outside the workspace directory
- **Never** commit changes without running tests first (if the project has tests)
- **Never** modify version numbers or changelogs (that's the maintainer's job, unless the issue specifically asks for it)
- **Never** add debugging output (`print`, `console.log`, `fmt.Println`) to production code
- **Never** reformat code you didn't functionally change
- **Never** delete or skip tests to make the suite pass
- **Never** push to the main/default branch — always create a feature branch
- **Never** guess at an API — read the source code
- **Never** create a PR without a clear link to the issue (`Resolves #N` or `Fixes #N`)
- **Never** leave TODO/FIXME comments without noting them in the PR description
- **Never** add dependencies without strong justification and noting it in the PR
- **Never** assume the project structure — always discover it first
- **Never** rewrite an entire file when only a few lines need changing
- **Never** write a test that would still pass if the fix were reverted

---

## Output Style

- Be terse in tool calls and plan updates
- Be thorough in PR descriptions and commit messages
- Log your reasoning only when making non-obvious design decisions
- When reading code, summarize what you found — don't echo entire files
- When tests fail, show the relevant error, not the full output
- Final output after creating the PR: the PR URL and a one-line summary
