---
description: Perform a structured code review on recent changes
params:
  scope:
    required: false
    default: staged
    description: What to review -- 'staged' (git staged changes), 'branch' (all commits on current branch), or a file path
  focus:
    required: false
    default: all
    description: Review focus -- 'all', 'security', 'performance', 'readability', or 'tests'
tags: ['code-review', 'quality', 'development']
---

Perform a thorough code review of the **{{scope}}** changes.

Focus area: **{{focus}}**

## Review Process

1. **Read the changes**: Use `git diff` (for staged) or `git diff main...HEAD`
   (for branch) to see all modified code.

2. **Analyze each file** for:
   - **Correctness**: Logic errors, off-by-one bugs, null/undefined handling
   - **Security**: Injection vulnerabilities, hardcoded secrets, unsafe deserialization
   - **Performance**: N+1 queries, unnecessary allocations, missing indexes
   - **Readability**: Unclear naming, missing context, overly complex logic
   - **Tests**: Missing test coverage for new behavior, brittle assertions

3. **Produce a review report** with this structure:

   ### Summary
   One paragraph overview of the changes and overall quality.

   ### Issues Found
   For each issue:
   - **File:line** -- severity (critical/warning/suggestion)
   - Description of the problem
   - Suggested fix (code snippet if helpful)

   ### Positive Notes
   Briefly mention well-written parts (good abstractions, thorough tests, etc.)

   ### Verdict
   One of: APPROVE, REQUEST_CHANGES, or COMMENT

Be constructive. Distinguish between must-fix issues and style suggestions.
