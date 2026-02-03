# Self-Documenting System Design

This document explains the architecture and design decisions behind Jaato's self-documenting user guide system.

## Philosophy

The user guide should:
1. **Stay current automatically** - Update when code changes
2. **Use AI to write itself** - Leverage Jaato's own capabilities
3. **Be fully automated** - Minimal manual maintenance
4. **Produce quality output** - Academic-style LaTeX PDF

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Build Pipeline                          │
│  (build_pdf.sh)                                            │
└───┬─────────────────────────────────────────────────────────┘
    │
    ├──> 0. Auto-Documentation (auto_document.py)
    │    │   ├─> Detect missing chapters
    │    │   ├─> Check dependency changes (hash comparison)
    │    │   ├─> Connect to AI model (JaatoClient headless)
    │    │   ├─> AI reads codebase with tools
    │    │   └─> Generate/update markdown chapters
    │    │
    ├──> 1. Content Extraction (generate_docs.py)
    │    │   ├─> Parse rich_client.py for commands
    │    │   ├─> Scan for os.getenv() calls
    │    │   ├─> Extract keybinding definitions
    │    │   └─> Generate markdown tables
    │    │
    ├──> 2. Screenshot Generation (generate_screenshots.py)
    │    │   ├─> Start server + client (pexpect)
    │    │   ├─> Execute command scenarios
    │    │   ├─> Capture terminal output
    │    │   └─> Save as SVG/PNG
    │    │
    ├──> 3. Markdown Combination
    │    │   ├─> Concatenate chapters/ in order
    │    │   ├─> Insert generated/ content
    │    │   └─> Output combined markdown
    │    │
    ├──> 4. LaTeX Conversion (Pandoc)
    │    │   ├─> Parse markdown
    │    │   ├─> Apply LaTeX template
    │    │   ├─> Syntax highlighting
    │    │   └─> Output .tex file
    │    │
    └──> 5. PDF Compilation (pdflatex)
         └─> 3 passes for references/TOC
```

### Data Flow

```
Codebase Changes
      │
      v
Dependency Tracking (SHA-256 hashes)
      │
      v
Change Detection (compare hashes)
      │
      ├─> No change: Use existing chapter
      │
      └─> Changed: Trigger AI update
            │
            v
      AI Generation (Jaato headless mode)
            │
            ├─> Read codebase (File tools, Grep, etc.)
            ├─> Analyze implementation
            ├─> Generate markdown content
            └─> Save to chapters/
                  │
                  v
            State Update (doc-state.json)
                  │
                  v
            Build Pipeline continues...
```

## Key Design Decisions

### 1. Headless Jaato for AI Generation

**Decision**: Use JaatoClient in headless mode to generate documentation.

**Rationale**:
- Dogfooding: Use Jaato to document Jaato
- Tool access: AI can read files, search code, analyze structure
- Consistency: Same AI capabilities as interactive usage
- Automation: Scriptable, no manual interaction needed

**Implementation**:
```python
client = JaatoClient()
client.connect(project="...", model="gemini-2.5-flash")
# No configure_tools() - pure text generation
response = client.send_message(prompt, on_output=collect_output)
```

### 2. Dependency-Based Invalidation

**Decision**: Track file/directory hashes to detect when updates are needed.

**Rationale**:
- Efficiency: Don't regenerate unchanged content
- Precision: Update only what's affected
- Transparency: Clear what triggered updates
- Cost savings: Fewer AI calls

**Implementation**:
```python
def _has_dependencies_changed(self, chapter: Chapter) -> Tuple[bool, List[str]]:
    for dep in chapter.depends_on:
        current_hash = compute_hash(dep)
        stored_hash = state.dependency_hashes.get(dep)
        if current_hash != stored_hash:
            return True, [dep]
    return False, []
```

### 3. Manifest-Driven Content

**Decision**: Define chapters and dependencies in `manifest.json`.

**Rationale**:
- Declarative: Easy to see what exists and relationships
- Extensible: Add chapters without code changes
- Customizable: Per-chapter prompts and settings
- Maintainable: Single source of truth

**Schema**:
```json
{
  "number": "06",
  "filename": "06-basic-commands.md",
  "title": "Basic Commands",
  "description": "...",
  "depends_on": ["rich-client/commands/", "rich-client/rich_client.py"],
  "prompt_template": null
}
```

### 4. Multi-Stage Extraction

**Decision**: Separate extraction stage for structured data.

**Rationale**:
- Reliability: Parsing code is more reliable than AI extraction
- Completeness: Ensure all env vars/commands are captured
- Speed: Faster than AI generation
- Accuracy: No AI hallucination for factual data

**What's extracted**:
- Commands: From command registry/docstrings
- Environment variables: From `os.getenv()` calls
- Keybindings: From config initialization
- Tool schemas: From plugin registry (future)

### 5. Automated Screenshots with pexpect

**Decision**: Drive rich client with pexpect to capture real terminal output.

**Rationale**:
- Authenticity: Real UI, not mockups
- Automation: No manual screenshot process
- Consistency: Same appearance each build
- Version tracking: Screenshots update with UI changes

**Approach**:
```python
child = pexpect.spawn(rich_client_command)
child.sendline("command")
time.sleep(wait_for_output)
child.sendline("screenshot nosend format svg")
```

### 6. LaTeX for Professional Output

**Decision**: Use LaTeX (via Pandoc) instead of direct PDF generation.

**Rationale**:
- Quality: Superior typesetting (academic standard)
- Features: TOC, index, cross-references, bookmarks
- Customization: Full control over layout
- Maintainability: Template-based styling

**Trade-offs**:
- Complexity: Requires LaTeX installation
- Build time: Slower than HTML/markdown
- Learning curve: LaTeX syntax for customization

### 7. State Persistence

**Decision**: Track generation state in `doc-state.json`.

**Rationale**:
- Incrementality: Only update what changed
- Debugging: See when/why chapters were generated
- Audit trail: Log of all generations
- Recovery: Can detect and fix corruption

**State tracked**:
```json
{
  "last_updated": "2024-01-15T10:30:00",
  "chapter_hashes": {"06-basic-commands.md": "abc123..."},
  "dependency_hashes": {"rich-client/commands/": "def456..."},
  "generation_log": [
    {"timestamp": "...", "chapter": "...", "mode": "update", "success": true}
  ]
}
```

## AI Prompt Engineering

### Creation Prompt Strategy

For new chapters:
1. **Context**: Describe chapter purpose and scope
2. **Requirements**: Formatting, style, structure rules
3. **Conventions**: Markdown conventions for code/paths/vars
4. **Dependencies**: Files/dirs to analyze
5. **Instruction**: "Read codebase and generate"

**Key elements**:
- Explicit about target audience (beginners + advanced)
- Clear formatting conventions (commands in backticks, etc.)
- Examples of structure expected
- Directive to output pure markdown (no preamble)

### Update Prompt Strategy

For existing chapters:
1. **Current content**: Show existing chapter
2. **Changes**: List which dependencies changed
3. **Task**: Update to reflect current state
4. **Constraints**: Maintain structure, style, detail level

**Key elements**:
- Preserve good existing content
- Only change what needs updating
- Maintain consistency with rest of guide

### Prompt Customization

Chapters can override default prompts in manifest:

```json
{
  "number": "15",
  "filename": "15-headless-mode.md",
  "prompt_template": "You are writing about headless mode API.\n\nFocus on:\n- API methods\n- Code examples\n- Use cases\n\nRead shared/jaato_client.py for API details..."
}
```

## Content Organization

### Chapter Structure

**Part I: Getting Started** (Chapters 1-3)
- Target: First-time users
- Goal: Running first conversation
- Style: Tutorial, step-by-step

**Part II: Core Features** (Chapters 4-8)
- Target: Regular users
- Goal: Understanding main features
- Style: Reference with examples

**Part III: Customization** (Chapters 9-14)
- Target: Power users
- Goal: Personalizing experience
- Style: Configuration reference

**Part IV: Advanced** (Chapters 15-17)
- Target: Developers, integrators
- Goal: Programmatic usage, optimization
- Style: Technical reference

**Part V: Support** (Chapters 18-19)
- Target: All users
- Goal: Problem solving
- Style: FAQ, troubleshooting

### Appendices

- **Appendix A-C**: Auto-generated tables (commands, keybindings, env vars)
- **Appendix D-E**: Hand-written references (provider comparison, glossary)

## Build Performance

### Optimization Strategies

1. **Skip flags**: Allow skipping expensive steps during iteration
   ```bash
   ./build_pdf.sh --skip-autodoc --skip-screenshots
   ```

2. **Incremental updates**: Only regenerate changed chapters

3. **Caching**: SHA-256 hashes avoid redundant work

4. **Parallel potential**: Screenshots could run in parallel (future)

### Typical Build Times

| Mode | Time | Use Case |
|------|------|----------|
| Full (with autodoc) | 10-30 min | CI, releases |
| Fast (no autodoc) | 2-5 min | Local iteration |
| PDF only | 30-60 sec | Layout changes |
| Single chapter | 1-3 min | Content updates |

Times vary based on:
- Number of chapters needing updates
- AI model response time (Gemini fast, Opus slow)
- LaTeX compilation complexity

## CI/CD Integration

### GitHub Actions Workflow

**Triggers**:
- Push to main (relevant paths)
- Pull requests
- Manual dispatch

**Jobs**:
1. **build-pdf**: Generate PDF and HTML
2. **validate-docs**: Check PDF integrity

**Artifacts**:
- Upload PDF/HTML for 90 days
- Attach to GitHub releases

**Secrets required**:
- `GCP_PROJECT_ID`, `GCP_LOCATION`, `GOOGLE_APPLICATION_CREDENTIALS`
- Optional: `ANTHROPIC_API_KEY` for Anthropic provider

**Fallback**: If no AI credentials, skip auto-doc and use existing chapters.

## Error Handling

### Failure Modes

1. **AI generation fails**: Keep existing chapter, log error, continue build

2. **Dependency missing**: Log warning, use empty hash, trigger update next time

3. **LaTeX compilation fails**: Show error, output log, exit with code 1

4. **Screenshot capture fails**: Log warning, continue without screenshots

### Recovery Strategies

**Corrupt state**:
```bash
rm docs/user-guide/generated/doc-state.json
./auto_document.py --force
```

**Missing chapters**:
```bash
./auto_document.py  # Will detect and generate missing
```

**Bad LaTeX output**:
```bash
# Check generated LaTeX
cat docs/user-guide/build/user-guide.tex

# Rebuild with verbose output
cd docs/user-guide/build && pdflatex user-guide.tex
```

## Future Enhancements

### Planned

1. **Incremental screenshots**: Only regenerate changed scenarios
2. **Multi-provider autodoc**: Support Anthropic, Ollama in CI
3. **Parallel chapter generation**: Speed up full rebuilds
4. **Interactive examples**: Embed runnable code snippets
5. **Version history**: Track changes over time
6. **Translation pipeline**: Multi-language support

### Possible

- **Video generation**: Screen recordings of workflows
- **Interactive HTML**: Searchable, collapsible sections
- **ePub format**: E-reader support
- **Diff reports**: Show what changed between versions
- **A/B testing**: Generate multiple variants, pick best

## Maintenance Guidelines

### When to Regenerate

**Always regenerate**:
- Before releases
- After major features
- When dependencies change significantly

**Don't regenerate**:
- For typo fixes in code comments
- For unrelated codebase changes
- During active development (use fast mode)

### Updating the System

**Adding new chapter**:
1. Add to `manifest.json`
2. Run `./auto_document.py --chapter new-chapter.md`
3. Review and refine
4. Commit chapter + manifest

**Changing dependencies**:
1. Update `depends_on` in manifest
2. State will trigger update automatically on next build

**Modifying prompts**:
1. Update default prompts in `auto_document.py._build_creation_prompt()`
2. Or add `prompt_template` to chapter in manifest

### Quality Assurance

Before committing generated content:
1. **Read AI-generated chapters** - Check for accuracy
2. **Test code examples** - Ensure they work
3. **Verify cross-references** - Check links between sections
4. **Review screenshots** - Ensure they're representative
5. **Build PDF locally** - Check layout and formatting

## Conclusion

This self-documenting system:
- **Reduces maintenance burden** through automation
- **Ensures accuracy** by reading actual code
- **Stays current** with dependency tracking
- **Produces professional output** with LaTeX
- **Leverages Jaato's own capabilities** for generation

The system is designed to be:
- **Transparent**: Clear what's automated vs manual
- **Controllable**: Fine-grained control over generation
- **Extensible**: Easy to add chapters and features
- **Reliable**: Graceful degradation and error recovery

By using Jaato to document itself, we demonstrate its capabilities while creating comprehensive, always-current user documentation.
