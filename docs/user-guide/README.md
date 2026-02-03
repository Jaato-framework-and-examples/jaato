# Jaato Rich Client User Guide

This directory contains the **self-documenting** user guide for the Jaato rich client.

## Overview

The user guide is:
- **Auto-generated**: Missing chapters are written by AI analyzing the codebase
- **Auto-updated**: Chapters update when their dependencies change
- **LaTeX-formatted**: Academic-style PDF with professional typesetting
- **Multi-format**: Available as PDF and HTML

## Directory Structure

```
docs/user-guide/
├── README.md                 # This file
├── manifest.json            # Chapter definitions and dependencies
├── chapters/                # Hand-written and AI-generated chapters
│   ├── 01-introduction.md
│   ├── 02-installation.md
│   └── ...
├── assets/
│   ├── screenshots/         # Auto-generated screenshots
│   └── diagrams/           # Architecture diagrams
├── generated/              # Auto-generated content
│   ├── commands.md         # Extracted from codebase
│   ├── env-vars.md         # Extracted from codebase
│   ├── keybindings.md      # Extracted from codebase
│   └── doc-state.json      # Generation state tracking
├── latex/
│   ├── template.tex        # LaTeX document template
│   └── style.css           # HTML styles
├── scripts/
│   ├── auto_document.py    # AI-powered doc generator
│   ├── generate_docs.py    # Extract content from code
│   ├── generate_screenshots.py  # Automated screenshot capture
│   └── build_pdf.sh        # Complete build pipeline
└── build/                  # Build artifacts (gitignored)
    ├── user-guide.pdf
    ├── user-guide.html
    └── ...
```

## Building the Documentation

### Prerequisites

```bash
# Python dependencies
.venv/bin/pip install -r requirements.txt

# System dependencies (Ubuntu/Debian)
sudo apt-get install texlive-latex-base texlive-latex-extra \
    texlive-fonts-recommended texlive-fonts-extra pandoc

# macOS
brew install --cask mactex
brew install pandoc
```

### Quick Build

```bash
# Full build (auto-doc + extract + screenshots + PDF)
./docs/user-guide/scripts/build_pdf.sh

# Skip auto-documentation (use existing chapters)
./docs/user-guide/scripts/build_pdf.sh --skip-autodoc

# Skip screenshots (faster for text-only updates)
./docs/user-guide/scripts/build_pdf.sh --skip-screenshots

# Force regenerate all chapters
./docs/user-guide/scripts/build_pdf.sh --force-autodoc

# Generate HTML too
./docs/user-guide/scripts/build_pdf.sh --html
```

### Output

- PDF: `docs/user-guide/user-guide.pdf`
- HTML: `docs/user-guide/user-guide.html` (if `--html`)

## Self-Documenting System

### How It Works

1. **Dependency Tracking**: Each chapter lists files/directories it depends on in `manifest.json`

2. **Change Detection**: `auto_document.py` computes hashes of dependencies

3. **AI Generation**: When dependencies change or chapters are missing:
   - Jaato client connects to AI model (in headless mode)
   - AI reads codebase using its tools
   - AI generates/updates markdown content
   - Content is saved to `chapters/`

4. **Content Extraction**: `generate_docs.py` extracts structured data:
   - Commands from command registry
   - Environment variables from `os.getenv()` calls
   - Keybindings from config

5. **Screenshot Generation**: `generate_screenshots.py` uses pexpect to:
   - Start server and client
   - Execute commands
   - Capture terminal output as SVG/PNG

6. **PDF Build**: `build_pdf.sh` orchestrates:
   - Auto-documentation → Content extraction → Screenshots
   - Markdown combination → Pandoc → LaTeX → PDF

### Manual Chapter Generation

Generate a specific chapter:

```bash
./docs/user-guide/scripts/auto_document.py \
    --chapter 06-basic-commands.md \
    --provider google_genai \
    --model gemini-2.5-flash
```

Generate all missing/stale chapters:

```bash
./docs/user-guide/scripts/auto_document.py
```

Force regenerate everything:

```bash
./docs/user-guide/scripts/auto_document.py --force
```

### Configuration

Edit `manifest.json` to:
- Add new chapters
- Update chapter dependencies
- Customize prompts for AI generation

Example chapter definition:

```json
{
  "number": "06",
  "filename": "06-basic-commands.md",
  "title": "Basic Commands",
  "description": "Essential user commands: sending messages, reset, model switching, theme, exit",
  "depends_on": ["rich-client/commands/", "rich-client/rich_client.py"]
}
```

When files in `depends_on` change, the chapter will be automatically updated on next build.

## CI/CD Integration

The `.github/workflows/build-docs.yml` workflow:

1. Triggers on changes to relevant files
2. Runs full build pipeline
3. Uploads PDF and HTML as artifacts
4. Attaches documentation to releases (on tags)

### Secrets Required

For AI-powered auto-documentation in CI:

```
GCP_PROJECT_ID: Your Google Cloud project ID
GCP_LOCATION: Vertex AI location (e.g., us-central1)
GOOGLE_APPLICATION_CREDENTIALS: Service account JSON (as secret)
```

Without these secrets, CI builds use existing chapters (skip auto-doc).

## Customization

### LaTeX Styling

Edit `latex/template.tex`:
- Change document class, fonts, colors
- Customize headers/footers
- Add custom macros
- Adjust spacing and layout

### Markdown Conventions

The guide uses these conventions:
- Commands: `` `command` ``
- File paths: `` `/path/to/file` ``
- Environment variables: `` `$VARIABLE_NAME` ``
- Keyboard shortcuts: `Ctrl+C` (plain text, LaTeX template adds formatting)

### Callout Boxes

Use standard markdown; these get converted to colored LaTeX boxes:

```markdown
> **Tip**: This is a helpful tip

> **Warning**: This is important to know

> **Note**: This is additional context
```

## Development Workflow

### Adding New Content

1. **Add chapter to manifest**:
   ```bash
   # Edit manifest.json, add new chapter definition
   ```

2. **Generate with AI**:
   ```bash
   ./docs/user-guide/scripts/auto_document.py --chapter new-chapter.md
   ```

3. **Review and refine**:
   ```bash
   # Edit chapters/new-chapter.md as needed
   ```

4. **Build and preview**:
   ```bash
   ./docs/user-guide/scripts/build_pdf.sh --skip-autodoc --skip-screenshots
   ```

### Updating Existing Content

Just edit the chapter markdown directly, or:

```bash
# Let AI update based on code changes
./docs/user-guide/scripts/auto_document.py --chapter existing-chapter.md
```

### Testing Changes

```bash
# Quick build (skip expensive steps)
./docs/user-guide/scripts/build_pdf.sh \
    --skip-autodoc \
    --skip-screenshots \
    --skip-extraction
```

## Maintenance

### State Management

The `generated/doc-state.json` file tracks:
- When each chapter was last generated
- Content hashes for change detection
- Dependency hashes
- Generation log

Delete this file to force full regeneration:

```bash
rm docs/user-guide/generated/doc-state.json
./docs/user-guide/scripts/auto_document.py
```

### Provider Configuration

Auto-documentation uses the same provider setup as normal Jaato usage.

For Google GenAI (default):
```bash
export PROJECT_ID=your-project
export LOCATION=us-central1
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

For Anthropic:
```bash
export ANTHROPIC_API_KEY=your-key
./docs/user-guide/scripts/auto_document.py --provider anthropic --model claude-sonnet-4
```

For Ollama (free, local):
```bash
ollama serve
ollama pull qwen2.5:32b
./docs/user-guide/scripts/auto_document.py --provider ollama --model qwen2.5:32b
```

## Troubleshooting

### PDF Build Fails

Check LaTeX log:
```bash
cat docs/user-guide/build/pdflatex.log | tail -n 50
```

Common issues:
- Missing LaTeX packages: Install `texlive-latex-extra`
- Long table overflow: Adjust table formatting in markdown
- Image not found: Check `assets/screenshots/` paths

### Auto-Documentation Fails

Check AI model access:
```bash
# Test provider connection
.venv/bin/python -c "
from shared.jaato_client import JaatoClient
client = JaatoClient()
client.connect(project='your-project', location='us-central1', model='gemini-2.5-flash')
print('✓ Connection successful')
"
```

### Screenshots Don't Generate

Requires `pexpect`:
```bash
.venv/bin/pip install pexpect
```

## Contributing

When modifying documentation automation:

1. **Test locally** before committing
2. **Update manifest.json** for new chapters
3. **Update this README** for new features
4. **Check CI passes** after pushing

## License

Same as main Jaato project.
