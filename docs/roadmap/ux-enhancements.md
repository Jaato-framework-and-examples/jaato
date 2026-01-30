# UX Enhancement Tickets

Implementation tickets for TUI features inspired by pi-mono analysis.

---

## Ticket 1: External Editor Integration (Ctrl+G)

**Priority**: High
**Effort**: Low
**Dependencies**: None (prompt_toolkit built-in)

### Description

Allow users to open the current prompt in their external editor (`$EDITOR` or `$VISUAL`) for complex multi-line editing.

### User Story

As a power user, I want to press Ctrl+G to open my prompt in vim/emacs/vscode so I can use my familiar editor keybindings for complex prompts.

### Implementation

1. **File**: `rich-client/pt_display.py`

2. **Changes to Buffer creation** (~line 507):
   ```python
   self._input_buffer = Buffer(
       completer=input_handler._completer if input_handler else None,
       history=input_handler._pt_history if input_handler else None,
       complete_while_typing=True if input_handler else False,
       enable_history_search=True,
       tempfile_suffix='.md',  # Syntax highlighting hint
   )
   ```

3. **Add keybinding**:
   ```python
   @kb.add('c-g')
   def open_in_editor(event):
       """Open current input in external editor."""
       event.app.current_buffer.open_in_editor()
   ```

4. **Add to configurable keybindings** in `keybindings.py`:
   - Action: `open_editor`
   - Default: `c-g`

### Acceptance Criteria

- [ ] Ctrl+G opens current buffer content in `$EDITOR`
- [ ] Edited content returns to input buffer on editor close
- [ ] Works with vim, nano, emacs, vscode
- [ ] Keybinding is configurable via `keybindings.json`
- [ ] Document in CLAUDE.md keybindings section

---

## Ticket 2: Fuzzy @file Completion

**Priority**: High
**Effort**: Medium
**Dependencies**: Optional `fd` command for best performance

### Description

Replace basic path completion with fuzzy search when typing `@`. Users can type partial filenames and get scored matches.

### User Story

As a developer, I want to type `@authctrl` and see `src/controllers/auth_controller.py` as a completion so I don't need to remember exact paths.

### Implementation

1. **File**: `simple-client/file_completer.py`

2. **Add fuzzy scoring function**:
   ```python
   def fuzzy_score(query: str, candidate: str) -> int:
       """Score candidate against query. Higher = better match."""
       query_lower = query.lower()
       candidate_lower = candidate.lower()
       basename = os.path.basename(candidate_lower)

       # Exact basename match
       if basename == query_lower:
           return 100
       # Basename prefix
       if basename.startswith(query_lower):
           return 90
       # Full path prefix
       if candidate_lower.startswith(query_lower):
           return 80
       # Basename contains
       if query_lower in basename:
           return 60
       # Full path contains
       if query_lower in candidate_lower:
           return 50
       # Fuzzy character sequence match
       qi = 0
       for c in candidate_lower:
           if qi < len(query_lower) and c == query_lower[qi]:
               qi += 1
       if qi == len(query_lower):
           return 30
       return 0
   ```

3. **Add `fd` integration** (optional, with fallback):
   ```python
   def find_files_fd(query: str, cwd: str, limit: int = 50) -> list[str]:
       """Use fd for fast .gitignore-aware search."""
       try:
           result = subprocess.run(
               ['fd', '--type', 'f', '--hidden', '--follow',
                '--max-results', str(limit * 2), query],
               cwd=cwd, capture_output=True, text=True, timeout=2
           )
           return result.stdout.strip().split('\n')[:limit]
       except (subprocess.TimeoutExpired, FileNotFoundError):
           return None  # Fallback to glob
   ```

4. **Modify `AtFileCompleter.get_completions()`**:
   - Try `fd` first if available
   - Fall back to walking directory tree
   - Score and sort results
   - Limit to top 20 matches

### Acceptance Criteria

- [ ] Typing `@` followed by partial name shows fuzzy matches
- [ ] Results sorted by relevance score
- [ ] Works without `fd` (slower fallback)
- [ ] Respects .gitignore when `fd` available
- [ ] Completion menu shows path + file type
- [ ] Directory bonus in scoring (+10 for directories)

---

## Ticket 3: Text Search in Session History

**Priority**: Medium
**Effort**: Medium
**Dependencies**: None

### Description

Add Ctrl+F search to find text across all messages in the current session output.

### User Story

As a user with a long session, I want to search for "authentication" to find where we discussed the auth approach without scrolling through hundreds of lines.

### Implementation

1. **New file**: `rich-client/search_overlay.py`
   ```python
   @dataclass
   class SearchMatch:
       line_idx: int
       start_pos: int
       end_pos: int

   class SearchOverlay:
       def __init__(self, output_buffer: OutputBuffer):
           self.buffer = output_buffer
           self.query = ""
           self.matches: list[SearchMatch] = []
           self.current_idx = 0
           self.visible = False

       def search(self, query: str) -> int:
           """Search all lines, return match count."""
           self.query = query.lower()
           self.matches = []
           for i, line in enumerate(self.buffer._lines):
               text = self._get_line_text(line).lower()
               start = 0
               while (pos := text.find(self.query, start)) != -1:
                   self.matches.append(SearchMatch(i, pos, pos + len(query)))
                   start = pos + 1
           self.current_idx = 0
           return len(self.matches)

       def next_match(self) -> Optional[int]:
           if not self.matches:
               return None
           self.current_idx = (self.current_idx + 1) % len(self.matches)
           return self.matches[self.current_idx].line_idx

       def prev_match(self) -> Optional[int]:
           if not self.matches:
               return None
           self.current_idx = (self.current_idx - 1) % len(self.matches)
           return self.matches[self.current_idx].line_idx
   ```

2. **Add search bar UI** in `pt_display.py`:
   - Floating bar at bottom when search active
   - Shows: `Search: [query] (3/15 matches)`
   - Input field for query

3. **Add keybindings**:
   - `c-f`: Open search
   - `enter` / `c-n`: Next match
   - `c-p` / `shift-enter`: Previous match
   - `escape`: Close search

4. **Highlight matches** in output:
   - Apply `search_match` style to matching text
   - Apply `search_current` style to current match

### Acceptance Criteria

- [ ] Ctrl+F opens search bar
- [ ] Incremental search as user types
- [ ] Match count displayed
- [ ] n/N navigate between matches
- [ ] Output scrolls to show current match
- [ ] Matches highlighted in output
- [ ] Escape closes search
- [ ] Search is case-insensitive

---

## Ticket 4: Bracketed Paste Handling

**Priority**: Low
**Effort**: Low-Medium
**Dependencies**: None

### Description

Detect large pastes and insert a placeholder to prevent UI freezing. Expand placeholder on submit.

### User Story

As a user pasting a large log file, I don't want the terminal to freeze while it processes 500 lines of text character by character.

### Implementation

1. **File**: `rich-client/pt_display.py` (keybindings section)

2. **Add paste registry**:
   ```python
   _paste_registry: dict[int, str] = {}
   _paste_counter: int = 0

   def register_paste(content: str) -> int:
       global _paste_counter
       _paste_counter += 1
       _paste_registry[_paste_counter] = content
       return _paste_counter

   def get_paste(paste_id: int) -> Optional[str]:
       return _paste_registry.pop(paste_id, None)
   ```

3. **Add custom paste handler**:
   ```python
   from prompt_toolkit.keys import Keys

   @kb.add(Keys.BracketedPaste)
   def handle_large_paste(event):
       data = event.data
       lines = data.count('\n')

       # Threshold: >10 lines OR >1000 chars
       if lines > 10 or len(data) > 1000:
           paste_id = register_paste(data)
           placeholder = f"[paste #{paste_id} +{lines} lines]"
           event.current_buffer.insert_text(placeholder)
       else:
           event.current_buffer.insert_text(data)
   ```

4. **Expand on submit** in input processing:
   ```python
   def expand_paste_placeholders(text: str) -> str:
       pattern = r'\[paste #(\d+) \+\d+ lines\]'
       def replace(m):
           paste_id = int(m.group(1))
           return get_paste(paste_id) or m.group(0)
       return re.sub(pattern, replace, text)
   ```

### Configuration

Add to settings:
```json
{
  "paste_threshold_lines": 10,
  "paste_threshold_chars": 1000
}
```

### Acceptance Criteria

- [ ] Large pastes (>10 lines) show placeholder
- [ ] Placeholder shows line count
- [ ] Content expands correctly on submit
- [ ] Small pastes work normally
- [ ] Thresholds configurable
- [ ] Multiple pastes in same prompt work

---

## Implementation Order

1. **External Editor** - Quick win, built-in support
2. **Fuzzy @file** - High daily impact
3. **Text Search** - Good for power users
4. **Bracketed Paste** - Polish feature

## Notes

- All features should be documented in CLAUDE.md
- Add to keybindings configuration where applicable
- Consider adding to design-philosophy.md if patterns emerge
