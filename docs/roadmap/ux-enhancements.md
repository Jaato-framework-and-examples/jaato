# UX Enhancement Tickets

Implementation tickets for TUI features inspired by pi-mono analysis.

---

## Ticket 1: External Editor Integration (Ctrl+G) [COMPLETED]

**Priority**: High
**Effort**: Low
**Dependencies**: None (prompt_toolkit built-in)
**Status**: Implemented in commit 12aeaec

### Description

Allow users to open the current prompt in their external editor (`$EDITOR` or `$VISUAL`) for complex multi-line editing.

### User Story

As a power user, I want to press Ctrl+G to open my prompt in vim/emacs/vscode so I can use my familiar editor keybindings for complex prompts.

### Implementation (Completed)

1. **File**: `rich-client/keybindings.py`
   - Added `open_editor: "c-g"` to DEFAULT_KEYBINDINGS
   - Added `open_editor` field to KeybindingConfig dataclass

2. **File**: `rich-client/pt_display.py`
   - Added keybinding handler using `event.current_buffer.open_in_editor()`

3. **File**: `CLAUDE.md`
   - Documented keybinding in keybindings section

### Acceptance Criteria

- [x] Ctrl+G opens current buffer content in `$EDITOR`
- [x] Edited content returns to input buffer on editor close
- [x] Works with vim, nano, emacs, vscode
- [x] Keybinding is configurable via `keybindings.json`
- [x] Document in CLAUDE.md keybindings section

---

## Ticket 2: Fuzzy @file and %prompt Completion [COMPLETED]

**Priority**: High
**Effort**: Medium
**Dependencies**: None
**Status**: Implemented

### Description

Replace basic prefix completion with fuzzy search for both `@file` and `%prompt` completers. Users can type partial names with non-consecutive characters and get scored matches.

### User Story

As a developer, I want to type `@utl` and see `utils.py` as a completion, and type `%cr` to see `code-review` as a prompt, so I don't need to remember exact names.

### Implementation (Completed)

1. **File**: `simple-client/file_completer.py`

2. **Added `FuzzyMatcher` utility class**:
   - Shared by both `AtFileCompleter` and `PercentPromptCompleter`
   - Characters must appear in order (not necessarily consecutive)
   - Scoring bonuses: consecutive matches (+10), word boundaries (+10), start of text (+15)
   - Gap penalty (-1 per character between matches)
   - Case-insensitive matching

3. **Updated `AtFileCompleter`**:
   - Replaced `PathCompleter` with custom directory listing
   - Uses `FuzzyMatcher.filter_and_sort()` for matching and ranking
   - Skips hidden files (starting with `.`)
   - Shows directories with trailing `/` in display

4. **Updated `PercentPromptCompleter`**:
   - Replaced `startswith()` with `FuzzyMatcher.filter_and_sort()`
   - Results sorted by match score

5. **Tests**: `simple-client/tests/test_fuzzy_matcher.py`
   - 26 tests covering FuzzyMatcher, AtFileCompleter fuzzy, PercentPromptCompleter fuzzy

### Acceptance Criteria

- [x] Typing `@` followed by partial name shows fuzzy matches
- [x] Typing `%` followed by partial name shows fuzzy matches
- [x] Results sorted by relevance score
- [x] Completion menu shows file type metadata
- [x] Shared `FuzzyMatcher` utility for both completers
- [x] Comprehensive test coverage

---

## Ticket 3: Text Search in Session History [COMPLETED]

**Priority**: Medium
**Effort**: Medium
**Dependencies**: None
**Status**: Implemented

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

### Implementation (Completed)

1. **File**: `rich-client/output_buffer.py`
   - Added search state: `_search_query`, `_search_matches`, `_search_current_idx`
   - Added methods: `search()`, `search_next()`, `search_prev()`, `clear_search()`, `get_search_status()`
   - Added helper: `_scroll_to_match()` to center current match in viewport

2. **File**: `rich-client/keybindings.py`
   - Added keybindings: `search`=c-f, `search_next`=enter, `search_prev`=c-p, `search_close`=escape

3. **File**: `rich-client/pt_display.py`
   - Added search mode state and UI
   - Added keybinding handlers for search mode
   - Search prompt shows status: `Search (1/5) [Enter: next, Ctrl+P: prev, Esc: close]>`

4. **File**: `rich-client/theme.py` and theme JSON files
   - Added styles: `search_prompt`, `search_match`, `search_match_current`

### Acceptance Criteria

- [x] Ctrl+F opens search bar
- [x] Incremental search as user types
- [x] Match count displayed
- [x] Enter/Ctrl+P navigate between matches
- [x] Output scrolls to show current match
- [x] Matches highlighted in output
- [x] Escape closes search
- [x] Search is case-insensitive

---

## Ticket 4: Bracketed Paste Handling [COMPLETED]

**Priority**: Low
**Effort**: Low-Medium
**Dependencies**: None
**Status**: Implemented

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

- [x] Large pastes (>10 lines) show placeholder
- [x] Placeholder shows line count
- [x] Content expands correctly on submit
- [x] Small pastes work normally
- [x] Thresholds configurable (via instance attributes)
- [x] Multiple pastes in same prompt work

---

## Implementation Order

1. ~~**External Editor** - Quick win, built-in support~~ [COMPLETED]
2. ~~**Fuzzy @file and %prompt** - High daily impact~~ [COMPLETED]
3. ~~**Text Search** - Good for power users~~ [COMPLETED]
4. ~~**Bracketed Paste** - Polish feature~~ [COMPLETED]

## Notes

- All features should be documented in CLAUDE.md
- Add to keybindings configuration where applicable
- Consider adding to design-philosophy.md if patterns emerge
