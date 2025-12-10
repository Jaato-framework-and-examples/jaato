# API Documentation Search

The jaato API documentation includes a full-text search feature powered by [Lunr.js](https://lunrjs.com/), a lightweight JavaScript search library.

## Features

- **Full-text search** across all 22 documentation pages
- **Instant results** as you type (with 300ms debounce)
- **Smart matching** with fuzzy search and relevance scoring
- **Keyboard navigation**:
  - Press `/` to focus the search box
  - Use `↓` and `↑` arrows to navigate results
  - Press `Escape` to close results
- **Result previews** with page title, section, and snippet
- **Fully self-contained** - works offline once loaded

## How It Works

1. **Search Index**: A pre-built JSON index (`docs/api/assets/js/search-index.json`) contains all page content, titles, headings, and snippets.

2. **Lunr.js**: Loaded from CDN (`https://unpkg.com/lunr@2.3.9/lunr.min.js`), Lunr.js provides the search engine that runs entirely in the browser.

3. **Search UI**: Custom JavaScript in `docs/api/assets/js/docs.js` handles:
   - Loading the index
   - Building the Lunr index
   - Performing searches
   - Displaying results with dropdown

4. **Styling**: CSS in `docs/api/assets/css/style.css` provides the search results dropdown appearance.

## File Structure

```
docs/
├── generate_search_index.py       # Generate search index from HTML
├── add_lunr_to_html.py           # Add Lunr.js script tag to HTML files
├── SEARCH_README.md              # This file
└── api/
    ├── assets/
    │   ├── css/
    │   │   └── style.css         # Includes search result styles
    │   └── js/
    │       ├── docs.js           # Includes search implementation
    │       └── search-index.json # Search index (77 KB)
    └── *.html                    # All HTML files include Lunr.js script
```

## Updating the Search Index

**When to update**: Whenever you add, modify, or remove documentation pages.

**How to update**:

```bash
# From the root of the jaato repository
python3 docs/generate_search_index.py
```

This will:
1. Scan all HTML files in `docs/api/`
2. Extract titles, headings, and text content
3. Generate a new `docs/api/assets/js/search-index.json` file

**Output example**:
```
============================================================
Lunr.js Search Index Generator
============================================================
Scanning /home/user/jaato/docs/api for HTML files...
Found 22 HTML files
  Processing: index.html
  Processing: core-concepts/providers.html
  ...
Successfully indexed 22 documents

Writing index to: /home/user/jaato/docs/api/assets/js/search-index.json
Index file size: 77.3 KB
Total documents: 22

✓ Search index generated successfully!
```

## Adding New HTML Files

If you add new HTML files that don't already include the Lunr.js script tag:

```bash
python3 docs/add_lunr_to_html.py
```

This will add the script tag to any HTML files missing it.

## Search Algorithm

The search uses Lunr.js with the following configuration:

- **Fields indexed**:
  - `title` (boost: 10x) - Page titles are most important
  - `headings` (boost: 5x) - Section headings are important
  - `content` (boost: 1x) - Full page text

- **Matching**: Partial word matching with wildcards (`*`)
  - Query "plugin" matches "plugin", "plugins", "PluginRegistry"

- **Results**: Top 10 results sorted by relevance score

## Browser Support

Search works in all modern browsers that support:
- ES5+ JavaScript
- Fetch API
- CSS Flexbox

## Performance

- **Index size**: ~77 KB (22 pages)
- **Load time**: <100ms on modern connections
- **Search time**: <50ms for typical queries
- **Memory**: ~2MB while search index is loaded

## Making It Fully Offline

The current implementation loads Lunr.js from a CDN. To make it completely self-contained:

1. Download Lunr.js:
   ```bash
   curl -o docs/api/assets/js/lunr.min.js https://unpkg.com/lunr@2.3.9/lunr.min.js
   ```

2. Update HTML files to use local copy:
   ```html
   <!-- Change from: -->
   <script src="https://unpkg.com/lunr@2.3.9/lunr.min.js"></script>

   <!-- To: -->
   <script src="assets/js/lunr.min.js"></script>
   ```

## Troubleshooting

**Search box doesn't show results**:
- Check browser console for errors
- Verify `search-index.json` exists and is valid JSON
- Ensure Lunr.js loaded (check Network tab)

**Search returns no results**:
- Index may need regeneration
- Try shorter search terms
- Check that index file isn't empty

**Search is slow**:
- Index may be too large (>1MB)
- Consider reducing content in `generate_search_index.py` (currently limited to 3000 chars per page)

## Customization

### Adjust search behavior

Edit `docs/api/assets/js/docs.js`, function `performSearch()`:

```javascript
// Change number of results
results = results.slice(0, 10);  // Change 10 to desired limit

// Adjust wildcard matching
var searchQuery = query.split(/\s+/).map(function(term) {
  return term + '*';  // Remove '*' for exact matches only
}).join(' ');
```

### Adjust search result appearance

Edit `docs/api/assets/css/style.css`, `.search-results` section:

```css
.search-results {
  max-height: 500px;  /* Adjust dropdown height */
}

.search-result-snippet {
  -webkit-line-clamp: 2;  /* Lines of snippet to show */
}
```

## Resources

- [Lunr.js Documentation](https://lunrjs.com/)
- [Lunr.js GitHub](https://github.com/olivernn/lunr.js)
