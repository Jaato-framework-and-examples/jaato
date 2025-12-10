#!/usr/bin/env python3
"""
Generate a Lunr.js search index from HTML documentation files.

This script scans all HTML files in docs/api/, extracts their content
(title, headings, text), and generates a JSON index file that can be
loaded by Lunr.js for client-side full-text search.

Usage:
    python docs/generate_search_index.py

Output:
    docs/api/assets/js/search-index.json
"""

import json
import os
from pathlib import Path
from html.parser import HTMLParser
import re


class HTMLTextExtractor(HTMLParser):
    """Extract text content from HTML, excluding script/style tags."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {'script', 'style', 'head'}
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag

    def handle_endtag(self, tag):
        if tag == self.current_tag:
            self.current_tag = None

    def handle_data(self, data):
        if self.current_tag not in self.skip_tags:
            text = data.strip()
            if text:
                self.text_parts.append(text)

    def get_text(self):
        return ' '.join(self.text_parts)


def clean_text(text):
    """Clean and normalize text content."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might break JSON
    text = text.strip()
    return text


def extract_headings(html_content):
    """Extract all h1, h2, h3 headings from HTML."""
    headings = []

    # Simple regex to find headings
    # Matches <h1>...</h1>, <h2>...</h2>, <h3>...</h3>
    heading_pattern = r'<h[123][^>]*>(.*?)</h[123]>'

    for match in re.finditer(heading_pattern, html_content, re.DOTALL | re.IGNORECASE):
        heading_html = match.group(1)
        # Remove HTML tags from heading text
        heading_text = re.sub(r'<[^>]+>', '', heading_html)
        heading_text = clean_text(heading_text)
        # Remove anchor link symbols
        heading_text = heading_text.replace('#', '').strip()
        if heading_text:
            headings.append(heading_text)

    return headings


def extract_title(html_content):
    """Extract page title from HTML."""
    title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = title_match.group(1)
        title = clean_text(title)
        # Remove " - jaato" suffix if present
        title = re.sub(r'\s*-\s*jaato\s*$', '', title)
        return title
    return "Untitled"


def extract_text_content(html_content):
    """Extract main text content from HTML."""
    # Remove script and style tags completely
    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)

    # Extract text using parser
    parser = HTMLTextExtractor()
    try:
        parser.feed(html_content)
        text = parser.get_text()
    except Exception as e:
        print(f"Warning: Error parsing HTML: {e}")
        # Fallback: strip all tags
        text = re.sub(r'<[^>]+>', ' ', html_content)
        text = clean_text(text)

    return text


def get_section_from_path(rel_path):
    """Determine section name from file path."""
    parts = rel_path.split('/')
    if len(parts) > 1:
        section = parts[0].replace('-', ' ').title()
        return section
    return "Home"


def process_html_file(file_path, docs_root):
    """Process a single HTML file and extract searchable content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Get relative path from docs/api/
    rel_path = os.path.relpath(file_path, docs_root)

    # Extract content
    title = extract_title(html_content)
    headings = extract_headings(html_content)
    text_content = extract_text_content(html_content)
    section = get_section_from_path(rel_path)

    # Create searchable content combining all text
    # Headings are more important, so we include them multiple times
    searchable_content = f"{title} {' '.join(headings)} {text_content}"

    # Limit content length to avoid huge index
    if len(searchable_content) > 3000:
        searchable_content = searchable_content[:3000] + '...'

    # Create snippet for search results (first paragraph or ~200 chars)
    snippet = text_content[:200].strip()
    if len(text_content) > 200:
        # Try to cut at sentence boundary
        last_period = snippet.rfind('.')
        if last_period > 100:
            snippet = snippet[:last_period + 1]
        else:
            snippet += '...'

    return {
        'id': rel_path,
        'title': title,
        'url': rel_path,
        'section': section,
        'headings': headings[:5],  # Top 5 headings
        'content': searchable_content,
        'snippet': snippet
    }


def generate_search_index(docs_root):
    """Generate search index from all HTML files in docs directory."""
    docs_root = Path(docs_root)

    if not docs_root.exists():
        print(f"Error: Directory {docs_root} does not exist")
        return None

    print(f"Scanning {docs_root} for HTML files...")

    # Find all HTML files
    html_files = list(docs_root.rglob('*.html'))

    if not html_files:
        print(f"Warning: No HTML files found in {docs_root}")
        return []

    print(f"Found {len(html_files)} HTML files")

    # Process each file
    index_data = []
    for file_path in html_files:
        try:
            print(f"  Processing: {file_path.relative_to(docs_root)}")
            doc_data = process_html_file(file_path, docs_root)
            index_data.append(doc_data)
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")

    print(f"Successfully indexed {len(index_data)} documents")
    return index_data


def main():
    """Main entry point."""
    # Determine paths
    script_dir = Path(__file__).parent
    docs_root = script_dir / 'api'
    output_file = docs_root / 'assets' / 'js' / 'search-index.json'

    print("=" * 60)
    print("Lunr.js Search Index Generator")
    print("=" * 60)

    # Generate index
    index_data = generate_search_index(docs_root)

    if not index_data:
        print("Error: No documents to index")
        return 1

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write index file
    print(f"\nWriting index to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    # Print stats
    total_size = output_file.stat().st_size
    print(f"Index file size: {total_size / 1024:.1f} KB")
    print(f"Total documents: {len(index_data)}")
    print("\n✓ Search index generated successfully!")

    return 0


if __name__ == '__main__':
    exit(main())
