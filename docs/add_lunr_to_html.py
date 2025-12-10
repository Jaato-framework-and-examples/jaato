#!/usr/bin/env python3
"""
Add Lunr.js script tag to all HTML files in docs/api/

This script adds the Lunr.js CDN script tag before the docs.js script tag
in all HTML documentation files.
"""

import os
import re
from pathlib import Path


def add_lunr_script(file_path):
    """Add Lunr.js script tag to an HTML file if not already present."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if Lunr.js is already included
    if 'lunr' in content.lower():
        print(f"  Skipped (already has Lunr): {file_path}")
        return False

    # Pattern to match the docs.js script tag
    # We'll insert the Lunr.js script right before it
    pattern = r'(\s*)(<script src="[^"]*assets/js/docs\.js"></script>)'

    match = re.search(pattern, content)
    if not match:
        print(f"  Warning: Could not find docs.js script tag in {file_path}")
        return False

    # Extract the indentation and the script tag
    indent = match.group(1)
    docs_script = match.group(2)

    # Create the Lunr.js script tag with the same indentation
    lunr_script = f'{indent}<script src="https://unpkg.com/lunr@2.3.9/lunr.min.js"></script>\n'

    # Replace the docs.js script tag with both scripts
    new_content = content.replace(
        match.group(0),
        lunr_script + indent + docs_script
    )

    # Write the updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return True


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    docs_root = script_dir / 'api'

    if not docs_root.exists():
        print(f"Error: Directory {docs_root} does not exist")
        return 1

    print("Adding Lunr.js script tag to HTML files...")
    print("=" * 60)

    # Find all HTML files
    html_files = list(docs_root.rglob('*.html'))

    updated_count = 0
    for file_path in sorted(html_files):
        rel_path = file_path.relative_to(docs_root)
        print(f"Processing: {rel_path}")

        if add_lunr_script(file_path):
            updated_count += 1

    print("=" * 60)
    print(f"Updated {updated_count} files")
    print("✓ Done!")

    return 0


if __name__ == '__main__':
    exit(main())
