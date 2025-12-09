#!/usr/bin/env python3
"""
Script to rename 'actor' to 'channel' throughout the codebase.

This script:
1. Replaces 'actor' with 'channel' in file contents, preserving original case
2. Renames files containing 'actor' in their names

Case preservation examples:
- actor -> channel
- Actor -> Channel
- ACTOR -> CHANNEL
- actorName -> channelName
- ActorConfig -> ChannelConfig
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Set


def case_preserving_replace(text: str, old: str, new: str, whole_word: bool = True) -> str:
    """
    Replace occurrences of 'old' with 'new', preserving the case pattern.

    Args:
        text: The text to process
        old: The word to find (lowercase)
        new: The replacement word (lowercase)
        whole_word: If True, only match whole words (not 'actor' in 'extractor')

    Returns:
        Text with case-preserving replacements
    """
    def replace_match(match: re.Match) -> str:
        original = match.group(0)
        result = []

        # Map each character's case from original to new
        for i, char in enumerate(new):
            if i < len(original):
                if original[i].isupper():
                    result.append(char.upper())
                else:
                    result.append(char.lower())
            else:
                # New word is longer than original - use lowercase for extra chars
                # unless all previous chars were uppercase
                if all(c.isupper() for c in original):
                    result.append(char.upper())
                else:
                    result.append(char.lower())

        return ''.join(result)

    if whole_word:
        # Build explicit case variants to avoid re.IGNORECASE affecting lookbehinds
        # Match patterns:
        # 1. After non-letter: any case of 'actor'
        # 2. CamelCase: lowercase letter before uppercase 'Actor'
        #
        # Generate all case variants: actor, Actor, ACTOR, aCtOr, etc.
        def case_variants(word):
            """Generate all case variants of a word."""
            if not word:
                return ['']
            first = word[0]
            rest_variants = case_variants(word[1:])
            return [first.lower() + r for r in rest_variants] + [first.upper() + r for r in rest_variants]

        variants = case_variants(old)

        # Build pattern for each variant
        # For variants starting with uppercase: also match after lowercase (CamelCase)
        # For all variants: match after non-letter
        patterns = []
        for variant in variants:
            if variant[0].isupper():
                # CamelCase: lowercase letter before uppercase start
                patterns.append(r'(?<=[a-z])' + re.escape(variant))
            # All variants: after non-letter
            patterns.append(r'(?<![a-zA-Z])' + re.escape(variant))

        pattern = re.compile('|'.join(patterns))
    else:
        pattern = re.compile(re.escape(old), re.IGNORECASE)

    return pattern.sub(replace_match, text)


def get_files_to_process(root_dir: Path, exclude_dirs: Set[str]) -> List[Path]:
    """Get all files that should be processed, excluding specified directories."""
    files = []

    for path in root_dir.rglob('*'):
        # Skip directories
        if path.is_dir():
            continue

        # Skip files in excluded directories
        skip = False
        for part in path.parts:
            if part in exclude_dirs:
                skip = True
                break
        if skip:
            continue

        # Skip binary files (by extension)
        binary_extensions = {
            '.pyc', '.pyo', '.so', '.dll', '.exe', '.bin',
            '.png', '.jpg', '.jpeg', '.gif', '.ico', '.bmp',
            '.pdf', '.zip', '.tar', '.gz', '.7z', '.rar',
            '.woff', '.woff2', '.ttf', '.eot', '.otf',
            '.mp3', '.mp4', '.wav', '.avi', '.mov',
            '.db', '.sqlite', '.sqlite3',
        }
        if path.suffix.lower() in binary_extensions:
            continue

        files.append(path)

    return files


def build_whole_word_pattern(word: str) -> re.Pattern:
    """Build regex pattern for whole-word matching that handles CamelCase."""
    def case_variants(w):
        """Generate all case variants of a word."""
        if not w:
            return ['']
        first = w[0]
        rest_variants = case_variants(w[1:])
        return [first.lower() + r for r in rest_variants] + [first.upper() + r for r in rest_variants]

    variants = case_variants(word)
    patterns = []
    for variant in variants:
        if variant[0].isupper():
            # CamelCase: lowercase letter before uppercase start
            patterns.append(r'(?<=[a-z])' + re.escape(variant))
        # All variants: after non-letter
        patterns.append(r'(?<![a-zA-Z])' + re.escape(variant))

    return re.compile('|'.join(patterns))


def count_whole_word_matches(text: str, word: str) -> int:
    """Count whole-word matches of a word (case-insensitive, CamelCase aware)."""
    pattern = build_whole_word_pattern(word)
    return len(pattern.findall(text))


def process_file_contents(file_path: Path, dry_run: bool = True) -> Tuple[bool, int]:
    """
    Process a file's contents, replacing 'actor' with 'channel'.

    Returns:
        Tuple of (was_modified, replacement_count)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, PermissionError):
        return False, 0

    # Count whole-word occurrences before replacement
    count = count_whole_word_matches(content, 'actor')
    if count == 0:
        return False, 0

    # Perform case-preserving replacement
    new_content = case_preserving_replace(content, 'actor', 'channel')

    if new_content != content:
        if not dry_run:
            file_path.write_text(new_content, encoding='utf-8')
        return True, count

    return False, 0


def filename_has_whole_word_actor(name: str) -> bool:
    """Check if filename contains 'actor' as a whole word (not in 'extractor')."""
    pattern = build_whole_word_pattern('actor')
    return bool(pattern.search(name))


def find_files_to_rename(root_dir: Path, exclude_dirs: Set[str], script_name: str = None) -> List[Tuple[Path, Path]]:
    """
    Find files that need to be renamed (contain 'actor' as whole word in filename).

    Returns:
        List of (old_path, new_path) tuples, sorted by path depth (deepest first)
    """
    renames = []

    for path in root_dir.rglob('*'):
        # Skip excluded directories
        skip = False
        for part in path.parts:
            if part in exclude_dirs:
                skip = True
                break
        if skip:
            continue

        # Skip the rename script itself
        if script_name and path.name == script_name:
            continue

        # Check if filename contains 'actor' as whole word (case-insensitive)
        if filename_has_whole_word_actor(path.name):
            new_name = case_preserving_replace(path.name, 'actor', 'channel')
            new_path = path.parent / new_name
            renames.append((path, new_path))

    # Sort by depth (deepest first) to avoid issues with parent directories
    renames.sort(key=lambda x: len(x[0].parts), reverse=True)

    return renames


def main():
    parser = argparse.ArgumentParser(
        description='Rename "actor" to "channel" throughout the codebase, preserving case.'
    )
    parser.add_argument(
        '--root', '-r',
        type=Path,
        default=Path('.'),
        help='Root directory to process (default: current directory)'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--exclude', '-e',
        nargs='*',
        default=['.git', '.venv', 'venv', '__pycache__', 'node_modules', '.mypy_cache', '.pytest_cache'],
        help='Directories to exclude'
    )
    parser.add_argument(
        '--contents-only', '-c',
        action='store_true',
        help='Only process file contents, skip file renames'
    )
    parser.add_argument(
        '--renames-only', '-f',
        action='store_true',
        help='Only rename files, skip content changes'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )

    args = parser.parse_args()
    root_dir = args.root.resolve()
    exclude_dirs = set(args.exclude)

    # Get this script's name to exclude it from processing
    script_name = Path(__file__).name

    print(f"Processing directory: {root_dir}")
    print(f"Excluding: {', '.join(exclude_dirs)}")
    print(f"Skipping script: {script_name}")
    if args.dry_run:
        print("DRY RUN - no changes will be made\n")
    else:
        print("LIVE RUN - changes will be applied\n")

    # Process file contents
    if not args.renames_only:
        print("=" * 60)
        print("FILE CONTENTS")
        print("=" * 60)

        files = get_files_to_process(root_dir, exclude_dirs)
        total_files_modified = 0
        total_replacements = 0

        for file_path in files:
            # Skip this script
            if file_path.name == script_name:
                continue
            modified, count = process_file_contents(file_path, dry_run=args.dry_run)
            if modified:
                total_files_modified += 1
                total_replacements += count
                rel_path = file_path.relative_to(root_dir)
                if args.verbose or args.dry_run:
                    print(f"  {rel_path}: {count} replacement(s)")

        print(f"\nContent changes: {total_files_modified} file(s), {total_replacements} replacement(s)")

    # Rename files
    if not args.contents_only:
        print("\n" + "=" * 60)
        print("FILE RENAMES")
        print("=" * 60)

        renames = find_files_to_rename(root_dir, exclude_dirs, script_name=script_name)

        if renames:
            for old_path, new_path in renames:
                old_rel = old_path.relative_to(root_dir)
                new_rel = new_path.relative_to(root_dir)
                print(f"  {old_rel}")
                print(f"    -> {new_rel}")

                if not args.dry_run:
                    # Create parent directory if needed
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    old_path.rename(new_path)

            print(f"\nFile renames: {len(renames)} file(s)")
        else:
            print("  No files need renaming")

    print("\nDone!")
    if args.dry_run:
        print("\nTo apply changes, run without --dry-run flag")


if __name__ == '__main__':
    main()
