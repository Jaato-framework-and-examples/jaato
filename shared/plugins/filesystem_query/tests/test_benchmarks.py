"""Performance benchmarks comparing filesystem_query tools vs CLI equivalents.

These tests measure and compare execution times between:
- glob_files vs `find` command
- grep_content vs `grep` command

Run with: pytest test_benchmarks.py -v -s
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pytest

from ..plugin import FilesystemQueryPlugin


# Benchmark configuration
NUM_FILES = 500  # Number of files to create for tests
NUM_DIRS = 50    # Number of directories
LINES_PER_FILE = 100  # Lines per file for grep tests
NUM_ITERATIONS = 5  # Number of iterations for timing


def create_test_codebase(root: Path, num_files: int, num_dirs: int, lines_per_file: int) -> Dict[str, int]:
    """Create a realistic test codebase structure.

    Returns:
        Dict with counts of created files by extension.
    """
    extensions = ['.py', '.js', '.ts', '.json', '.md', '.txt', '.yaml']
    counts = {ext: 0 for ext in extensions}

    # Create directory structure
    dirs = [root]
    for i in range(num_dirs):
        depth = i % 4  # Max depth of 4
        parent = dirs[i % len(dirs)]
        new_dir = parent / f"dir_{i}"
        new_dir.mkdir(exist_ok=True)
        dirs.append(new_dir)

    # Create some "excluded" directories that would normally be skipped
    (root / "node_modules").mkdir(exist_ok=True)
    (root / "node_modules" / "lodash").mkdir(exist_ok=True)
    (root / "__pycache__").mkdir(exist_ok=True)
    (root / ".git").mkdir(exist_ok=True)
    (root / ".git" / "objects").mkdir(exist_ok=True)

    # Create files in excluded dirs (to test exclusion efficiency)
    for i in range(50):
        (root / "node_modules" / "lodash" / f"file_{i}.js").write_text(f"// lodash file {i}\n" * lines_per_file)
        (root / "__pycache__" / f"module_{i}.pyc").write_bytes(b"\x00" * 100)

    # Create regular files
    for i in range(num_files):
        ext = extensions[i % len(extensions)]
        target_dir = dirs[i % len(dirs)]
        filename = f"file_{i}{ext}"
        filepath = target_dir / filename

        # Create content with searchable patterns
        if ext == '.py':
            content = f'''"""Module {i} documentation."""

import os
import sys
from typing import List

def function_{i}(arg1, arg2):
    """Function {i} docstring."""
    # TODO: implement this
    result = arg1 + arg2
    return result

class Class_{i}:
    """Class {i} docstring."""

    def method_{i}(self):
        # FIXME: needs refactoring
        pass

'''
            content += f"# Line padding\n" * (lines_per_file - 20)
        elif ext == '.js':
            content = f'''// Module {i}
const module{i} = require('./module');

function process{i}(data) {{
    // TODO: add validation
    return data.map(x => x * 2);
}}

class Handler{i} {{
    constructor() {{
        this.id = {i};
    }}
}}

module.exports = {{ process{i}, Handler{i} }};
'''
            content += f"// Line {i}\n" * (lines_per_file - 15)
        else:
            content = f"Line content for file {i}\n" * lines_per_file

        filepath.write_text(content)
        counts[ext] += 1

    return counts


def time_function(func: Callable, *args, iterations: int = NUM_ITERATIONS, **kwargs) -> Tuple[float, float, float]:
    """Time a function over multiple iterations.

    Returns:
        Tuple of (min_time, avg_time, max_time) in seconds.
    """
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return min(times), sum(times) / len(times), max(times)


def time_cli_command(cmd: List[str], cwd: str, iterations: int = NUM_ITERATIONS) -> Tuple[float, float, float]:
    """Time a CLI command over multiple iterations.

    Returns:
        Tuple of (min_time, avg_time, max_time) in seconds.
    """
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        subprocess.run(cmd, cwd=cwd, capture_output=True)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return min(times), sum(times) / len(times), max(times)


class TestGlobVsFind:
    """Benchmark glob_files against `find` command."""

    @pytest.fixture(scope="class")
    def test_codebase(self, tmp_path_factory):
        """Create a test codebase for benchmarks."""
        root = tmp_path_factory.mktemp("benchmark_codebase")
        counts = create_test_codebase(root, NUM_FILES, NUM_DIRS, LINES_PER_FILE)
        return root, counts

    @pytest.fixture
    def plugin(self):
        """Create and initialize the plugin."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize()
        yield plugin
        plugin.shutdown()

    def test_glob_all_python_files(self, test_codebase, plugin):
        """Benchmark: Find all Python files."""
        root, counts = test_codebase

        # glob_files
        glob_min, glob_avg, glob_max = time_function(
            plugin._execute_glob_files,
            {"pattern": "**/*.py", "root": str(root), "max_results": 10000}
        )

        # find command
        find_min, find_avg, find_max = time_cli_command(
            ["find", ".", "-name", "*.py", "-type", "f"],
            cwd=str(root)
        )

        print(f"\n{'='*60}")
        print(f"Benchmark: Find all Python files ({counts['.py']} files)")
        print(f"{'='*60}")
        print(f"glob_files:  min={glob_min:.4f}s  avg={glob_avg:.4f}s  max={glob_max:.4f}s")
        print(f"find:        min={find_min:.4f}s  avg={find_avg:.4f}s  max={find_max:.4f}s")
        print(f"Ratio (glob/find): {glob_avg/find_avg:.2f}x")

        # Verify correctness
        result = plugin._execute_glob_files({"pattern": "**/*.py", "root": str(root), "max_results": 10000})
        assert result["total"] == counts[".py"], f"Expected {counts['.py']} files, got {result['total']}"

    def test_glob_with_exclusions_benefit(self, test_codebase, plugin):
        """Benchmark: Measure benefit of automatic exclusions."""
        root, counts = test_codebase

        # glob_files (with automatic exclusions - skips node_modules, __pycache__)
        glob_min, glob_avg, glob_max = time_function(
            plugin._execute_glob_files,
            {"pattern": "**/*.js", "root": str(root), "max_results": 10000}
        )

        # find command (searches everything including node_modules)
        find_min, find_avg, find_max = time_cli_command(
            ["find", ".", "-name", "*.js", "-type", "f"],
            cwd=str(root)
        )

        # find with exclusions (fair comparison)
        find_excl_min, find_excl_avg, find_excl_max = time_cli_command(
            ["find", ".", "-name", "*.js", "-type", "f",
             "-not", "-path", "*/node_modules/*",
             "-not", "-path", "*/__pycache__/*",
             "-not", "-path", "*/.git/*"],
            cwd=str(root)
        )

        print(f"\n{'='*60}")
        print(f"Benchmark: Find JS files (with exclusion comparison)")
        print(f"{'='*60}")
        print(f"glob_files (auto-exclude): min={glob_min:.4f}s  avg={glob_avg:.4f}s")
        print(f"find (no exclusions):      min={find_min:.4f}s  avg={find_avg:.4f}s")
        print(f"find (with exclusions):    min={find_excl_min:.4f}s  avg={find_excl_avg:.4f}s")
        print(f"Ratio (glob/find_excl): {glob_avg/find_excl_avg:.2f}x")

        # Verify exclusions work
        result = plugin._execute_glob_files({"pattern": "**/*.js", "root": str(root), "max_results": 10000})
        # Should NOT include files from node_modules
        for f in result["files"]:
            assert "node_modules" not in f["path"], f"Should exclude node_modules: {f['path']}"

    def test_glob_specific_pattern(self, test_codebase, plugin):
        """Benchmark: Find files with specific pattern (test_*.py)."""
        root, counts = test_codebase

        # Create some test files
        (root / "test_main.py").write_text("# test\n")
        (root / "test_utils.py").write_text("# test\n")
        (root / "dir_0" / "test_module.py").write_text("# test\n")

        glob_min, glob_avg, glob_max = time_function(
            plugin._execute_glob_files,
            {"pattern": "**/test_*.py", "root": str(root), "max_results": 10000}
        )

        find_min, find_avg, find_max = time_cli_command(
            ["find", ".", "-name", "test_*.py", "-type", "f"],
            cwd=str(root)
        )

        print(f"\n{'='*60}")
        print(f"Benchmark: Find test files (test_*.py pattern)")
        print(f"{'='*60}")
        print(f"glob_files:  min={glob_min:.4f}s  avg={glob_avg:.4f}s  max={glob_max:.4f}s")
        print(f"find:        min={find_min:.4f}s  avg={find_avg:.4f}s  max={find_max:.4f}s")
        print(f"Ratio (glob/find): {glob_avg/find_avg:.2f}x")


class TestGrepVsGrep:
    """Benchmark grep_content against `grep` command."""

    @pytest.fixture(scope="class")
    def test_codebase(self, tmp_path_factory):
        """Create a test codebase for benchmarks."""
        root = tmp_path_factory.mktemp("benchmark_codebase_grep")
        counts = create_test_codebase(root, NUM_FILES, NUM_DIRS, LINES_PER_FILE)
        return root, counts

    @pytest.fixture
    def plugin(self):
        """Create and initialize the plugin."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize()
        yield plugin
        plugin.shutdown()

    def test_grep_simple_pattern(self, test_codebase, plugin):
        """Benchmark: Search for simple string pattern."""
        root, counts = test_codebase

        # grep_content
        grep_content_min, grep_content_avg, grep_content_max = time_function(
            plugin._execute_grep_content,
            {"pattern": "TODO", "path": str(root), "max_results": 10000, "context_lines": 0}
        )

        # grep command
        grep_min, grep_avg, grep_max = time_cli_command(
            ["grep", "-r", "TODO", "."],
            cwd=str(root)
        )

        # grep with exclusions (fair comparison)
        grep_excl_min, grep_excl_avg, grep_excl_max = time_cli_command(
            ["grep", "-r", "TODO", ".",
             "--exclude-dir=node_modules",
             "--exclude-dir=__pycache__",
             "--exclude-dir=.git"],
            cwd=str(root)
        )

        print(f"\n{'='*60}")
        print(f"Benchmark: Search for 'TODO' pattern")
        print(f"{'='*60}")
        print(f"grep_content (auto-exclude): min={grep_content_min:.4f}s  avg={grep_content_avg:.4f}s")
        print(f"grep (no exclusions):        min={grep_min:.4f}s  avg={grep_avg:.4f}s")
        print(f"grep (with exclusions):      min={grep_excl_min:.4f}s  avg={grep_excl_avg:.4f}s")
        print(f"Ratio (grep_content/grep_excl): {grep_content_avg/grep_excl_avg:.2f}x")

        # Verify we found matches
        result = plugin._execute_grep_content({"pattern": "TODO", "path": str(root), "max_results": 10000})
        assert result["total_matches"] > 0, "Should find TODO matches"

    def test_grep_regex_pattern(self, test_codebase, plugin):
        """Benchmark: Search with regex pattern."""
        root, counts = test_codebase

        pattern = r"def\s+\w+\("

        grep_content_min, grep_content_avg, grep_content_max = time_function(
            plugin._execute_grep_content,
            {"pattern": pattern, "path": str(root), "file_glob": "*.py", "max_results": 10000, "context_lines": 0}
        )

        grep_min, grep_avg, grep_max = time_cli_command(
            ["grep", "-rE", pattern, ".", "--include=*.py"],
            cwd=str(root)
        )

        print(f"\n{'='*60}")
        print(f"Benchmark: Search for function definitions (regex)")
        print(f"{'='*60}")
        print(f"grep_content:  min={grep_content_min:.4f}s  avg={grep_content_avg:.4f}s")
        print(f"grep -E:       min={grep_min:.4f}s  avg={grep_avg:.4f}s")
        print(f"Ratio (grep_content/grep): {grep_content_avg/grep_avg:.2f}x")

        result = plugin._execute_grep_content({
            "pattern": pattern, "path": str(root), "file_glob": "*.py", "max_results": 10000
        })
        assert result["total_matches"] > 0, "Should find function definitions"

    def test_grep_with_context(self, test_codebase, plugin):
        """Benchmark: Search with context lines."""
        root, counts = test_codebase

        grep_content_min, grep_content_avg, grep_content_max = time_function(
            plugin._execute_grep_content,
            {"pattern": "class", "path": str(root), "file_glob": "*.py", "context_lines": 3, "max_results": 1000}
        )

        grep_min, grep_avg, grep_max = time_cli_command(
            ["grep", "-rC", "3", "class", ".", "--include=*.py"],
            cwd=str(root)
        )

        print(f"\n{'='*60}")
        print(f"Benchmark: Search with 3 lines of context")
        print(f"{'='*60}")
        print(f"grep_content:  min={grep_content_min:.4f}s  avg={grep_content_avg:.4f}s")
        print(f"grep -C 3:     min={grep_min:.4f}s  avg={grep_avg:.4f}s")
        print(f"Ratio (grep_content/grep): {grep_content_avg/grep_avg:.2f}x")

    def test_grep_case_insensitive(self, test_codebase, plugin):
        """Benchmark: Case-insensitive search."""
        root, counts = test_codebase

        grep_content_min, grep_content_avg, grep_content_max = time_function(
            plugin._execute_grep_content,
            {"pattern": "function", "path": str(root), "case_sensitive": False, "max_results": 10000, "context_lines": 0}
        )

        grep_min, grep_avg, grep_max = time_cli_command(
            ["grep", "-ri", "function", "."],
            cwd=str(root)
        )

        print(f"\n{'='*60}")
        print(f"Benchmark: Case-insensitive search")
        print(f"{'='*60}")
        print(f"grep_content:  min={grep_content_min:.4f}s  avg={grep_content_avg:.4f}s")
        print(f"grep -i:       min={grep_min:.4f}s  avg={grep_avg:.4f}s")
        print(f"Ratio (grep_content/grep): {grep_content_avg/grep_avg:.2f}x")


class TestScalability:
    """Test how performance scales with codebase size."""

    @pytest.fixture
    def plugin(self):
        """Create and initialize the plugin."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize()
        yield plugin
        plugin.shutdown()

    @pytest.mark.parametrize("num_files", [100, 250, 500, 1000])
    def test_glob_scalability(self, plugin, num_files, tmp_path):
        """Measure glob_files performance as file count increases."""
        create_test_codebase(tmp_path, num_files, num_files // 10, 50)

        min_time, avg_time, max_time = time_function(
            plugin._execute_glob_files,
            {"pattern": "**/*.py", "root": str(tmp_path), "max_results": 10000},
            iterations=3
        )

        result = plugin._execute_glob_files({"pattern": "**/*.py", "root": str(tmp_path), "max_results": 10000})

        print(f"\nglob_files with {num_files} total files ({result['total']} .py): avg={avg_time:.4f}s")

    @pytest.mark.parametrize("num_files", [100, 250, 500])
    def test_grep_scalability(self, plugin, num_files, tmp_path):
        """Measure grep_content performance as file count increases."""
        create_test_codebase(tmp_path, num_files, num_files // 10, 100)

        min_time, avg_time, max_time = time_function(
            plugin._execute_grep_content,
            {"pattern": "def", "path": str(tmp_path), "file_glob": "*.py", "max_results": 10000, "context_lines": 0},
            iterations=3
        )

        result = plugin._execute_grep_content({
            "pattern": "def", "path": str(tmp_path), "file_glob": "*.py", "max_results": 10000
        })

        print(f"\ngrep_content with {num_files} files ({result['files_searched']} searched, {result['total_matches']} matches): avg={avg_time:.4f}s")


class TestExclusionEfficiency:
    """Measure the efficiency gains from automatic exclusions."""

    @pytest.fixture
    def plugin(self):
        plugin = FilesystemQueryPlugin()
        plugin.initialize()
        yield plugin
        plugin.shutdown()

    @pytest.fixture
    def plugin_no_exclusions(self):
        """Plugin configured with no exclusions."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize(config={"exclude_mode": "replace", "exclude_patterns": []})
        yield plugin
        plugin.shutdown()

    def test_exclusion_time_savings(self, plugin, plugin_no_exclusions, tmp_path):
        """Measure time saved by excluding node_modules, etc."""
        # Create structure with heavy excluded directories
        create_test_codebase(tmp_path, 200, 20, 50)

        # Add extra files in excluded dirs
        node_modules = tmp_path / "node_modules"
        for i in range(200):
            pkg_dir = node_modules / f"package_{i}"
            pkg_dir.mkdir(exist_ok=True)
            (pkg_dir / "index.js").write_text(f"module.exports = {i};\n" * 100)

        # With exclusions
        with_excl_min, with_excl_avg, with_excl_max = time_function(
            plugin._execute_glob_files,
            {"pattern": "**/*.js", "root": str(tmp_path), "max_results": 10000},
            iterations=3
        )

        # Without exclusions
        no_excl_min, no_excl_avg, no_excl_max = time_function(
            plugin_no_exclusions._execute_glob_files,
            {"pattern": "**/*.js", "root": str(tmp_path), "max_results": 10000},
            iterations=3
        )

        result_with = plugin._execute_glob_files({"pattern": "**/*.js", "root": str(tmp_path), "max_results": 10000})
        result_without = plugin_no_exclusions._execute_glob_files({"pattern": "**/*.js", "root": str(tmp_path), "max_results": 10000})

        print(f"\n{'='*60}")
        print(f"Benchmark: Exclusion efficiency (node_modules heavy)")
        print(f"{'='*60}")
        print(f"With exclusions:    {with_excl_avg:.4f}s ({result_with['total']} files)")
        print(f"Without exclusions: {no_excl_avg:.4f}s ({result_without['total']} files)")
        print(f"Time saved: {(1 - with_excl_avg/no_excl_avg) * 100:.1f}%")
        print(f"Files excluded: {result_without['total'] - result_with['total']}")


def print_benchmark_summary():
    """Print a summary header for benchmark results."""
    print("\n" + "=" * 70)
    print("FILESYSTEM_QUERY PLUGIN BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Configuration: {NUM_FILES} files, {NUM_DIRS} dirs, {LINES_PER_FILE} lines/file")
    print(f"Iterations per test: {NUM_ITERATIONS}")
    print("=" * 70)


if __name__ == "__main__":
    print_benchmark_summary()
    pytest.main([__file__, "-v", "-s"])
