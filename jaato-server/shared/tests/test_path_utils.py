"""Tests for shared.path_utils MSYS2/cross-platform path utilities."""

import os
import sys
from unittest import mock

import pytest

from shared.path_utils import (
    is_msys2_environment,
    msys2_to_windows_path,
    windows_to_msys2_path,
    normalize_path,
    normalize_for_comparison,
    normalized_startswith,
    normalized_equals,
    normalize_result_path,
    get_display_separator,
)


# ============================================================================
# is_msys2_environment() tests
# ============================================================================


class TestIsMsys2Environment:
    """Tests for MSYS2 environment detection."""

    def setup_method(self):
        """Clear the lru_cache before each test."""
        is_msys2_environment.cache_clear()

    def teardown_method(self):
        """Clear the lru_cache after each test."""
        is_msys2_environment.cache_clear()

    @mock.patch.dict(os.environ, {"MSYSTEM": "MINGW64"}, clear=False)
    @mock.patch("shared.path_utils.sys")
    def test_mingw64_detected(self, mock_sys):
        mock_sys.platform = "win32"
        assert is_msys2_environment() is True

    @mock.patch.dict(os.environ, {"MSYSTEM": "MINGW32"}, clear=False)
    @mock.patch("shared.path_utils.sys")
    def test_mingw32_detected(self, mock_sys):
        mock_sys.platform = "win32"
        assert is_msys2_environment() is True

    @mock.patch.dict(os.environ, {"MSYSTEM": "MSYS"}, clear=False)
    @mock.patch("shared.path_utils.sys")
    def test_msys_detected(self, mock_sys):
        mock_sys.platform = "win32"
        assert is_msys2_environment() is True

    @mock.patch.dict(os.environ, {"MSYSTEM": "UCRT64"}, clear=False)
    @mock.patch("shared.path_utils.sys")
    def test_ucrt64_detected(self, mock_sys):
        mock_sys.platform = "win32"
        assert is_msys2_environment() is True

    @mock.patch.dict(os.environ, {"MSYSTEM": "CLANG64"}, clear=False)
    @mock.patch("shared.path_utils.sys")
    def test_clang64_detected(self, mock_sys):
        mock_sys.platform = "win32"
        assert is_msys2_environment() is True

    @mock.patch.dict(os.environ, {"MSYSTEM": "CLANGARM64"}, clear=False)
    @mock.patch("shared.path_utils.sys")
    def test_clangarm64_detected(self, mock_sys):
        mock_sys.platform = "win32"
        assert is_msys2_environment() is True

    @mock.patch.dict(os.environ, {"TERM_PROGRAM": "mintty"}, clear=False)
    @mock.patch("shared.path_utils.sys")
    def test_mintty_detected(self, mock_sys):
        mock_sys.platform = "win32"
        # Remove MSYSTEM if present
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSYSTEM", None)
            is_msys2_environment.cache_clear()
            assert is_msys2_environment() is True

    @mock.patch("shared.path_utils.sys")
    def test_not_detected_on_linux(self, mock_sys):
        mock_sys.platform = "linux"
        assert is_msys2_environment() is False

    @mock.patch("shared.path_utils.sys")
    def test_not_detected_on_darwin(self, mock_sys):
        mock_sys.platform = "darwin"
        assert is_msys2_environment() is False

    @mock.patch("shared.path_utils.sys")
    def test_mingw_prefix_detected(self, mock_sys):
        """MINGW_PREFIX env var triggers detection (daemon fallback)."""
        mock_sys.platform = "win32"
        mock_sys.executable = "C:\\Python312\\python.exe"
        env = {"MINGW_PREFIX": "/mingw64"}
        with mock.patch.dict(os.environ, env, clear=True):
            is_msys2_environment.cache_clear()
            assert is_msys2_environment() is True

    @mock.patch("shared.path_utils.sys")
    def test_mingw_chost_detected(self, mock_sys):
        """MINGW_CHOST env var triggers detection (daemon fallback)."""
        mock_sys.platform = "win32"
        mock_sys.executable = "C:\\Python312\\python.exe"
        env = {"MINGW_CHOST": "x86_64-w64-mingw32"}
        with mock.patch.dict(os.environ, env, clear=True):
            is_msys2_environment.cache_clear()
            assert is_msys2_environment() is True

    @mock.patch("shared.path_utils.sys")
    def test_executable_in_msys64_detected(self, mock_sys):
        """Python executable inside msys64 directory triggers detection."""
        mock_sys.platform = "win32"
        mock_sys.executable = "C:\\msys64\\mingw64\\bin\\python.exe"
        with mock.patch.dict(os.environ, {}, clear=True):
            is_msys2_environment.cache_clear()
            assert is_msys2_environment() is True

    @mock.patch("shared.path_utils.sys")
    def test_executable_in_mingw64_detected(self, mock_sys):
        """Python executable inside mingw64 directory triggers detection."""
        mock_sys.platform = "win32"
        mock_sys.executable = "C:\\mingw64\\bin\\python.exe"
        with mock.patch.dict(os.environ, {}, clear=True):
            is_msys2_environment.cache_clear()
            assert is_msys2_environment() is True

    @mock.patch("shared.path_utils.sys")
    def test_executable_in_ucrt64_detected(self, mock_sys):
        """Python executable inside ucrt64 directory triggers detection."""
        mock_sys.platform = "win32"
        mock_sys.executable = "D:\\ucrt64\\bin\\python3.exe"
        with mock.patch.dict(os.environ, {}, clear=True):
            is_msys2_environment.cache_clear()
            assert is_msys2_environment() is True

    @mock.patch.dict(os.environ, {}, clear=False)
    @mock.patch("shared.path_utils.sys")
    def test_not_detected_plain_windows(self, mock_sys):
        mock_sys.platform = "win32"
        mock_sys.executable = "C:\\Python312\\python.exe"
        # Ensure no MSYS2 indicators
        env_copy = dict(os.environ)
        env_copy.pop("MSYSTEM", None)
        env_copy.pop("TERM_PROGRAM", None)
        env_copy.pop("MINGW_PREFIX", None)
        env_copy.pop("MINGW_CHOST", None)
        with mock.patch.dict(os.environ, env_copy, clear=True):
            is_msys2_environment.cache_clear()
            assert is_msys2_environment() is False


# ============================================================================
# msys2_to_windows_path() tests
# ============================================================================


class TestMsys2ToWindowsPath:
    """Tests for MSYS2 -> Windows drive letter conversion."""

    def test_lowercase_drive(self):
        assert msys2_to_windows_path("/c/Users/foo") == "C:/Users/foo"

    def test_uppercase_drive(self):
        assert msys2_to_windows_path("/C/Users/foo") == "C:/Users/foo"

    def test_other_drives(self):
        assert msys2_to_windows_path("/d/data/file.txt") == "D:/data/file.txt"
        assert msys2_to_windows_path("/e/backup") == "E:/backup"

    def test_drive_root(self):
        assert msys2_to_windows_path("/c/") == "C:/"

    def test_drive_only(self):
        # /c with nothing after it
        assert msys2_to_windows_path("/c") == "C:/"

    def test_not_a_drive_path(self):
        # /config/ or /cache/ should NOT be converted (multi-letter names)
        assert msys2_to_windows_path("/config/foo") == "/config/foo"
        assert msys2_to_windows_path("/cache/bar") == "/cache/bar"
        assert msys2_to_windows_path("/home/user") == "/home/user"
        assert msys2_to_windows_path("/tmp/test") == "/tmp/test"

    def test_already_windows_path(self):
        assert msys2_to_windows_path("C:/Users/foo") == "C:/Users/foo"
        assert msys2_to_windows_path("C:\\Users\\foo") == "C:\\Users\\foo"

    def test_relative_path(self):
        assert msys2_to_windows_path("src/main/file.py") == "src/main/file.py"

    def test_empty_path(self):
        assert msys2_to_windows_path("") == ""

    def test_preserves_trailing_separator(self):
        assert msys2_to_windows_path("/c/Users/foo/") == "C:/Users/foo/"


# ============================================================================
# windows_to_msys2_path() tests
# ============================================================================


class TestWindowsToMsys2Path:
    """Tests for Windows -> MSYS2 drive letter conversion."""

    def test_uppercase_drive(self):
        assert windows_to_msys2_path("C:/Users/foo") == "/c/Users/foo"

    def test_lowercase_drive(self):
        assert windows_to_msys2_path("c:/Users/foo") == "/c/Users/foo"

    def test_backslash_drive(self):
        assert windows_to_msys2_path("C:\\Users\\foo") == "/c/Users/foo"

    def test_mixed_separators(self):
        assert windows_to_msys2_path("D:\\data/file.txt") == "/d/data/file.txt"

    def test_drive_root(self):
        assert windows_to_msys2_path("C:/") == "/c/"
        assert windows_to_msys2_path("C:\\") == "/c/"

    def test_not_a_drive_path(self):
        assert windows_to_msys2_path("/home/user") == "/home/user"
        assert windows_to_msys2_path("relative/path") == "relative/path"

    def test_already_msys2_path(self):
        assert windows_to_msys2_path("/c/Users/foo") == "/c/Users/foo"

    def test_empty_path(self):
        assert windows_to_msys2_path("") == ""

    def test_roundtrip(self):
        """msys2_to_windows then windows_to_msys2 returns original."""
        original = "/c/Users/foo/project/file.py"
        assert windows_to_msys2_path(msys2_to_windows_path(original)) == original

    def test_roundtrip_from_windows(self):
        """windows_to_msys2 then msys2_to_windows returns original format."""
        original = "C:/Users/foo/project/file.py"
        result = msys2_to_windows_path(windows_to_msys2_path(original))
        assert result == original


# ============================================================================
# normalize_path() tests
# ============================================================================


class TestNormalizePath:
    """Tests for MSYS2-conditional path normalization (including drive letter)."""

    def setup_method(self):
        is_msys2_environment.cache_clear()

    def teardown_method(self):
        is_msys2_environment.cache_clear()

    @mock.patch("shared.path_utils.is_msys2_environment", return_value=True)
    def test_windows_to_msys2_under_msys2(self, _mock):
        """Under MSYS2, converts C:\\foo -> /c/foo."""
        assert normalize_path(r"C:\Users\foo\project\file.py") == "/c/Users/foo/project/file.py"

    @mock.patch("shared.path_utils.is_msys2_environment", return_value=True)
    def test_already_forward_unchanged_under_msys2(self, _mock):
        assert normalize_path("/home/user/project/file.py") == "/home/user/project/file.py"

    @mock.patch("shared.path_utils.is_msys2_environment", return_value=True)
    def test_mixed_separators_under_msys2(self, _mock):
        """Under MSYS2, mixed separators with drive letter -> /c/... format."""
        assert normalize_path(r"C:\Users/foo\project/file.py") == "/c/Users/foo/project/file.py"

    @mock.patch("shared.path_utils.is_msys2_environment", return_value=False)
    def test_unchanged_when_not_msys2(self, _mock):
        original = r"C:\Users\foo\project\file.py"
        assert normalize_path(original) == original

    @mock.patch("shared.path_utils.is_msys2_environment", return_value=True)
    def test_empty_path(self, _mock):
        assert normalize_path("") == ""

    @mock.patch("shared.path_utils.is_msys2_environment", return_value=True)
    def test_relative_path_under_msys2(self, _mock):
        assert normalize_path(r"src\main\file.py") == "src/main/file.py"


# ============================================================================
# normalize_for_comparison() tests
# ============================================================================


class TestNormalizeForComparison:
    """Tests for comparison-safe path normalization."""

    @mock.patch("shared.path_utils.sys")
    def test_backslash_normalized_on_windows(self, mock_sys):
        mock_sys.platform = "win32"
        assert normalize_for_comparison(r"C:\Users\foo\file.py") == "C:/Users/foo/file.py"

    @mock.patch("shared.path_utils.sys")
    def test_forward_slash_unchanged_on_windows(self, mock_sys):
        mock_sys.platform = "win32"
        assert normalize_for_comparison("C:/Users/foo/file.py") == "C:/Users/foo/file.py"

    @mock.patch("shared.path_utils.sys")
    def test_unchanged_on_linux(self, mock_sys):
        mock_sys.platform = "linux"
        assert normalize_for_comparison("/home/user/file.py") == "/home/user/file.py"

    @mock.patch("shared.path_utils.sys")
    def test_empty_path(self, mock_sys):
        mock_sys.platform = "win32"
        assert normalize_for_comparison("") == ""


# ============================================================================
# normalized_startswith() tests
# ============================================================================


class TestNormalizedStartswith:
    """Tests for separator-safe prefix matching."""

    @mock.patch("shared.path_utils.sys")
    def test_mixed_separators_match(self, mock_sys):
        mock_sys.platform = "win32"
        assert normalized_startswith(r"C:\project\file.py", "C:/project/") is True

    @mock.patch("shared.path_utils.sys")
    def test_same_separators_match(self, mock_sys):
        mock_sys.platform = "win32"
        assert normalized_startswith("C:\\project\\file.py", "C:\\project\\") is True

    @mock.patch("shared.path_utils.sys")
    def test_no_match(self, mock_sys):
        mock_sys.platform = "win32"
        assert normalized_startswith(r"C:\other\file.py", "C:/project/") is False

    @mock.patch("shared.path_utils.sys")
    def test_unix_paths_work_normally(self, mock_sys):
        mock_sys.platform = "linux"
        assert normalized_startswith("/home/user/file.py", "/home/user/") is True


# ============================================================================
# normalized_equals() tests
# ============================================================================


class TestNormalizedEquals:
    """Tests for separator-safe equality."""

    @mock.patch("shared.path_utils.sys")
    def test_mixed_separators_equal(self, mock_sys):
        mock_sys.platform = "win32"
        assert normalized_equals(r"C:\project\file.py", "C:/project/file.py") is True

    @mock.patch("shared.path_utils.sys")
    def test_different_paths_not_equal(self, mock_sys):
        mock_sys.platform = "win32"
        assert normalized_equals(r"C:\project\a.py", "C:/project/b.py") is False


# ============================================================================
# normalize_result_path() tests
# ============================================================================


class TestNormalizeResultPath:
    """Tests for tool result path normalization."""

    def setup_method(self):
        is_msys2_environment.cache_clear()

    def teardown_method(self):
        is_msys2_environment.cache_clear()

    @mock.patch("shared.path_utils.is_msys2_environment", return_value=True)
    def test_normalizes_under_msys2(self, _mock):
        assert normalize_result_path(r"src\main\file.py") == "src/main/file.py"

    @mock.patch("shared.path_utils.is_msys2_environment", return_value=True)
    def test_drive_letter_conversion_under_msys2(self, _mock):
        """Tool results should show /c/... format under MSYS2."""
        assert normalize_result_path(r"C:\Users\foo\file.py") == "/c/Users/foo/file.py"

    @mock.patch("shared.path_utils.is_msys2_environment", return_value=False)
    def test_unchanged_when_not_msys2(self, _mock):
        original = r"src\main\file.py"
        assert normalize_result_path(original) == original


# ============================================================================
# get_display_separator() tests
# ============================================================================


class TestGetDisplaySeparator:
    """Tests for display separator selection."""

    def setup_method(self):
        is_msys2_environment.cache_clear()

    def teardown_method(self):
        is_msys2_environment.cache_clear()

    @mock.patch("shared.path_utils.is_msys2_environment", return_value=True)
    def test_forward_slash_under_msys2(self, _mock):
        assert get_display_separator() == "/"

    @mock.patch("shared.path_utils.is_msys2_environment", return_value=False)
    def test_os_sep_when_not_msys2(self, _mock):
        assert get_display_separator() == os.sep
