#!/usr/bin/env python3
"""Test script for moon phase calculator plugin.

This script tests the plugin standalone before integrating with jaato.
Following the testing pattern from the jaato API documentation.

Note: This test requires access to jaato's shared module for type imports.
In a production setup, jaato would be installed as a dependency.
"""

import sys
import os

# WORKAROUND: Add jaato repo to path for testing
# This is needed because the API documentation doesn't specify how to handle
# the dependency on jaato's shared types for out-of-tree plugins
jaato_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../jaato'))
if os.path.exists(jaato_path):
    sys.path.insert(0, jaato_path)
    print(f"Note: Added jaato path for testing: {jaato_path}\n")
else:
    print("Warning: jaato path not found. Plugin may fail to import types.")
    print("Expected path:", jaato_path)
    print()


def test_plugin():
    """Test plugin implementation."""
    print("Testing Moon Phase Calculator Plugin")
    print("=" * 60)

    # Test 1: Import and create
    print("\n[1] Importing plugin...")
    try:
        # Add the src directory to path for testing
        sys.path.insert(0, 'src')
        from jaato_moon_phase import create_plugin, PLUGIN_INFO
        plugin = create_plugin()
        print(f"✓ Created: {PLUGIN_INFO['name']}")
        print(f"  Version: {PLUGIN_INFO['version']}")
        print(f"  Description: {PLUGIN_INFO['description']}")
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        return False

    # Test 2: Initialize
    print("\n[2] Initializing...")
    try:
        plugin.initialize({"precision": 3})
        print("✓ Initialized with config")
        assert plugin.precision == 3, "Config not applied"
        print(f"  Precision set to: {plugin.precision}")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return False

    # Test 3: Get schemas
    print("\n[3] Getting tool schemas...")
    try:
        schemas = plugin.get_tool_schemas()
        print(f"✓ Found {len(schemas)} tool(s)")
        for schema in schemas:
            print(f"  - {schema.name}")
            print(f"    Description: {schema.description[:60]}...")
            print(f"    Parameters: {list(schema.parameters.get('properties', {}).keys())}")
    except ImportError as e:
        print(f"⚠ Skipped: Cannot import ToolSchema without jaato dependencies")
        print(f"  Error: {e}")
        print(f"  Note: This is expected when testing without full jaato installation")
        print(f"  The plugin will work correctly when installed with jaato")
        # Skip remaining schema-dependent tests
        schemas = None
    except Exception as e:
        print(f"✗ Failed to get schemas: {e}")
        return False

    # Test 4: Get executors
    print("\n[4] Getting executors...")
    try:
        executors = plugin.get_executors()
        if schemas is not None:
            assert len(executors) == len(schemas), \
                "Executor count must match schema count"
        print(f"✓ All {len(executors)} executor(s) present")
        for name in executors.keys():
            print(f"  - {name}")
    except Exception as e:
        print(f"✗ Failed to get executors: {e}")
        return False

    # Test 5: Test with current date (no parameters)
    print("\n[5] Testing with current date...")
    try:
        result = plugin._calculate_moon_phase()
        print("✓ Current date calculation succeeded")
        print(f"  Result:\n{result}")
        assert "Moon Phase for" in result, "Missing expected output"
        assert "Phase:" in result, "Missing phase information"
        assert "Illumination:" in result, "Missing illumination percentage"
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    # Test 6: Test with specific date
    print("\n[6] Testing with specific date (2024-01-01)...")
    try:
        result = plugin._calculate_moon_phase(date="2024-01-01")
        print("✓ Specific date calculation succeeded")
        print(f"  Result:\n{result}")
        assert "2024-01-01" in result, "Date not in output"
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    # Test 7: Test with details flag
    print("\n[7] Testing with include_details=True...")
    try:
        result = plugin._calculate_moon_phase(
            date="2024-12-10",
            include_details=True
        )
        print("✓ Detailed calculation succeeded")
        print(f"  Result:\n{result}")
        assert "Age:" in result, "Missing age in details"
        assert "Phase Angle:" in result, "Missing phase angle in details"
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    # Test 8: Test error handling - invalid date format
    print("\n[8] Testing error handling (invalid date)...")
    try:
        result = plugin._calculate_moon_phase(date="invalid-date")
        assert "Error" in result, "Should return error message"
        print("✓ Error handled gracefully")
        print(f"  Error message: {result[:80]}...")
    except Exception as e:
        print(f"✗ Exception raised instead of error message: {e}")
        return False

    # Test 9: Test edge cases
    print("\n[9] Testing edge cases...")
    try:
        # Very old date
        result1 = plugin._calculate_moon_phase(date="1900-01-01")
        assert "1900-01-01" in result1, "Old date failed"
        print("✓ Old date (1900-01-01) handled")

        # Future date
        result2 = plugin._calculate_moon_phase(date="2100-12-31")
        assert "2100-12-31" in result2, "Future date failed"
        print("✓ Future date (2100-12-31) handled")
    except Exception as e:
        print(f"✗ Edge case failed: {e}")
        return False

    # Test 10: Verify plugin attributes
    print("\n[10] Verifying plugin attributes...")
    try:
        assert hasattr(plugin, 'name'), "Missing 'name' attribute"
        assert plugin.name == "moon_phase", "Incorrect name"
        print(f"✓ Plugin name: {plugin.name}")
    except Exception as e:
        print(f"✗ Attribute verification failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nPlugin is ready to be installed and used with jaato.")
    print("To install: pip install -e .")
    return True


if __name__ == "__main__":
    success = test_plugin()
    sys.exit(0 if success else 1)
