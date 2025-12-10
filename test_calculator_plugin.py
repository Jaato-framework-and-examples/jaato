#!/usr/bin/env python3
"""
Test script for the calculator plugin.
This verifies that the plugin was built correctly using only the API documentation.
"""

import sys
import json


def test_calculator_plugin():
    """Test the calculator plugin implementation."""
    print("=" * 60)
    print("Testing Calculator Plugin")
    print("=" * 60)

    # Test 1: Import the plugin
    print("\n[Test 1] Importing plugin...")
    try:
        from shared.plugins.calculator import create_plugin, PLUGIN_INFO
        print("✓ Plugin imported successfully")
        print(f"  Plugin Info: {PLUGIN_INFO}")
    except ImportError as e:
        print(f"✗ Failed to import plugin: {e}")
        return False

    # Test 2: Create plugin instance
    print("\n[Test 2] Creating plugin instance...")
    try:
        plugin = create_plugin()
        print("✓ Plugin instance created")
    except Exception as e:
        print(f"✗ Failed to create plugin: {e}")
        return False

    # Test 3: Initialize plugin
    print("\n[Test 3] Initializing plugin...")
    try:
        plugin.initialize({"precision": 3})
        print("✓ Plugin initialized with config")
        print(f"  Precision set to: {plugin.precision}")
    except Exception as e:
        print(f"✗ Failed to initialize plugin: {e}")
        return False

    # Test 4: Get tool schemas
    print("\n[Test 4] Getting tool schemas...")
    try:
        schemas = plugin.get_tool_schemas()
        print(f"✓ Got {len(schemas)} tool schemas")
        for schema in schemas:
            print(f"  - {schema.name}: {schema.description[:50]}...")
    except Exception as e:
        print(f"✗ Failed to get tool schemas: {e}")
        return False

    # Test 5: Get executors
    print("\n[Test 5] Getting executors...")
    try:
        executors = plugin.get_executors()
        print(f"✓ Got {len(executors)} executors")
        print(f"  Executor names: {list(executors.keys())}")
    except Exception as e:
        print(f"✗ Failed to get executors: {e}")
        return False

    # Test 6: Test add operation
    print("\n[Test 6] Testing add operation...")
    try:
        result = plugin._add(5, 3)
        print(f"✓ Add result: {result}")
        data = json.loads(result)
        assert data["result"] == 8.0, f"Expected 8.0, got {data['result']}"
        print("  Assertion passed: 5 + 3 = 8")
    except Exception as e:
        print(f"✗ Add operation failed: {e}")
        return False

    # Test 7: Test subtract operation
    print("\n[Test 7] Testing subtract operation...")
    try:
        result = plugin._subtract(10, 4)
        print(f"✓ Subtract result: {result}")
        data = json.loads(result)
        assert data["result"] == 6.0, f"Expected 6.0, got {data['result']}"
        print("  Assertion passed: 10 - 4 = 6")
    except Exception as e:
        print(f"✗ Subtract operation failed: {e}")
        return False

    # Test 8: Test multiply operation
    print("\n[Test 8] Testing multiply operation...")
    try:
        result = plugin._multiply(6, 7)
        print(f"✓ Multiply result: {result}")
        data = json.loads(result)
        assert data["result"] == 42.0, f"Expected 42.0, got {data['result']}"
        print("  Assertion passed: 6 * 7 = 42")
    except Exception as e:
        print(f"✗ Multiply operation failed: {e}")
        return False

    # Test 9: Test divide operation
    print("\n[Test 9] Testing divide operation...")
    try:
        result = plugin._divide(20, 4)
        print(f"✓ Divide result: {result}")
        data = json.loads(result)
        assert data["result"] == 5.0, f"Expected 5.0, got {data['result']}"
        print("  Assertion passed: 20 / 4 = 5")
    except Exception as e:
        print(f"✗ Divide operation failed: {e}")
        return False

    # Test 10: Test divide by zero
    print("\n[Test 10] Testing divide by zero error handling...")
    try:
        result = plugin._divide(10, 0)
        print(f"✓ Divide by zero handled: {result}")
        assert "Error" in result, "Expected error message"
        print("  Assertion passed: Division by zero properly handled")
    except Exception as e:
        print(f"✗ Divide by zero test failed: {e}")
        return False

    # Test 11: Test calculate operation
    print("\n[Test 11] Testing calculate operation...")
    try:
        result = plugin._calculate("2 + 3 * 4")
        print(f"✓ Calculate result: {result}")
        data = json.loads(result)
        assert data["result"] == 14.0, f"Expected 14.0, got {data['result']}"
        print("  Assertion passed: 2 + 3 * 4 = 14")
    except Exception as e:
        print(f"✗ Calculate operation failed: {e}")
        return False

    # Test 12: Test calculate with parentheses
    print("\n[Test 12] Testing calculate with parentheses...")
    try:
        result = plugin._calculate("(2 + 3) * 4")
        print(f"✓ Calculate result: {result}")
        data = json.loads(result)
        assert data["result"] == 20.0, f"Expected 20.0, got {data['result']}"
        print("  Assertion passed: (2 + 3) * 4 = 20")
    except Exception as e:
        print(f"✗ Calculate with parentheses failed: {e}")
        return False

    # Test 13: Test invalid expression error handling
    print("\n[Test 13] Testing invalid expression error handling...")
    try:
        result = plugin._calculate("invalid + + expression")
        print(f"✓ Invalid expression handled: {result}")
        assert "Error" in result, "Expected error message"
        print("  Assertion passed: Invalid expression properly handled")
    except Exception as e:
        print(f"✗ Invalid expression test failed: {e}")
        return False

    # Test 14: Test precision setting
    print("\n[Test 14] Testing precision setting...")
    try:
        plugin.initialize({"precision": 4})
        result = plugin._divide(10, 3)
        data = json.loads(result)
        print(f"✓ Precision result: {result}")
        # 10/3 = 3.3333... rounded to 4 decimal places
        assert data["result"] == 3.3333, f"Expected 3.3333, got {data['result']}"
        print("  Assertion passed: Precision setting works correctly")
    except Exception as e:
        print(f"✗ Precision test failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_calculator_plugin()
    sys.exit(0 if success else 1)
