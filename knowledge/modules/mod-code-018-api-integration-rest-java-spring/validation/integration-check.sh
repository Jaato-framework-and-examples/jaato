#!/bin/bash
# Validation script for mod-018-api-integration-rest-java-spring
# Tier-3 validation: Module-specific checks

set -e

TARGET_DIR="${1:-.}"
ERRORS=0
WARNINGS=0

echo "=== Integration REST Validation ==="
echo "Target: $TARGET_DIR"
echo ""

# Find client files
CLIENT_FILES=$(find "$TARGET_DIR" -name "*Client.java" -path "*/integration/*" 2>/dev/null || true)

if [ -z "$CLIENT_FILES" ]; then
    echo "⚠️  WARNING: No integration client files found"
    exit 0
fi

for file in $CLIENT_FILES; do
    echo "Checking: $file"
    
    # ERROR: Correlation headers must be propagated
    if ! grep -q "X-Correlation-ID" "$file"; then
        echo "  ❌ ERROR: Missing X-Correlation-ID header propagation"
        ERRORS=$((ERRORS + 1))
    else
        echo "  ✅ X-Correlation-ID header present"
    fi
    
    # ERROR: Base URL must be externalized
    if grep -q 'baseUrl = "http' "$file"; then
        echo "  ❌ ERROR: Base URL is hardcoded, must use environment variable"
        ERRORS=$((ERRORS + 1))
    else
        echo "  ✅ Base URL externalized"
    fi
    
    # WARNING: Source system header
    if ! grep -q "X-Source-System" "$file"; then
        echo "  ⚠️  WARNING: X-Source-System header not set"
        WARNINGS=$((WARNINGS + 1))
    else
        echo "  ✅ X-Source-System header present"
    fi
    
    # WARNING: Logging
    if ! grep -q "log.debug\|log.info" "$file"; then
        echo "  ⚠️  WARNING: No logging found in client"
        WARNINGS=$((WARNINGS + 1))
    else
        echo "  ✅ Logging present"
    fi
    
    echo ""
done

# Summary
echo "=== Summary ==="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"

if [ $ERRORS -gt 0 ]; then
    echo "❌ Validation FAILED"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo "⚠️  Validation passed with warnings"
    exit 2
else
    echo "✅ Validation PASSED"
    exit 0
fi
