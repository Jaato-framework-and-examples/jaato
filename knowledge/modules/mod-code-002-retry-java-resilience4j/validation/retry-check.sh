#!/bin/bash
# =============================================================================
# MOD-002: Retry Pattern Validation Script
# Tier 3 validation for Resilience4j Retry implementation
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
ERRORS=0
WARNINGS=0

# Target directory (passed as argument)
TARGET_DIR="${1:-.}"

echo "=============================================="
echo "MOD-002: Retry Pattern Validation"
echo "Target: $TARGET_DIR"
echo "=============================================="

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((ERRORS++))
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    ((WARNINGS++))
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

info() {
    echo -e "[INFO] $1"
}

# -----------------------------------------------------------------------------
# Validation checks
# -----------------------------------------------------------------------------

echo ""
echo "--- Structural Constraints ---"

# Check 1: @Retry not in domain layer
info "Checking @Retry not in domain layer..."
if grep -r "@Retry" "$TARGET_DIR/src/main/java" 2>/dev/null | grep -q "/domain/"; then
    error "@Retry annotation found in domain layer - MUST be in application layer only"
else
    success "@Retry not in domain layer"
fi

# Check 2: @Retry not directly on controllers
info "Checking @Retry not on controllers..."
if grep -r "@Retry" "$TARGET_DIR/src/main/java" 2>/dev/null | grep -q "/controller/"; then
    warning "@Retry found on controller - SHOULD be on application service"
else
    success "@Retry not on controllers"
fi

# Check 3: @Retry exists somewhere in application layer
info "Checking @Retry exists in application layer..."
if grep -r "@Retry" "$TARGET_DIR/src/main/java" 2>/dev/null | grep -q "/application/"; then
    success "@Retry found in application layer"
else
    warning "@Retry not found in application layer - expected for retry pattern"
fi

# Check 4: CircuitBreaker before Retry (if both exist)
info "Checking annotation order (CircuitBreaker before Retry)..."
if grep -r "@Retry" "$TARGET_DIR/src/main/java" 2>/dev/null | grep -q "@CircuitBreaker"; then
    # Both annotations exist, check order
    FILES_WITH_BOTH=$(grep -rl "@Retry" "$TARGET_DIR/src/main/java" 2>/dev/null | xargs grep -l "@CircuitBreaker" 2>/dev/null || true)
    ORDER_OK=true
    for file in $FILES_WITH_BOTH; do
        # Check if @CircuitBreaker comes before @Retry
        CB_LINE=$(grep -n "@CircuitBreaker" "$file" | head -1 | cut -d: -f1)
        RETRY_LINE=$(grep -n "@Retry" "$file" | head -1 | cut -d: -f1)
        if [ -n "$CB_LINE" ] && [ -n "$RETRY_LINE" ]; then
            if [ "$RETRY_LINE" -lt "$CB_LINE" ]; then
                error "In $file: @Retry appears before @CircuitBreaker - order MUST be @CircuitBreaker then @Retry"
                ORDER_OK=false
            fi
        fi
    done
    if [ "$ORDER_OK" = true ]; then
        success "Annotation order correct (@CircuitBreaker before @Retry)"
    fi
else
    info "Only @Retry found (no @CircuitBreaker combination)"
fi

echo ""
echo "--- Configuration Constraints ---"

# Check 5: Retry configuration exists in application.yml
info "Checking retry configuration exists..."
if [ -f "$TARGET_DIR/src/main/resources/application.yml" ]; then
    if grep -q "resilience4j:" "$TARGET_DIR/src/main/resources/application.yml" && \
       grep -q "retry:" "$TARGET_DIR/src/main/resources/application.yml"; then
        success "Resilience4j retry configuration found"
    else
        error "resilience4j.retry configuration not found in application.yml"
    fi
else
    error "application.yml not found"
fi

# Check 6: retryExceptions configured
info "Checking retryExceptions configured..."
if [ -f "$TARGET_DIR/src/main/resources/application.yml" ]; then
    if grep -q "retryExceptions:" "$TARGET_DIR/src/main/resources/application.yml"; then
        success "retryExceptions configured"
    else
        warning "retryExceptions not explicitly configured - SHOULD be defined"
    fi
fi

# Check 7: ignoreExceptions configured
info "Checking ignoreExceptions configured..."
if [ -f "$TARGET_DIR/src/main/resources/application.yml" ]; then
    if grep -q "ignoreExceptions:" "$TARGET_DIR/src/main/resources/application.yml"; then
        success "ignoreExceptions configured"
    else
        warning "ignoreExceptions not configured - SHOULD include business exceptions"
    fi
fi

echo ""
echo "--- Dependency Constraints ---"

# Check 8: Resilience4j dependency
info "Checking Resilience4j dependency..."
if [ -f "$TARGET_DIR/pom.xml" ]; then
    if grep -q "resilience4j-spring-boot" "$TARGET_DIR/pom.xml"; then
        success "Resilience4j dependency found"
    else
        error "Resilience4j dependency not found in pom.xml"
    fi
elif [ -f "$TARGET_DIR/build.gradle" ]; then
    if grep -q "resilience4j" "$TARGET_DIR/build.gradle"; then
        success "Resilience4j dependency found"
    else
        error "Resilience4j dependency not found in build.gradle"
    fi
else
    warning "No pom.xml or build.gradle found"
fi

# Check 9: Spring AOP dependency
info "Checking Spring AOP dependency..."
if [ -f "$TARGET_DIR/pom.xml" ]; then
    if grep -q "spring-boot-starter-aop" "$TARGET_DIR/pom.xml"; then
        success "Spring AOP dependency found"
    else
        error "spring-boot-starter-aop dependency not found - required for @Retry"
    fi
fi

echo ""
echo "--- Testing Constraints ---"

# Check 10: Test files exist for services with @Retry
info "Checking test coverage for retry..."
RETRY_FILES=$(grep -rl "@Retry" "$TARGET_DIR/src/main/java" 2>/dev/null || true)
for file in $RETRY_FILES; do
    # Extract class name
    CLASS_NAME=$(basename "$file" .java)
    TEST_FILE="$TARGET_DIR/src/test/java"
    if find "$TEST_FILE" -name "${CLASS_NAME}Test.java" 2>/dev/null | grep -q .; then
        success "Test found for $CLASS_NAME"
    else
        warning "No test found for $CLASS_NAME with @Retry"
    fi
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "Validation Summary"
echo "=============================================="
echo -e "Errors:   ${RED}$ERRORS${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo ""

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}VALIDATION FAILED${NC}"
    exit 1
else
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}VALIDATION PASSED WITH WARNINGS${NC}"
    else
        echo -e "${GREEN}VALIDATION PASSED${NC}"
    fi
    exit 0
fi
