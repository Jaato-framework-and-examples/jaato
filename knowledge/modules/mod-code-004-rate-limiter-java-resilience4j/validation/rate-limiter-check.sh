#!/bin/bash
# =============================================================================
# MOD-004: Rate Limiter Pattern Validation Script
# Tier 3 validation for Resilience4j RateLimiter implementation
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
echo "MOD-004: Rate Limiter Pattern Validation"
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

# Check 1: @RateLimiter not in domain layer
info "Checking @RateLimiter not in domain layer..."
if grep -r "@RateLimiter" "$TARGET_DIR/src/main/java" 2>/dev/null | grep -q "/domain/"; then
    error "@RateLimiter annotation found in domain layer - MUST be in application layer only"
else
    success "@RateLimiter not in domain layer"
fi

# Check 2: @RateLimiter not directly on controllers
info "Checking @RateLimiter not on controllers..."
if grep -r "@RateLimiter" "$TARGET_DIR/src/main/java" 2>/dev/null | grep -q "/controller/"; then
    warning "@RateLimiter found on controller - SHOULD be on application service"
else
    success "@RateLimiter not on controllers"
fi

# Check 3: Annotation order (RateLimiter before Retry)
info "Checking annotation order..."
RATELIMITER_FILES=$(grep -rl "@RateLimiter" "$TARGET_DIR/src/main/java" 2>/dev/null || true)
for file in $RATELIMITER_FILES; do
    if grep -q "@Retry" "$file"; then
        RL_LINE=$(grep -n "@RateLimiter" "$file" | head -1 | cut -d: -f1)
        R_LINE=$(grep -n "@Retry" "$file" | head -1 | cut -d: -f1)
        if [ -n "$RL_LINE" ] && [ -n "$R_LINE" ] && [ "$R_LINE" -lt "$RL_LINE" ]; then
            error "In $file: @Retry appears before @RateLimiter - wrong order"
        fi
    fi
done
success "Annotation order checked"

echo ""
echo "--- Configuration Constraints ---"

# Check 4: RateLimiter configuration exists in application.yml
info "Checking ratelimiter configuration exists..."
if [ -f "$TARGET_DIR/src/main/resources/application.yml" ]; then
    if grep -q "resilience4j:" "$TARGET_DIR/src/main/resources/application.yml" && \
       grep -q "ratelimiter:" "$TARGET_DIR/src/main/resources/application.yml"; then
        success "Resilience4j ratelimiter configuration found"
    else
        error "resilience4j.ratelimiter configuration not found in application.yml"
    fi
else
    error "application.yml not found"
fi

# Check 5: limitForPeriod configured
info "Checking limitForPeriod configured..."
if [ -f "$TARGET_DIR/src/main/resources/application.yml" ]; then
    if grep -q "limitForPeriod:" "$TARGET_DIR/src/main/resources/application.yml"; then
        success "limitForPeriod configured"
    else
        error "limitForPeriod not configured - MUST be explicitly defined"
    fi
fi

echo ""
echo "--- Dependency Constraints ---"

# Check 6: Resilience4j dependency
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

# Check 7: Spring AOP dependency
info "Checking Spring AOP dependency..."
if [ -f "$TARGET_DIR/pom.xml" ]; then
    if grep -q "spring-boot-starter-aop" "$TARGET_DIR/pom.xml"; then
        success "Spring AOP dependency found"
    else
        error "spring-boot-starter-aop dependency not found - required for @RateLimiter"
    fi
fi

echo ""
echo "--- Exception Handling ---"

# Check 8: RequestNotPermitted exception handling exists
info "Checking RequestNotPermitted exception handling..."
if grep -r "RequestNotPermitted" "$TARGET_DIR/src/main/java" 2>/dev/null | grep -q "ExceptionHandler\|fallbackMethod"; then
    success "RequestNotPermitted handling found"
else
    warning "No explicit RequestNotPermitted handling found - ensure fallback or exception handler exists"
fi

echo ""
echo "--- Testing Constraints ---"

# Check 9: Test files exist for services with @RateLimiter
info "Checking test coverage for rate limiter..."
for file in $RATELIMITER_FILES; do
    CLASS_NAME=$(basename "$file" .java)
    TEST_FILE="$TARGET_DIR/src/test/java"
    if find "$TEST_FILE" -name "${CLASS_NAME}Test.java" 2>/dev/null | grep -q .; then
        success "Test found for $CLASS_NAME"
    else
        warning "No test found for $CLASS_NAME with @RateLimiter"
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
