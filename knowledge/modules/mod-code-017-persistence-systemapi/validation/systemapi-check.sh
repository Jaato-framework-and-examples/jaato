#!/bin/bash
# =============================================================================
# MOD-017: System API Persistence Validation Script
# Tier 3 validation for System API persistence implementation
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

TARGET_DIR="${1:-.}"

echo "=============================================="
echo "MOD-017: System API Persistence Validation"
echo "Target: $TARGET_DIR"
echo "=============================================="

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

echo ""
echo "--- Structural Constraints ---"

# Check 1: Repository interface in domain
info "Checking repository interface in domain/repository/..."
if find "$TARGET_DIR/src/main/java" -path "*/domain/repository/*Repository.java" 2>/dev/null | grep -q .; then
    success "Repository interface found in domain/repository/"
else
    error "Repository interface not found in domain/repository/"
fi

# Check 2: System API adapter in correct location
info "Checking System API adapter in adapter/systemapi/..."
if find "$TARGET_DIR/src/main/java" -path "*/adapter/systemapi/*Adapter.java" 2>/dev/null | grep -q .; then
    success "System API adapter found in adapter/systemapi/"
else
    error "System API adapter not found in adapter/systemapi/"
fi

# Check 3: Client in adapter/systemapi/client
info "Checking client in adapter/systemapi/client/..."
if find "$TARGET_DIR/src/main/java" -path "*/adapter/systemapi/client/*Client.java" 2>/dev/null | grep -q .; then
    success "System API client found in adapter/systemapi/client/"
else
    warning "System API client not found in adapter/systemapi/client/"
fi

echo ""
echo "--- Resilience Constraints ---"

# Check 4: CircuitBreaker annotation present
info "Checking @CircuitBreaker on adapter..."
if grep -r "@CircuitBreaker" "$TARGET_DIR/src/main/java" 2>/dev/null | grep -q "systemapi"; then
    success "@CircuitBreaker found on System API adapter"
else
    error "@CircuitBreaker MUST be present on System API adapter methods"
fi

# Check 5: Retry annotation present
info "Checking @Retry on adapter..."
if grep -r "@Retry" "$TARGET_DIR/src/main/java" 2>/dev/null | grep -q "systemapi"; then
    success "@Retry found on System API adapter"
else
    error "@Retry MUST be present on System API adapter methods"
fi

# Check 6: Resilience4j configuration
info "Checking Resilience4j configuration..."
if [ -f "$TARGET_DIR/src/main/resources/application.yml" ]; then
    if grep -q "resilience4j:" "$TARGET_DIR/src/main/resources/application.yml"; then
        success "Resilience4j configuration found"
    else
        error "Resilience4j configuration not found in application.yml"
    fi
fi

echo ""
echo "--- Configuration Constraints ---"

# Check 7: Base URL externalized
info "Checking base URL externalization..."
if grep -r "base-url\|baseUrl" "$TARGET_DIR/src/main/resources" 2>/dev/null | grep -q '\${'; then
    success "Base URL uses environment variable"
else
    warning "Base URL should be externalized via environment variable"
fi

# Check 8: Timeouts configured
info "Checking timeout configuration..."
if grep -r "timeout" "$TARGET_DIR/src/main/resources/application.yml" 2>/dev/null | grep -q .; then
    success "Timeout configuration found"
else
    warning "Connection/read timeouts should be configured"
fi

echo ""
echo "--- Headers Constraints ---"

# Check 9: Correlation headers
info "Checking correlation headers..."
if grep -r "X-Correlation-Id\|X-Source-System" "$TARGET_DIR/src/main/java" 2>/dev/null | grep -q .; then
    success "Correlation headers found"
else
    warning "X-Correlation-Id and X-Source-System headers should be set"
fi

echo ""
echo "--- Dependency Constraints ---"

# Check 10: Resilience4j dependency
info "Checking Resilience4j dependency..."
if [ -f "$TARGET_DIR/pom.xml" ]; then
    if grep -q "resilience4j" "$TARGET_DIR/pom.xml"; then
        success "Resilience4j dependency found"
    else
        error "Resilience4j dependency not found in pom.xml"
    fi
fi

echo ""
echo "--- Testing Constraints ---"

# Check 11: Adapter tests exist
info "Checking System API adapter tests..."
if find "$TARGET_DIR/src/test/java" -name "*SystemApiAdapter*Test.java" 2>/dev/null | grep -q .; then
    success "System API adapter tests found"
else
    warning "No System API adapter tests found"
fi

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
