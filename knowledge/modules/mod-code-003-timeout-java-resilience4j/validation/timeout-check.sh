#!/bin/bash
# =============================================================================
# MOD-003: Timeout Pattern Validation Script
# Tier 3 validation for timeout implementation
# 
# Supports two timeout strategies:
# 1. @TimeLimiter annotation (async methods with CompletableFuture)
# 2. Client-level timeout (RestClient/WebClient with timeout config)
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Counters
ERRORS=0
WARNINGS=0

# Target directory
TARGET_DIR="${1:-.}"

echo "=============================================="
echo "MOD-003: Timeout Pattern Validation"
echo "Target: $TARGET_DIR"
echo "=============================================="

error() { echo -e "${RED}[ERROR]${NC} $1"; ((ERRORS++)) || true; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; ((WARNINGS++)) || true; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
info() { echo -e "[INFO] $1"; }

# Detect which timeout strategy is used
USES_TIMELIMITER=false
USES_CLIENT_TIMEOUT=false

# Check for @TimeLimiter annotation in Java code
if grep -rq "^\s*@TimeLimiter" "$TARGET_DIR/src/main/java" 2>/dev/null; then
    USES_TIMELIMITER=true
fi

# Check for client-level timeout configuration
if grep -rq "setConnectTimeout\|setReadTimeout\|ClientHttpRequestFactory" \
    "$TARGET_DIR/src/main/java" 2>/dev/null; then
    USES_CLIENT_TIMEOUT=true
fi

if grep -qE "connect-timeout|read-timeout" "$TARGET_DIR/src/main/resources/application.yml" 2>/dev/null; then
    USES_CLIENT_TIMEOUT=true
fi

echo ""
echo "--- Strategy Detection ---"

if [ "$USES_TIMELIMITER" = true ]; then
    info "Detected: @TimeLimiter annotation strategy"
elif [ "$USES_CLIENT_TIMEOUT" = true ]; then
    info "Detected: Client-level timeout strategy"
else
    warning "No timeout strategy detected - consider adding timeouts"
fi

# =============================================================================
# STRATEGY 1: @TimeLimiter Validation
# =============================================================================

if [ "$USES_TIMELIMITER" = true ]; then
    echo ""
    echo "--- @TimeLimiter Validation ---"
    
    # Check: @TimeLimiter not in domain layer
    info "Checking @TimeLimiter not in domain layer..."
    if grep -r "@TimeLimiter" "$TARGET_DIR/src/main/java" 2>/dev/null | grep -q "/domain/"; then
        error "@TimeLimiter annotation found in domain layer"
    else
        success "@TimeLimiter not in domain layer"
    fi
    
    # Check: Methods return CompletableFuture
    info "Checking @TimeLimiter methods return CompletableFuture..."
    TIMELIMITER_FILES=$(grep -rl "@TimeLimiter" "$TARGET_DIR/src/main/java" 2>/dev/null || true)
    for file in $TIMELIMITER_FILES; do
        if grep -A5 "@TimeLimiter" "$file" | grep -q "public.*CompletableFuture"; then
            success "$(basename "$file"): Returns CompletableFuture"
        else
            if grep -A5 "@TimeLimiter" "$file" | grep -q "public"; then
                error "$(basename "$file"): @TimeLimiter method must return CompletableFuture"
            fi
        fi
    done
    
    # Check: Annotation order
    info "Checking annotation order (@CircuitBreaker before @TimeLimiter before @Retry)..."
    for file in $TIMELIMITER_FILES; do
        if grep -q "@CircuitBreaker" "$file" && grep -q "@TimeLimiter" "$file"; then
            CB_LINE=$(grep -n "@CircuitBreaker" "$file" | head -1 | cut -d: -f1)
            TL_LINE=$(grep -n "@TimeLimiter" "$file" | head -1 | cut -d: -f1)
            if [ -n "$CB_LINE" ] && [ -n "$TL_LINE" ] && [ "$TL_LINE" -lt "$CB_LINE" ]; then
                error "$(basename "$file"): @TimeLimiter before @CircuitBreaker - wrong order"
            fi
        fi
        
        if grep -q "@TimeLimiter" "$file" && grep -q "@Retry" "$file"; then
            TL_LINE=$(grep -n "@TimeLimiter" "$file" | head -1 | cut -d: -f1)
            R_LINE=$(grep -n "@Retry" "$file" | head -1 | cut -d: -f1)
            if [ -n "$TL_LINE" ] && [ -n "$R_LINE" ] && [ "$R_LINE" -lt "$TL_LINE" ]; then
                error "$(basename "$file"): @Retry before @TimeLimiter - wrong order"
            fi
        fi
    done
    if [ $ERRORS -eq 0 ]; then
        success "Annotation order correct"
    fi
fi

# =============================================================================
# STRATEGY 2: Client-level Timeout Validation
# =============================================================================

if [ "$USES_CLIENT_TIMEOUT" = true ]; then
    echo ""
    echo "--- Client-level Timeout Validation ---"
    
    # Check: Timeout configuration in application.yml
    info "Checking timeout configuration..."
    if [ -f "$TARGET_DIR/src/main/resources/application.yml" ]; then
        if grep -qE "connect-timeout|read-timeout|connectTimeout|readTimeout" \
            "$TARGET_DIR/src/main/resources/application.yml"; then
            success "Timeout configuration found in application.yml"
        else
            warning "Timeout not configured in application.yml"
        fi
    fi
    
    # Check: RestClient/WebClient with timeout
    info "Checking client timeout setup..."
    if grep -rq "setConnectTimeout\|setReadTimeout\|ClientHttpRequestFactory" \
        "$TARGET_DIR/src/main/java" 2>/dev/null; then
        success "Client timeout factory configured"
    elif grep -rq "\.timeout\|Duration\." "$TARGET_DIR/src/main/java" 2>/dev/null; then
        success "Timeout configuration found in Java code"
    else
        warning "No explicit timeout setup in client configuration"
    fi
fi

# =============================================================================
# Common Checks
# =============================================================================

echo ""
echo "--- Common Checks ---"

# Check: Resilience4j timelimiter config (if @TimeLimiter used)
if [ "$USES_TIMELIMITER" = true ]; then
    info "Checking Resilience4j timelimiter configuration..."
    if [ -f "$TARGET_DIR/src/main/resources/application.yml" ]; then
        if grep -q "timelimiter:" "$TARGET_DIR/src/main/resources/application.yml"; then
            success "Resilience4j timelimiter configuration found"
            
            if grep -q "timeoutDuration:" "$TARGET_DIR/src/main/resources/application.yml"; then
                success "timeoutDuration configured"
            else
                warning "timeoutDuration not explicitly set - using defaults"
            fi
        else
            error "timelimiter configuration missing (required for @TimeLimiter)"
        fi
    fi
fi

# Check: Dependencies
info "Checking dependencies..."
if [ -f "$TARGET_DIR/pom.xml" ]; then
    if grep -q "resilience4j" "$TARGET_DIR/pom.xml"; then
        success "Resilience4j dependency found"
    elif [ "$USES_TIMELIMITER" = true ]; then
        error "Resilience4j dependency required for @TimeLimiter"
    fi
    
    if grep -q "spring-boot-starter-aop" "$TARGET_DIR/pom.xml"; then
        success "Spring AOP dependency found"
    elif [ "$USES_TIMELIMITER" = true ]; then
        error "spring-boot-starter-aop required for @TimeLimiter"
    fi
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "Validation Summary"
echo "=============================================="

if [ "$USES_TIMELIMITER" = true ]; then
    echo "Strategy: @TimeLimiter (async)"
elif [ "$USES_CLIENT_TIMEOUT" = true ]; then
    echo "Strategy: Client-level timeout (sync)"
else
    echo "Strategy: None detected"
fi

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
