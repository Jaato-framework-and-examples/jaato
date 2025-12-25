#!/bin/bash
# circuit-breaker-check.sh
# Validates Circuit Breaker implementation (Resilience4j pattern)

SERVICE_DIR=${1:-.}

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✅ PASS:${NC} $1"; }
fail() { echo -e "${RED}❌ FAIL:${NC} $1"; ERRORS=$((ERRORS + 1)); }
warn() { echo -e "${YELLOW}⚠️  WARN:${NC} $1"; }

ERRORS=0

# 1. Check @CircuitBreaker annotation exists
CB_COUNT=$(find "$SERVICE_DIR/src/main/java" -name "*.java" -exec grep -l "@CircuitBreaker" {} \; 2>/dev/null | wc -l)
if [ "$CB_COUNT" -gt 0 ]; then
    pass "@CircuitBreaker annotation found in $CB_COUNT file(s)"
    find "$SERVICE_DIR/src/main/java" -name "*.java" -exec grep -l "@CircuitBreaker" {} \; 2>/dev/null | while read file; do
        echo "     - $(basename $file)"
    done
else
    fail "@CircuitBreaker annotation not found"
fi

# 2. Check fallback methods are defined
FALLBACK_COUNT=$(grep -r "fallbackMethod" "$SERVICE_DIR/src/main/java" 2>/dev/null | wc -l)
if [ "$FALLBACK_COUNT" -gt 0 ]; then
    pass "Fallback methods defined ($FALLBACK_COUNT reference(s))"
else
    fail "Fallback methods not defined"
fi

# 3. Verify fallback methods exist (not just referenced)
if [ "$FALLBACK_COUNT" -gt 0 ]; then
    # Extract fallback method names
    FALLBACK_METHODS=$(grep -oP 'fallbackMethod\s*=\s*"\K[^"]+' "$SERVICE_DIR/src/main/java"/**/*.java 2>/dev/null || true)
    
    for method in $FALLBACK_METHODS; do
        if grep -r "private.*$method\|public.*$method" "$SERVICE_DIR/src/main/java" > /dev/null 2>&1; then
            pass "Fallback method implemented: $method()"
        else
            fail "Fallback method declared but not implemented: $method()"
        fi
    done
fi

# 4. Check resilience4j configuration in application.yml
if [ -f "$SERVICE_DIR/src/main/resources/application.yml" ]; then
    if grep -q "resilience4j:" "$SERVICE_DIR/src/main/resources/application.yml"; then
        pass "Resilience4j configuration present in application.yml"
        
        # Check specific circuit breaker instance configuration
        if grep -A 10 "circuitbreaker:" "$SERVICE_DIR/src/main/resources/application.yml" | grep -q "instances:"; then
            pass "Circuit breaker instances configured"
        else
            warn "Circuit breaker instances not explicitly configured"
        fi
    else
        fail "Resilience4j configuration missing in application.yml"
    fi
else
    warn "application.yml not found"
fi

# 5. Check pom.xml has resilience4j dependency
if [ -f "$SERVICE_DIR/pom.xml" ]; then
    if grep -q "resilience4j-spring-boot" "$SERVICE_DIR/pom.xml"; then
        pass "resilience4j-spring-boot dependency in pom.xml"
    else
        fail "resilience4j-spring-boot dependency missing in pom.xml"
    fi
    
    if grep -q "spring-boot-starter-aop" "$SERVICE_DIR/pom.xml"; then
        pass "spring-boot-starter-aop dependency (required for CB)"
    else
        warn "spring-boot-starter-aop dependency missing (CB may not work)"
    fi
else
    warn "pom.xml not found"
fi

# 6. Check actuator circuit breaker endpoint
if [ -f "$SERVICE_DIR/src/main/resources/application.yml" ]; then
    if grep -q "circuitbreakers" "$SERVICE_DIR/src/main/resources/application.yml"; then
        pass "Circuit breaker metrics exposed via actuator"
    else
        warn "Circuit breaker metrics not exposed (/actuator/circuitbreakers)"
    fi
fi

# 7. Verify @CircuitBreaker is on application service (not domain)
CB_FILES=$(find "$SERVICE_DIR/src/main/java" -name "*.java" -exec grep -l "@CircuitBreaker" {} \; 2>/dev/null)
for file in $CB_FILES; do
    if echo "$file" | grep -q "/application/"; then
        pass "@CircuitBreaker correctly placed in application layer"
    elif echo "$file" | grep -q "/domain/"; then
        fail "@CircuitBreaker in domain layer (should be in application layer)"
    fi
done

echo ""
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ Circuit Breaker Feature: VALIDATED${NC}"
    exit 0
else
    echo -e "${RED}❌ Circuit Breaker Feature: VALIDATION FAILED${NC}"
    echo "   Errors: $ERRORS"
    exit 1
fi
