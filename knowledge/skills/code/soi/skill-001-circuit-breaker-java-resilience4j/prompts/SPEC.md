# Circuit Breaker Addition - Execution Specification

**Skill:** skill-001-add-circuit-breaker-java-resilience4j  
**Version:** 2.0  
**Module:** mod-code-001-circuit-breaker-java-resilience4j

---

## Task

Transform Java class by adding circuit breaker pattern to specified method.

---

## Execution Steps

### Step 1: Load Module

**Action:** Load templates

```
Read: ../../modules/mod-code-001-circuit-breaker-java-resilience4j.md
Templates available:
  - Template 1: Basic with Single Fallback
  - Template 2: Multiple Fallbacks Chain
  - Template 3: Fail Fast (No Fallback)
  - Template 4: Programmatic
```

### Step 2: Parse Target Class

**Actions:**
1. Read target Java file
2. Locate method by name
3. Extract signature:
   - Return type
   - Parameters
   - Exceptions
4. Extract method body

### Step 3: Select Template

**Logic:**
```
if pattern.type == "basic_fallback": use Template 1
elif pattern.type == "multiple_fallbacks": use Template 2
elif pattern.type == "fail_fast": use Template 3
elif pattern.type == "programmatic": use Template 4
else: default Template 1
```

### Step 4: Generate Code

**Actions:**
1. Prepare variables:
   - {{circuitBreakerName}}
   - {{fallbackMethodName}}
   - {{returnType}}
   - {{methodName}}
   - {{methodParameters}}
   - {{originalMethodBody}}

2. Apply template from module

3. Generate fallback logic based on returnType:
   - Optional<T> → Optional.empty()
   - Collection → Collections.emptyList()
   - Custom → new Object() or failed()
   - Primitive → 0, false, etc.

### Step 5: Add Imports

```java
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
```

### Step 6: Add Logger (if missing)

```java
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

private static final Logger log = LoggerFactory.getLogger(ClassName.class);
```

### Step 7: Update pom.xml

Add Resilience4j dependency if not present.

### Step 8: Generate Configuration

Use config template from module:
```yaml
resilience4j:
  circuitbreaker:
    instances:
      {{circuitBreakerName}}:
        failure-rate-threshold: {{value}}
        ...
```

### Step 9: Validate

Check:
- Annotation syntax
- Fallback signature (if applicable)
- Throwable parameter (if applicable)
- No {{variables}} remaining

---

## Critical Rules

### MUST:
1. Add Throwable parameter to fallback
2. Use exact template from module
3. Preserve original logic
4. Add logging in fallback
5. Validate output

### MUST NOT:
1. Modify business logic
2. Change method signature (except fail_fast)
3. Throw in fallback (except fail_fast)
4. Hardcode values

---

## Module Reference

**Location:** ../../modules/mod-code-001-circuit-breaker-java-resilience4j.md

**Usage:**
- Load at Step 1
- Select at Step 3
- Apply at Step 4

---

## Output Format

```json
{
  "status": "SUCCESS",
  "modifiedFiles": [...],
  "output": {...}
}
```

---

**Version:** 2.0  
**Last Updated:** 2025-11-21
