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

**Action:** Load templates from module

```
Read: enablement/modules/mod-code-001-circuit-breaker-java-resilience4j/MODULE.md
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
   - Custom → new Object() or .failed()
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

Add Resilience4j dependency if not present:
```xml
<dependency>
    <groupId>io.github.resilience4j</groupId>
    <artifactId>resilience4j-spring-boot3</artifactId>
    <version>2.1.0</version>
</dependency>
```

### Step 8: Generate Configuration

Use config template from module:
```yaml
resilience4j:
  circuitbreaker:
    instances:
      {{circuitBreakerName}}:
        failure-rate-threshold: {{failureRateThreshold}}
        wait-duration-in-open-state: {{waitDurationInOpenState}}s
        sliding-window-size: {{slidingWindowSize}}
        minimum-number-of-calls: {{minimumNumberOfCalls}}
```

### Step 9: Validate

Check:
- Annotation syntax correct
- Fallback signature matches (same params + Throwable)
- Throwable parameter present in fallback
- No {{variables}} remaining in output
- Code compiles

---

## Critical Rules

### MUST:
1. Add Throwable parameter to fallback method
2. Use EXACT templates from module - no modifications
3. Preserve original business logic unchanged
4. Add logging in fallback method
5. Validate all output before completion

### MUST NOT:
1. Modify business logic inside method
2. Change method signature (except fail_fast adds throws)
3. Throw exceptions in fallback methods (except fail_fast)
4. Hardcode configuration values - use variables

---

## Output Format

```json
{
  "status": "SUCCESS|FAILED",
  "modifiedFiles": [
    "src/main/java/com/.../Service.java",
    "pom.xml",
    "src/main/resources/application.yml"
  ],
  "validation": {
    "annotationPresent": true,
    "fallbackSignatureValid": true,
    "dependencyAdded": true,
    "configGenerated": true,
    "compilationSuccess": true
  }
}
```

---

**Version:** 2.0
**Last Updated:** 2025-11-21
