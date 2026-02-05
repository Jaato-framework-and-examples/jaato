# Enablement 2.0 - Knowledge Base Model v3.0

## Overview

Enablement 2.0 is an AI-powered platform for automated software development. The Knowledge Base (KB) contains the structured knowledge that guides AI agents in generating, transforming, and maintaining code according to enterprise standards.

This document defines the **data model** - the entities, relationships, and rules that govern the KB.

### What's New in v3.0

- **Skills eliminated** as runtime entities (logic moved to Features)
- **Unified discovery** through capability-index only
- **Features enriched** with config, input_spec, and multi-implementation support
- **Stack detection** explicit with defaults and auto-detection
- **Two generic flows**: generate and transform

---

## Core Philosophy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DESIGN PRINCIPLES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. CAPABILITY-DRIVEN MODEL                                                 │
│     - Capabilities define WHAT features are available                       │
│     - Features define HOW to use each capability (config, input)            │
│     - Modules define HOW to implement each feature (templates)              │
│                                                                              │
│  2. SINGLE DISCOVERY PATH                                                   │
│     - All discovery goes through capability-index                           │
│     - No separate skill discovery                                           │
│     - prompt → capabilities → features → modules                            │
│                                                                              │
│  3. MULTI-IMPLEMENTATION SUPPORT                                            │
│     - One feature can have multiple implementations                         │
│     - Differentiated by stack (java-spring, nodejs, etc.)                   │
│     - Differentiated by pattern (annotation, client-config, etc.)           │
│                                                                              │
│  4. EXPLICIT STACK RESOLUTION                                               │
│     - Stack detected from existing code or prompt                           │
│     - Organizational defaults when not specified                            │
│     - Enables multi-technology support                                      │
│                                                                              │
│  5. SINGLE SOURCE OF TRUTH                                                  │
│     - capability-index.yaml is THE definitive source                        │
│     - Features contain all logic (no separate skills)                       │
│     - Modules contain templates for specific stack+pattern                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Entity Model

### Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENTITY RELATIONSHIPS (v3.0)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                           ┌──────────────┐                                  │
│                           │  CAPABILITY  │                                  │
│                           │              │                                  │
│                           │ - structural │                                  │
│                           │ - compositional                                 │
│                           └──────┬───────┘                                  │
│                                  │                                          │
│                               features                                      │
│                                  │                                          │
│                                  ▼                                          │
│                           ┌──────────────┐                                  │
│                           │   FEATURE    │                                  │
│                           │              │                                  │
│                           │ - keywords   │                                  │
│                           │ - requires   │                                  │
│                           │ - config     │                                  │
│                           │ - input_spec │                                  │
│                           └──────┬───────┘                                  │
│                                  │                                          │
│                           implementations                                   │
│                                  │                                          │
│          ┌───────────────────────┼───────────────────────┐                 │
│          │                       │                       │                  │
│          ▼                       ▼                       ▼                  │
│   ┌──────────────┐       ┌──────────────┐       ┌──────────────┐           │
│   │IMPLEMENTATION│       │IMPLEMENTATION│       │IMPLEMENTATION│           │
│   │              │       │              │       │              │           │
│   │ stack: A     │       │ stack: B     │       │ stack: A     │           │
│   │ pattern: X   │       │ pattern: X   │       │ pattern: Y   │           │
│   └──────┬───────┘       └──────┬───────┘       └──────┬───────┘           │
│          │                       │                       │                  │
│       module                  module                  module                │
│          │                       │                       │                  │
│          ▼                       ▼                       ▼                  │
│   ┌──────────────┐       ┌──────────────┐       ┌──────────────┐           │
│   │    MODULE    │       │    MODULE    │       │    MODULE    │           │
│   │              │       │              │       │              │           │
│   │ - templates  │       │ - templates  │       │ - templates  │           │
│   │ - rules      │       │ - rules      │       │ - rules      │           │
│   └──────────────┘       └──────────────┘       └──────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Comparison with v2.0

```
v2.0:  Skill → Capability → Feature → Module
v3.0:          Capability → Feature → Implementation → Module
                                          ↑
                                    (stack + pattern)
```

---

## Entity Definitions

### 1. CAPABILITY

A Capability is a **conceptual grouping of related features**. It represents a technical concern (architecture, persistence, resilience, etc.).

#### Capability Types (v2.2)

| Type | Description | Cardinality | Required for Generate | Transformable |
|------|-------------|-------------|----------------------|---------------|
| **foundational** | Base architecture, defines project structure | exactly-one | YES | NO |
| **layered** | Adds layers on top of foundational | multiple | NO | YES |
| **cross-cutting** | Decorators on existing code | multiple | NO | YES |

**Type Behaviors:**

- **FOUNDATIONAL:** Exactly one required for `flow-generate`. Cannot be added later. All layered capabilities require a foundational.
- **LAYERED:** Requires foundational to exist (auto-added if missing). Phase determined by `phase_group`.
- **CROSS-CUTTING:** Does NOT require foundational. Can apply to existing projects via `flow-transform`.

#### Phase Groups (v2.2)

| Phase Group | Phase | Description | Capabilities |
|-------------|-------|-------------|--------------|
| `structural` | 1 | Project structure, adapters IN | architecture, api-architecture |
| `implementation` | 2 | External connections, adapters OUT | integration, persistence |
| `cross-cutting` | 3+ | Decorators, annotations | resilience, distributed-transactions |

#### Capability Attributes (v2.2)

| Attribute | Required | Description |
|-----------|----------|-------------|
| `description` | Yes | Human-readable description |
| `type` | Yes | foundational, layered, or cross-cutting |
| `phase_group` | Yes | structural, implementation, or cross-cutting |
| `cardinality` | Yes | exactly-one or multiple |
| `transformable` | Yes | Whether this can be added to existing code |
| `documentation` | No | Path to capability documentation |
| `keywords` | **v2.2** | Keywords at capability level for generic matches |
| `default_feature` | **v2.2** | Feature to select when capability matches but no specific feature |

#### Capability Definition

```yaml
# In capability-index.yaml
capabilities:
  architecture:
    description: "Foundational architectural patterns"
    type: foundational
    phase_group: structural
    cardinality: exactly-one
    transformable: false

    keywords:
      - microservicio
      - microservice
      - service
      - application

    default_feature: hexagonal-light

    features:
      hexagonal-light:
        is_default: true
        # ...feature definition

  api-architecture:
    description: "API types according to Fusion model"
    type: layered
    phase_group: structural
    cardinality: multiple
    transformable: true

    keywords:
      - API
      - REST

    default_feature: standard  # NOT domain-api

    features:
      standard:
        is_default: true
        requires:
          - architecture  # Points to capability, not specific feature
        # ...feature definition
      domain-api:
        requires:
          - architecture
        # ...feature definition

  resilience:
    description: "Fault tolerance and resilience patterns"
    type: cross-cutting
    phase_group: cross-cutting
    cardinality: multiple
    transformable: true

    keywords:
      - resilience
      - fault tolerance

    # NO default_feature - user must specify which pattern
    # NO requires - can apply to existing code

    features:
      circuit-breaker:
        # ...feature definition
      retry:
        # ...feature definition
```

#### Capabilities in KB

Each capability has a documentation file in `model/domains/code/capabilities/`:

```
model/domains/code/capabilities/
├── architecture.md              # Foundational: base patterns
├── api_architecture.md          # Layered: API types (standard, Domain, System, etc.)
├── integration.md               # Layered: External system integration
├── persistence.md               # Layered: Data persistence
├── resilience.md                # Cross-cutting: Fault tolerance
└── distributed_transactions.md  # Cross-cutting: SAGA patterns
```

---

### 2. FEATURE

A Feature is a **specific option within a capability**. It contains all the logic needed to use that feature, including configuration, input specification, and multiple implementations.

#### Feature Definition

```yaml
# In capability-index.yaml
features:
  domain-api:
    description: "API de Dominio - expone capacidades de negocio"

    # Discovery keywords
    keywords:
      - Domain API
      - API de dominio
      - Fusion Domain

    # Dependencies on other features
    requires:
      - architecture.hexagonal-light

    # Mutual exclusions
    incompatible_with:
      - api-architecture.system-api  # Can't be both

    # Feature-specific configuration
    config:
      hateoas: true
      compensation_available: true

    # Input specification (what the user must provide)
    input_spec:
      serviceName:
        type: string
        required: true
        pattern: "^[a-z][a-z0-9-]*$"
      basePackage:
        type: string
        required: true
      entities:
        type: array
        required: true

    # Multiple implementations (by stack/pattern)
    implementations:
      - id: java-spring
        module: mod-code-019-api-public-exposure-java-spring
        stack: java-spring

    # Default implementation
    default: java-spring
```

#### Feature Attributes

| Attribute | Required | Description |
|-----------|----------|-------------|
| `description` | Yes | Human-readable description |
| `keywords` | Yes | Terms for discovery matching |
| `requires` | No | Dependencies on other features |
| `incompatible_with` | No | Mutually exclusive features |
| `config` | No | Feature-specific configuration |
| `input_spec` | No | Schema for user input |
| `implementations` | Yes | List of implementations |
| `default` | Yes* | Default implementation (*if multiple) |

---

### 3. IMPLEMENTATION

An Implementation is a **specific realization of a feature** for a particular stack and/or pattern.

#### Implementation Definition

```yaml
implementations:
  - id: java-spring-annotation
    module: mod-code-003-timeout-java-resilience4j
    stack: java-spring
    pattern: annotation
    description: "@TimeLimiter de Resilience4j"

  - id: java-spring-client-config
    module: mod-code-003b-timeout-restclient
    stack: java-spring
    pattern: client-config
    description: "Timeout en RestClient"
```

#### Implementation Attributes

| Attribute | Required | Description |
|-----------|----------|-------------|
| `id` | Yes | Unique identifier within feature |
| `module` | Yes | Module that implements this |
| `stack` | Yes | Technology stack |
| `pattern` | No | Implementation pattern (if alternatives exist) |
| `description` | No | Human-readable description |

#### Stack Values

Defined in capability-index.yaml:

```yaml
stacks:
  java-spring:
    description: "Java 17+ con Spring Boot 3.x"
    detection:
      - file: pom.xml
        contains: "spring-boot-starter"
  java-quarkus:
    description: "Java 17+ con Quarkus"
    detection:
      - file: pom.xml
        contains: "quarkus"
  nodejs:
    description: "Node.js"
    detection:
      - file: package.json
```

---

### 4. MODULE

A Module contains **templates and rules** for generating code for a specific feature + stack + pattern combination.

#### Module Definition

```yaml
# In MODULE.md frontmatter
---
id: mod-code-001-circuit-breaker-java-resilience4j
title: "Circuit Breaker - Java/Resilience4j"
version: 1.0
status: Active

# What this module implements
implements:
  capability: resilience
  feature: circuit-breaker
  stack: java-spring
  pattern: annotation

# Source reference
derived_from: eri-code-008-circuit-breaker-java-resilience4j
---
```

#### Module Structure

```
modules/
└── mod-code-001-circuit-breaker-java-resilience4j/
    ├── MODULE.md           # Documentation and rules
    ├── templates/          # Code templates
    │   ├── annotation/
    │   ├── config/
    │   └── test/
    └── validation/         # Validation scripts
```

---

### 5. CONFIG FLAGS (v3.0.11)

Config Flags enable **cross-module influence** without tight coupling. Feature modules can affect code generation in core modules through a publish/subscribe pattern.

#### The Problem

Feature modules (e.g., HATEOAS) need to influence core module outputs (e.g., Response.java) but:
- Modules should not directly reference each other
- Templates should remain in their owning module
- The relationship should be explicit and governable

#### The Solution: Pub/Sub Pattern

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CONFIG FLAGS FLOW                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   CAPABILITY-INDEX              CONTEXT                TEMPLATE      │
│   (publishes_flags)       (config_flags)          (conditional)     │
│                                                                      │
│   ┌──────────────┐        ┌──────────────┐       ┌──────────────┐   │
│   │  mod-019     │        │              │       │  mod-015     │   │
│   │  domain-api  │───────►│ hateoas:true │──────►│ Response.tpl │   │
│   │              │        │              │       │              │   │
│   │ publishes:   │        │              │       │ subscribes:  │   │
│   │  hateoas:true│        │              │       │  hateoas     │   │
│   └──────────────┘        └──────────────┘       └──────────────┘   │
│                                                                      │
│   Publisher                 Runtime                  Subscriber      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Publisher Syntax (capability-index.yaml)

Features declare flags they publish when activated:

```yaml
api-architecture:
  features:
    domain-api:
      module: mod-code-019-api-public-exposure-java-spring
      publishes_flags:
        hateoas: true
        pagination: true
```

#### Subscriber Syntax (MODULE.md)

Modules declare which flags affect their templates:

```yaml
# In MODULE.md
subscribes_to_flags:
  - flag: hateoas
    affects:
      - templates/application/dto/Response.java.tpl
    behavior: |
      When true:  Generate class extending RepresentationModel
      When false: Generate record (immutable DTO)
```

#### Template Conditional Syntax

Templates use Mustache conditionals on `config` object:

```java
{{#config.hateoas}}
public class {{Entity}}Response extends RepresentationModel<{{Entity}}Response> {
{{/config.hateoas}}
{{^config.hateoas}}
public record {{Entity}}Response(
{{/config.hateoas}}
```

#### Runtime Propagation (generation-context.json)

The Context Agent collects all `publishes_flags` from active modules:

```json
{
  "config_flags": {
    "hateoas": true,
    "pagination": true,
    "resilience": false
  }
}
```

#### Governance Benefits

| Benefit | Description |
|---------|-------------|
| **Visibility** | Query all pub/sub relationships across KB |
| **Impact Analysis** | "If I activate mod-019, what templates change?" |
| **Validation** | Detect orphan flags (published but not subscribed) |
| **Documentation** | Auto-generate dependency matrix |

#### Standard Flags Registry

| Flag | Publisher | Subscribers | Description |
|------|-----------|-------------|-------------|
| `hateoas` | mod-019 | mod-015 (Response.tpl) | HATEOAS hypermedia support |
| `pagination` | mod-019 | mod-015 (Controller.tpl) | Pagination support |
| `jpa` | mod-016 | mod-015 (Entity.tpl) | JPA annotations |
| `systemapi` | mod-017 | mod-015 (Repository.tpl) | System API persistence |
| `circuit-breaker` | mod-001 | mod-017 (Adapter.tpl) | Circuit breaker pattern |

---

## Discovery Flow

### Single Path Discovery

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DISCOVERY FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: prompt + context (existing code?)                                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: STACK RESOLUTION                                             │   │
│  │                                                                       │   │
│  │  1. Explicit in prompt? → "API en Quarkus" → java-quarkus           │   │
│  │  2. Detected from code? → pom.xml with spring → java-spring         │   │
│  │  3. Default? → defaults.stack (java-spring)                          │   │
│  │                                                                       │   │
│  │  Output: stack = "java-spring"                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: FEATURE MATCHING                                             │   │
│  │                                                                       │   │
│  │  prompt keywords → capability-index keywords → features              │   │
│  │                                                                       │   │
│  │  "API de dominio" → api-architecture.domain-api                      │   │
│  │  "circuit breaker" → resilience.circuit-breaker                      │   │
│  │  "System API backend" → persistence.systemapi                        │   │
│  │                                                                       │   │
│  │  Output: features = [domain-api, circuit-breaker, systemapi]         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: RESOLVE DEPENDENCIES                                         │   │
│  │                                                                       │   │
│  │  For each feature, resolve 'requires':                               │   │
│  │    domain-api.requires → architecture.hexagonal-light                │   │
│  │    systemapi.requires → integration.api-rest                         │   │
│  │                                                                       │   │
│  │  Output: all_features = [hexagonal-light, domain-api, api-rest,      │   │
│  │                          systemapi, circuit-breaker]                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: RESOLVE IMPLEMENTATIONS                                      │   │
│  │                                                                       │   │
│  │  For each feature, select implementation by stack:                   │   │
│  │    hexagonal-light + java-spring → mod-015                           │   │
│  │    domain-api + java-spring → mod-019                                │   │
│  │    api-rest + java-spring → mod-018                                  │   │
│  │    systemapi + java-spring → mod-017                                 │   │
│  │    circuit-breaker + java-spring → mod-001                           │   │
│  │                                                                       │   │
│  │  Output: modules = [mod-015, mod-019, mod-018, mod-017, mod-001]     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 5: DETERMINE FLOW                                               │   │
│  │                                                                       │   │
│  │  If no existing code → flow-generate                                 │   │
│  │  If existing code → flow-transform                                   │   │
│  │                                                                       │   │
│  │  Output: flow = "generate"                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  OUTPUT:                                                                    │
│    - flow: generate                                                         │
│    - stack: java-spring                                                     │
│    - features: [hexagonal-light, domain-api, ...]                          │
│    - modules: [mod-015, mod-019, ...]                                      │
│    - config: { hateoas: true, ... }                                        │
│    - input_spec: { serviceName, basePackage, entities }                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Flows

### Flow Types

| Flow | Purpose | Input | Output |
|------|---------|-------|--------|
| **flow-generate** | Create project from scratch | features, modules, input | New project |
| **flow-transform** | Modify existing project | features, modules, existing code | Modified code |

### Flow Selection

```python
def select_flow(context):
    if not context.has_existing_code:
        return "flow-generate"
    else:
        return "flow-transform"
```

### Phase Planning

Both flows use the Phase Planner to organize work:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE PLANNING                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Features are grouped by NATURE:                                            │
│                                                                              │
│  PHASE 1: STRUCTURAL                                                        │
│  └── architecture.hexagonal-light                                           │
│  └── api-architecture.domain-api                                            │
│      (Generates: project structure, domain model, ports, DTOs)              │
│                                                                              │
│  PHASE 2: IMPLEMENTATION                                                    │
│  └── persistence.systemapi                                                  │
│  └── integration.api-rest                                                   │
│      (Generates: adapters, clients)                                         │
│      (Modifies: application service)                                        │
│                                                                              │
│  PHASE 3+: CROSS-CUTTING                                                    │
│  └── resilience.circuit-breaker                                             │
│  └── resilience.retry                                                       │
│      (Modifies: adapters with annotations)                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Capability Index Structure

The `capability-index.yaml` is the single source of truth:

```yaml
version: "2.1"
domain: code
last_updated: "2026-01-20"

# Organizational defaults
defaults:
  stack: java-spring
  patterns:
    timeout: annotation

# Supported technology stacks
stacks:
  java-spring:
    description: "Java 17+ with Spring Boot 3.x"
    detection:
      - file: pom.xml
        contains: "spring-boot-starter"
  # ... more stacks

# Capability definitions
capabilities:
  architecture:
    type: structural
    features:
      hexagonal-light:
        keywords: [hexagonal, clean architecture, ports and adapters]
        implementations:
          - id: java-spring
            module: mod-code-015-hexagonal-base-java-spring
            stack: java-spring
        default: java-spring

  resilience:
    type: compositional
    keywords: [resilience, fault tolerance]
    features:
      circuit-breaker:
        keywords: [circuit breaker, breaker]
        implementations:
          - id: java-spring-resilience4j
            module: mod-code-001-circuit-breaker-java-resilience4j
            stack: java-spring
            pattern: annotation
        default: java-spring-resilience4j
      # ... more features

  # ... more capabilities
```

---

## Migration from v2.0

### What Changed

| v2.0 Entity | v3.0 Status | Migration |
|-------------|-------------|-----------|
| Skill (generation) | **Eliminated** | Logic moved to feature.config, feature.input_spec |
| Skill (transformation) | **Eliminated** | Use flow-transform + capability |
| skill-index.yaml | **Eliminated** | Only capability-index.yaml remains |
| Capability | Enhanced | Added keywords at capability level |
| Feature | **Enhanced** | Added config, input_spec, implementations, default |
| Module | Updated | Added stack, pattern in frontmatter |

### Skills → Features Mapping

| v2.0 Skill | v3.0 Feature |
|------------|--------------|
| skill-020-microservice | architecture.hexagonal-light |
| skill-021-api-rest-domain | api-architecture.domain-api |
| skill-021-api-rest-system | api-architecture.system-api |
| skill-040-add-resilience | flow-transform + resilience.* |
| skill-041-add-api-exposure | flow-transform + api-architecture.* |
| skill-042-add-persistence | flow-transform + persistence.* |

---

## Validation Rules

### Discovery Validation

1. **All matched features must have a valid implementation for the resolved stack**
2. **Required dependencies (requires) must be resolvable**
3. **Incompatible features cannot be selected together**

### Module Validation

1. **Module must declare stack in frontmatter**
2. **Module must reference valid capability.feature**
3. **Module templates must exist for declared patterns**

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL v3.0 SUMMARY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ENTITIES:                                                                  │
│    Capability → Feature → Implementation → Module                           │
│                                                                              │
│  DISCOVERY:                                                                 │
│    Single path through capability-index.yaml                                │
│    prompt → features → implementations → modules                            │
│                                                                              │
│  STACK:                                                                     │
│    Explicit resolution: prompt > detection > default                        │
│                                                                              │
│  FLOWS:                                                                     │
│    flow-generate (from scratch)                                             │
│    flow-transform (modify existing)                                         │
│                                                                              │
│  SOURCE OF TRUTH:                                                           │
│    capability-index.yaml                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
