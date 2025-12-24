# Knowledge Base

> Pure knowledge assets: context for humans and agents

## Purpose

This folder contains the **foundational knowledge** that informs all automation:
- **ADRs** define strategic decisions and constraints
- **ERIs** provide tactical reference implementations

## Structure

```
knowledge/
├── ADRs/           # Architecture Decision Records
│   ├── adr-001-api-design-standards/
│   ├── adr-004-resilience-patterns/
│   └── ...
│
└── ERIs/           # Enterprise Reference Implementations
    ├── eri-code-001-hexagonal-light-java-spring/
    ├── eri-code-008-circuit-breaker-java-resilience4j/
    └── ...
```

## How Knowledge Flows

```
ADR (Strategic Decision)
 │
 └──> ERI (Reference Implementation)
       │
       └──> Informs MODULE creation (in /modules)
             │
             └──> Used by SKILL (in /skills)
```

## Current Inventory

### ADRs (5)
| ADR | Topic |
|-----|-------|
| adr-001 | API Design Standards |
| adr-004 | Resilience Patterns |
| adr-009 | Service Architecture (Hexagonal) |
| adr-011 | Persistence Patterns |
| adr-012 | API Integration Patterns |

### ERIs (7)
| ERI | Pattern |
|-----|---------|
| eri-code-001 | Hexagonal Light Java Spring |
| eri-code-008 | Circuit Breaker Resilience4j |
| eri-code-009 | Retry Resilience4j |
| eri-code-010 | Timeout Resilience4j |
| eri-code-011 | Rate Limiter Resilience4j |
| eri-code-012 | Persistence Patterns |
| eri-code-013 | API Integration REST |

## Related

- Model definition: `/model/`
- Executable skills: `/skills/`
- Reusable modules: `/modules/`
