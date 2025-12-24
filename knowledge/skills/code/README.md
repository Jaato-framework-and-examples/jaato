# CODE Domain Skills

Skills for source code generation and transformation.

## Layer Structure

CODE domain skills are organized by architectural layer:

| Layer | Directory | Description |
|-------|-----------|-------------|
| **SoE** | `soe/` | System of Engagement - UI, frontend, digital channels |
| **SoI** | `soi/` | System of Integration - Microservices, APIs, orchestration |
| **SoR** | `sor/` | System of Record - Core systems, mainframe |

## Current Skills

### SoE (System of Engagement)
*No skills yet*

### SoI (System of Integration)
| Skill | Purpose | Flow |
|-------|---------|------|
| `skill-001-circuit-breaker-java-resilience4j` | Add circuit breaker pattern | ADD |
| `skill-020-microservice-java-spring` | Generate complete microservice | GENERATE |

### SoR (System of Record)
*No skills yet*

## Discovery

Skills are indexed in `runtime/discovery/skill-index.yaml` for efficient discovery.

The agent identifies the layer using signals (keywords, artifacts) and then queries the index to get filtered candidates.

## Adding New Skills

1. Create skill in appropriate layer directory: `skills/code/{layer}/skill-{NNN}-{name}/`
2. Follow `model/standards/authoring/SKILL.md` guide
3. Register in `runtime/discovery/skill-index.yaml`

---

See `model/domains/code/DOMAIN.md` for CODE domain specification.
