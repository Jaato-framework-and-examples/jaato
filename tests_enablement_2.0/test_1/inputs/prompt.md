# Generation Request: Customer Domain API

## Objetivo

Implementar la Domain API de Customer definida en `domain-api-spec.yaml`.

## Especificaciones

### Domain API (a generar)
- **Spec:** `domain-api-spec.yaml`
- **Descripción:** API REST normalizada para gestión de clientes

### System API (backend)
- **Spec:** `system-api-parties.yaml`
- **Descripción:** API que expone datos del mainframe (CICS/COBOL)
- **Base URL:** Variable de entorno `PARTIES_SYSTEM_API_URL`

### Mapping
- **Spec:** `mapping.json`
- **Descripción:** Mapeo de campos y transformaciones entre Domain y System API
- **Estado:** Validado por humano

## Requisitos No Funcionales

### Resilience

| Patrón | Configuración |
|--------|---------------|
| **Circuit Breaker** | Activar tras 5 fallos, esperar 30s antes de half-open |
| **Retry** | 3 intentos, backoff exponencial (100ms, 200ms, 400ms) |
| **Timeout** | 5 segundos máximo por llamada |

### Observabilidad

- Propagar `X-Correlation-ID` en todas las llamadas
- Logging estructurado (JSON)
- Health check endpoint

## Restricciones Técnicas

| Aspecto | Valor |
|---------|-------|
| Java version | 17 |
| Spring Boot | 3.2.x |
| Build tool | Maven |
| REST Client | RestClient (Spring 6.1+) |

## Output Esperado

Código Java/Spring Boot completo siguiendo arquitectura Hexagonal Light:
- Domain layer (entidades, puertos)
- Application layer (servicios)
- Adapter layer (REST controller, System API client, mappers)
- Infrastructure (configuración)
- Tests unitarios
