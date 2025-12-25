// Template: Request.java.tpl
// Output: {{basePackage}}/adapter/out/persistence/dto/{{Entity}}SystemApiRequest.java
// Purpose: Request DTO for System API communication
// STRUCTURE: MUST be record (per DETERMINISM-RULES.md)

package {{basePackage}}.adapter.out.persistence.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Request DTO for {{SystemApiName}} System API.
 * 
 * Used for create/update operations.
 * Uses record for immutability.
 * 
 * @generated {{skillId}} v{{skillVersion}}
 * @module mod-code-017-persistence-systemapi
 */
public record {{Entity}}SystemApiRequest(
{{#fields}}
    @JsonProperty("{{jsonField}}") {{type}} {{fieldName}}{{^last}},{{/last}}
{{/fields}}
) {}
