// Template: Enum.java.tpl
// Output: {{basePackage}}/domain/model/{{EnumName}}.java
// Purpose: Domain enum (pure Java enum, NO framework annotations)
// STRUCTURE: Simple enum WITHOUT attributes (per DETERMINISM-RULES.md)
// NOTE: Code mapping (e.g., to mainframe codes) belongs in Mapper class, NOT here

package {{basePackage}}.domain.model;

/**
 * {{EnumName}} domain enumeration.
 * 
 * This is a pure domain enum with NO external system concerns.
 * Mapping to/from external codes (mainframe, API) is handled
 * in the appropriate Mapper class.
 * 
 * @generated {{skillId}} v{{skillVersion}}
 * @module mod-code-015-hexagonal-base-java-spring
 */
public enum {{EnumName}} {
{{#enumValues}}
    {{value}}{{^last}},{{/last}}{{#last}}{{/last}}
{{/enumValues}}
}
