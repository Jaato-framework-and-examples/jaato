// Template: FilterRequest.java.tpl
// Output: {{basePackagePath}}/adapter/in/rest/dto/{{entityName}}Filter.java
// Purpose: Filter criteria for {{entityName}} queries

package {{basePackage}}.adapter.in.rest.dto;

/**
 * Filter criteria for {{entityName}} queries.
 * Add domain-specific filter fields as needed.
 */
public record {{entityName}}Filter(
    String status,
    String country
) {
    public boolean hasStatus() {
        return status != null && !status.isBlank();
    }
    
    public boolean hasCountry() {
        return country != null && !country.isBlank();
    }
    
    /**
     * Returns true if any filter is active.
     */
    public boolean hasAnyFilter() {
        return hasStatus() || hasCountry();
    }
}
