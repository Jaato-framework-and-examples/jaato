# Template: application-pagination.yml.tpl
# Output: src/main/resources/application-pagination.yml
# Purpose: Pagination configuration per ADR-001

# Pagination defaults per ADR-001
spring:
  data:
    web:
      pageable:
        default-page-size: {{defaultPageSize}}
        max-page-size: {{maxPageSize}}
        one-indexed-parameters: false  # Zero-based pagination
        page-parameter: page
        size-parameter: size
      sort:
        sort-parameter: sort

  # HATEOAS configuration
  hateoas:
    use-hal-as-default-json-media-type: true
