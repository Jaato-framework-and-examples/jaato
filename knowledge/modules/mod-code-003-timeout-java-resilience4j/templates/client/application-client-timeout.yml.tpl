# Template: application-client-timeout.yml.tpl
# Output: src/main/resources/application.yml (merge)
# Purpose: Timeout configuration for client-level timeout
# Variant: client-timeout
#
# @generated {{skillId}} v{{skillVersion}}
# @module mod-code-003-timeout-java-resilience4j
# @variant client-timeout

# ========== Client Timeout Configuration ==========
# These settings configure timeout at the HTTP client level.
# Simpler than @TimeLimiter, works with synchronous code.

integration:
  timeout:
    connect: {{connectTimeout}}s      # Time to establish connection
    read: {{readTimeout}}s            # Time to read response
    connection-request: {{connectionRequestTimeout}}s  # Time to get connection from pool

# Note: These timeouts apply to ALL REST clients using the configured RestClient.Builder.
# For per-client timeout overrides, configure separate RestClient instances.
