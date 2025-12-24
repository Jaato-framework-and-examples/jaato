// Template: timeout-config.java.tpl
// Output: {{basePackage}}/infrastructure/config/RestClientConfig.java
// Purpose: Configure timeout at HTTP client level (DEFAULT variant)
// Variant: client-timeout
//
// This is the RECOMMENDED approach for timeout handling:
// - Simpler than @TimeLimiter (no CompletableFuture required)
// - Works with synchronous code
// - Configured at infrastructure level
//
// @generated {{skillId}} v{{skillVersion}}
// @module mod-code-003-timeout-java-resilience4j
// @variant client-timeout

package {{basePackage}}.infrastructure.config;

import org.apache.hc.client5.http.config.RequestConfig;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.util.Timeout;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.HttpComponentsClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

/**
 * REST client configuration with timeout settings.
 * 
 * Timeouts are configured at the HTTP client level, which is simpler
 * and more predictable than annotation-based approaches.
 * 
 * @generated {{skillId}} v{{skillVersion}}
 * @module mod-code-003-timeout-java-resilience4j
 * @variant client-timeout
 */
@Configuration
public class RestClientConfig {
    
    @Value("${integration.timeout.connect:5s}")
    private java.time.Duration connectTimeout;
    
    @Value("${integration.timeout.read:10s}")
    private java.time.Duration readTimeout;
    
    @Value("${integration.timeout.connection-request:5s}")
    private java.time.Duration connectionRequestTimeout;
    
    @Bean
    public RestClient.Builder restClientBuilder() {
        return RestClient.builder()
            .requestFactory(clientHttpRequestFactory());
    }
    
    @Bean
    public HttpComponentsClientHttpRequestFactory clientHttpRequestFactory() {
        HttpComponentsClientHttpRequestFactory factory = 
            new HttpComponentsClientHttpRequestFactory(httpClient());
        return factory;
    }
    
    @Bean
    public CloseableHttpClient httpClient() {
        RequestConfig requestConfig = RequestConfig.custom()
            .setConnectTimeout(Timeout.of(connectTimeout))
            .setResponseTimeout(Timeout.of(readTimeout))
            .setConnectionRequestTimeout(Timeout.of(connectionRequestTimeout))
            .build();
        
        return HttpClients.custom()
            .setDefaultRequestConfig(requestConfig)
            .build();
    }
}
