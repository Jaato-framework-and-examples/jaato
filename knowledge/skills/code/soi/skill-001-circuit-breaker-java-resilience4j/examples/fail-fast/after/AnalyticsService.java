package com.example.analytics;

import io.github.resilience4j.circuitbreaker.CallNotPermittedException;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class AnalyticsService {
    
    private static final Logger log = LoggerFactory.getLogger(AnalyticsService.class);
    
    private final AnalyticsClient analyticsClient;
    
    public AnalyticsService(AnalyticsClient analyticsClient) {
        this.analyticsClient = analyticsClient;
    }
    
    @CircuitBreaker(name = "analyticsServiceCB")
    public void sendEvent(String eventName, String userId, String data) throws Exception {
        log.debug("Sending analytics event: {} for user: {}", eventName, userId);
        
        // Protected call - fails fast when circuit is open
        analyticsClient.track(eventName, userId, data);
    }
}
