package com.example.analytics;

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
    
    public void sendEvent(String eventName, String userId, String data) {
        log.debug("Sending analytics event: {} for user: {}", eventName, userId);
        
        // Direct call - if it fails, it's not critical
        analyticsClient.track(eventName, userId, data);
    }
}
