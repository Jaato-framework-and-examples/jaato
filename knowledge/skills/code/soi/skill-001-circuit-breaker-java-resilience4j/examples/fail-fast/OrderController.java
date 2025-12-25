package com.example.order;

import io.github.resilience4j.circuitbreaker.CallNotPermittedException;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/orders")
public class OrderController {
    
    private final AnalyticsService analyticsService;
    
    public OrderController(AnalyticsService analyticsService) {
        this.analyticsService = analyticsService;
    }
    
    @PostMapping
    public OrderResponse createOrder(@RequestBody OrderRequest request) {
        // Business logic...
        Order order = processOrder(request);
        
        // Non-critical analytics - handle circuit breaker gracefully
        try {
            analyticsService.sendEvent("order_created", request.getUserId(), order.getId());
        } catch (CallNotPermittedException e) {
            // Circuit is open - skip analytics (non-critical)
            log.warn("Analytics circuit breaker is open, skipping event");
        } catch (Exception e) {
            // Other errors - log but don't fail order creation
            log.error("Failed to send analytics event: {}", e.getMessage());
        }
        
        return OrderResponse.from(order);
    }
}
