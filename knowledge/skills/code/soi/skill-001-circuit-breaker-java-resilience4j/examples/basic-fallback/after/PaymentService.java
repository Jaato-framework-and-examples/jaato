package com.example.payment;

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class PaymentService {
    
    private static final Logger log = LoggerFactory.getLogger(PaymentService.class);
    
    private final PaymentClient paymentClient;
    
    public PaymentService(PaymentClient paymentClient) {
        this.paymentClient = paymentClient;
    }
    
    @CircuitBreaker(name = "paymentServiceCB", fallbackMethod = "processPaymentFallback")
    public PaymentResult processPayment(String orderId, double amount) {
        log.info("Processing payment for order: {} amount: {}", orderId, amount);
        
        // Protected call to external service
        PaymentResult result = paymentClient.charge(orderId, amount);
        
        log.info("Payment processed successfully for order: {}", orderId);
        return result;
    }
    
    /**
     * Fallback method for processPayment.
     * Called when circuit breaker is open or method fails.
     *
     * @param throwable The exception that triggered the fallback
     * @return Failed payment result with error message
     */
    private PaymentResult processPaymentFallback(String orderId, double amount, Throwable throwable) {
        log.warn("Circuit breaker fallback for processPayment: {}", throwable.getMessage());
        return PaymentResult.failed("Payment service temporarily unavailable. Please try again later.");
    }
}
