package com.example.payment;

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
    
    public PaymentResult processPayment(String orderId, double amount) {
        log.info("Processing payment for order: {} amount: {}", orderId, amount);
        
        // Direct call to external service - no protection
        PaymentResult result = paymentClient.charge(orderId, amount);
        
        log.info("Payment processed successfully for order: {}", orderId);
        return result;
    }
}
