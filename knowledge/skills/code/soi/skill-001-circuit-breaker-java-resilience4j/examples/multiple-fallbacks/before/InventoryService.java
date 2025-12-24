package com.example.inventory;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class InventoryService {
    
    private static final Logger log = LoggerFactory.getLogger(InventoryService.class);
    
    private final InventoryClient primaryWarehouse;
    
    public InventoryService(InventoryClient primaryWarehouse) {
        this.primaryWarehouse = primaryWarehouse;
    }
    
    public StockLevel checkStock(String productId) {
        log.info("Checking stock for product: {}", productId);
        
        // Single point of failure - no alternative
        StockLevel stock = primaryWarehouse.getStock(productId);
        
        return stock;
    }
}
