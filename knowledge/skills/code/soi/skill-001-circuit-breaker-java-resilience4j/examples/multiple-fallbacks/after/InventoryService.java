package com.example.inventory;

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import java.util.Optional;

@Service
public class InventoryService {
    
    private static final Logger log = LoggerFactory.getLogger(InventoryService.class);
    
    private final InventoryClient primaryWarehouse;
    private final InventoryClient alternativeWarehouse;
    private final StockCache stockCache;
    
    public InventoryService(InventoryClient primaryWarehouse, 
                           InventoryClient alternativeWarehouse,
                           StockCache stockCache) {
        this.primaryWarehouse = primaryWarehouse;
        this.alternativeWarehouse = alternativeWarehouse;
        this.stockCache = stockCache;
    }
    
    @CircuitBreaker(name = "inventoryServiceCB", fallbackMethod = "checkStockFromAlternativeWarehouse")
    public StockLevel checkStock(String productId) {
        log.info("Checking stock for product: {}", productId);
        
        // Protected call to primary warehouse
        StockLevel stock = primaryWarehouse.getStock(productId);
        
        return stock;
    }
    
    /**
     * Primary fallback: Try alternative warehouse
     */
    private StockLevel checkStockFromAlternativeWarehouse(String productId, Throwable throwable) {
        log.warn("Primary warehouse failed, trying alternative: {}", throwable.getMessage());
        try {
            return alternativeWarehouse.getStock(productId);
        } catch (Exception e) {
            log.warn("Alternative warehouse also failed, using cached data");
            return checkStockFromCache(productId, e);
        }
    }
    
    /**
     * Secondary fallback: Return cached data
     */
    private StockLevel checkStockFromCache(String productId, Throwable throwable) {
        log.warn("Using cached stock data for product: {}", productId);
        Optional<StockLevel> cached = stockCache.get(productId);
        if (cached.isPresent()) {
            log.info("Returning cached stock level for product: {}", productId);
            return cached.get();
        }
        return checkStockDefault(productId, throwable);
    }
    
    /**
     * Tertiary fallback: Return default/unknown stock level
     */
    private StockLevel checkStockDefault(String productId, Throwable throwable) {
        log.error("All fallback strategies exhausted for product: {}, returning unknown", productId);
        return StockLevel.unknown(productId);
    }
}
