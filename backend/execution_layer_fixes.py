#!/usr/bin/env python3
"""
Execution Layer Fixes - Comprehensive Patch
Addresses all identified issues in live_executor.py

Issues Fixed:
1. Order status checking - Handle all Dhan API status codes properly
2. Trade recording - Only record after confirmed execution
3. Error handling - Add retry logic and better error recovery
4. Portfolio sync - Ensure proper synchronization after execution
5. Logging - Add comprehensive execution flow logging

Usage: Apply these patches to live_executor.py
"""

# ============================================================================
# FIX 1: Improved Order Status Checking (Lines ~728-775)
# ============================================================================

FIX_ORDER_STATUS_CHECKING = """
Replace lines 728-775 in execute_buy_order method with:

                try:
                    # Check if the order was immediately executed by checking the status
                    order_details = self.dhan_client.get_order_by_id(order_id)
                    order_status = order_details.get("orderStatus", "").upper() if order_details else "UNKNOWN"
                    
                    # Dhan API order statuses: PENDING, OPEN, TRADED, CANCELLED, REJECTED, MODIFYING, MODIFIED
                    # Also handle common variations
                    executed_statuses = ["TRADED", "FILLED", "COMPLETE"]
                    pending_statuses = ["PENDING", "OPEN", "TRIGGER PENDING"]
                    failed_statuses = ["CANCELLED", "REJECTED", "REJECTED BY EXCHANGE"]
                    
                    if order_status in executed_statuses:
                        # Order was immediately executed, record it in database
                        # Get actual execution price from order details if available
                        executed_price = float(order_details.get("price", current_price))
                        executed_qty = int(order_details.get("tradedQuantity", adjusted_quantity))
                        
                        logger.info(f"✅ Order EXECUTED: {order_status} | Price: {executed_price} | Qty: {executed_qty}")
                        
                        self.portfolio_manager.record_trade(
                            ticker=symbol,
                            action="buy",
                            quantity=executed_qty,
                            price=executed_price,
                            stop_loss=signal_data.get("stop_loss"),
                            take_profit=signal_data.get("take_profit"),
                            product_type=product_type
                        )
                        logger.info(
                            f"✅ Buy trade executed and recorded: {executed_qty} {symbol} at Rs.{executed_price:.2f}")

                        # Also update the portfolio after execution
                        self._update_portfolio_after_execution(
                            symbol, "BUY", executed_qty, executed_price
                        )
                    elif order_status in pending_statuses:
                        # Order is pending, don't record in database yet - wait for execution
                        logger.info(
                            f"⏳ Buy order placed but PENDING execution: {adjusted_quantity} {symbol} at Rs.{current_price:.2f}")
                        logger.info(f"   Order ID: {order_id}")
                        logger.info(f"   Current status: {order_status}")
                    elif order_status in failed_statuses:
                        # Order failed immediately
                        logger.error(
                            f"❌ Buy order FAILED: {order_status} | Order ID: {order_id}")
                        logger.error(f"   Symbol: {symbol} | Quantity: {adjusted_quantity}")
                        # Don't record in database - order failed
                    else:
                        # Unexpected status - log warning and don't record
                        logger.warning(
                            f"⚠️ Unexpected order status: {order_status}")
                        logger.info(f"   Order ID: {order_id}")
                        logger.info(f"   Full order details: {order_details}")
                        
                except Exception as db_error:
                    logger.error(
                        f"❌ Error checking immediate execution or recording trade: {db_error}")
                    logger.error(f"   Order ID: {order_id}")
                    import traceback
                    logger.error(f"   Traceback: {traceback.format_exc()}")
                    
                    # Don't auto-record on error - wait for proper confirmation
                    # This prevents recording failed orders
                    logger.warning(
                        f"⚠️ Skipping trade recording due to error - will sync on next check")
"""

# ============================================================================
# FIX 2: Apply Same Fix to execute_short_sell_order (Lines ~860-900)
# ============================================================================

FIX_SHORT_SELL_ORDER_STATUS = """
Apply similar fix to execute_short_sell_order method around lines 860-900:

                try:
                    order_details = self.dhan_client.get_order_by_id(order_id)
                    order_status = order_details.get("orderStatus", "").upper() if order_details else "UNKNOWN"
                    
                    executed_statuses = ["TRADED", "FILLED", "COMPLETE"]
                    pending_statuses = ["PENDING", "OPEN", "TRIGGER PENDING"]
                    failed_statuses = ["CANCELLED", "REJECTED", "REJECTED BY EXCHANGE"]
                    
                    if order_status in executed_statuses:
                        executed_price = float(order_details.get("price", current_price))
                        executed_qty = int(order_details.get("tradedQuantity", adjusted_quantity))
                        
                        logger.info(f"✅ Short sell EXECUTED: {order_status} | Price: {executed_price} | Qty: {executed_qty}")
                        
                        self.portfolio_manager.record_trade(
                            ticker=symbol,
                            action="short_sell",
                            quantity=executed_qty,
                            price=executed_price,
                            stop_loss=signal_data.get("stop_loss"),
                            take_profit=signal_data.get("take_profit"),
                            product_type="INTRADAY",
                            is_short_sell=True
                        )
                        logger.info(f"✅ Short sell executed: {executed_qty} {symbol} at Rs.{executed_price:.2f}")
                        
                        self._update_portfolio_after_execution(
                            symbol, "SELL", executed_qty, executed_price, is_short_sell=True
                        )
                    elif order_status in pending_statuses:
                        logger.info(f"⏳ Short sell order PENDING: {adjusted_quantity} {symbol} at Rs.{current_price:.2f}")
                        logger.info(f"   Order ID: {order_id} | Status: {order_status}")
                    elif order_status in failed_statuses:
                        logger.error(f"❌ Short sell order FAILED: {order_status} | Order ID: {order_id}")
                    else:
                        logger.warning(f"⚠️ Unexpected short sell order status: {order_status}")
                        logger.info(f"   Order details: {order_details}")
                        
                except Exception as db_error:
                    logger.error(f"❌ Error checking short sell execution: {db_error}")
                    logger.warning(f"⚠️ Skipping trade recording - will sync on next check")
"""

# ============================================================================
# FIX 3: Apply Same Fix to execute_buy_to_cover_order (Lines ~975-1020)
# ============================================================================

FIX_BUY_TO_COVER_ORDER_STATUS = """
Apply similar fix to execute_buy_to_cover_order method around lines 975-1020:

                try:
                    order_details = self.dhan_client.get_order_by_id(order_id)
                    order_status = order_details.get("orderStatus", "").upper() if order_details else "UNKNOWN"
                    
                    executed_statuses = ["TRADED", "FILLED", "COMPLETE"]
                    pending_statuses = ["PENDING", "OPEN", "TRIGGER PENDING"]
                    failed_statuses = ["CANCELLED", "REJECTED", "REJECTED BY EXCHANGE"]
                    
                    if order_status in executed_statuses:
                        executed_price = float(order_details.get("price", price))
                        executed_qty = int(order_details.get("tradedQuantity", quantity))
                        
                        logger.info(f"✅ Buy-to-cover EXECUTED: {order_status} | Price: {executed_price} | Qty: {executed_qty}")
                        
                        self.portfolio_manager.record_trade(
                            ticker=symbol,
                            action="buy_to_cover",
                            quantity=executed_qty,
                            price=executed_price,
                            pnl=signal_data.get("pnl", 0),
                            product_type="INTRADAY",
                            is_cover_short=True
                        )
                        logger.info(f"✅ Buy-to-cover executed: {executed_qty} {symbol} at Rs.{executed_price:.2f}")
                        
                        self._update_portfolio_after_execution(
                            symbol, "BUY", executed_qty, executed_price, is_cover_short=True
                        )
                    elif order_status in pending_statuses:
                        logger.info(f"⏳ Buy-to-cover order PENDING: {quantity} {symbol} at Rs.{price:.2f}")
                        logger.info(f"   Order ID: {order_id} | Status: {order_status}")
                    elif order_status in failed_statuses:
                        logger.error(f"❌ Buy-to-cover order FAILED: {order_status} | Order ID: {order_id}")
                    else:
                        logger.warning(f"⚠️ Unexpected buy-to-cover order status: {order_status}")
                        
                except Exception as db_error:
                    logger.error(f"❌ Error checking buy-to-cover execution: {db_error}")
                    logger.warning(f"⚠️ Skipping trade recording - will sync on next check")
"""

# ============================================================================
# FIX 4: Apply Same Fix to execute_sell_order (Lines ~1135-1185)
# ============================================================================

FIX_SELL_ORDER_STATUS = """
Apply similar fix to execute_sell_order method around lines 1135-1185:

                    try:
                        order_details = self.dhan_client.get_order_by_id(order_id)
                        order_status = order_details.get("orderStatus", "").upper() if order_details else "UNKNOWN"
                        
                        executed_statuses = ["TRADED", "FILLED", "COMPLETE"]
                        pending_statuses = ["PENDING", "OPEN", "TRIGGER PENDING"]
                        failed_statuses = ["CANCELLED", "REJECTED", "REJECTED BY EXCHANGE"]
                        
                        if order_status in executed_statuses:
                            executed_price = float(order_details.get("price", current_price))
                            executed_qty = int(order_details.get("tradedQuantity", holding.quantity))
                            
                            logger.info(f"✅ Sell EXECUTED: {order_status} | Price: {executed_price} | Qty: {executed_qty}")
                            
                            self.portfolio_manager.record_trade(
                                ticker=symbol,
                                action="sell",
                                quantity=executed_qty,
                                price=executed_price,
                                pnl=pnl,
                                stop_loss=signal_data.get("stop_loss"),
                                take_profit=signal_data.get("take_profit"),
                                product_type=product_type
                            )
                            logger.info(f"✅ Sell trade executed: {executed_qty} {symbol} at Rs.{executed_price:.2f}")
                            
                            self._update_portfolio_after_execution(
                                symbol, "SELL", executed_qty, executed_price
                            )
                        elif order_status in pending_statuses:
                            logger.info(f"⏳ Sell order PENDING: {holding.quantity} {symbol} at Rs.{current_price:.2f}")
                            logger.info(f"   Order ID: {order_id} | Status: {order_status}")
                        elif order_status in failed_statuses:
                            logger.error(f"❌ Sell order FAILED: {order_status} | Order ID: {order_id}")
                        else:
                            logger.warning(f"⚠️ Unexpected sell order status: {order_status}")
                            
                    except Exception as db_error:
                        logger.error(f"❌ Error checking sell execution: {db_error}")
                        logger.warning(f"⚠️ Skipping trade recording - will sync on next check")
"""

# ============================================================================
# FIX 5: Enhanced check_and_update_orders with Better Status Handling
# ============================================================================

FIX_CHECK_AND_UPDATE_ORDERS = """
Replace the check_and_update_orders method (around lines 1210-1276) with:

    def check_and_update_orders(self) -> List[Dict]:
        \"\"\"Check status of pending orders and update portfolio with comprehensive status handling\"\"\"
        updated_orders = []

        try:
            # Get all orders from Dhan
            all_orders = self.dhan_client.get_orders()
            
            # Define status categories
            executed_statuses = ["TRADED", "FILLED", "COMPLETE"]
            pending_statuses = ["PENDING", "OPEN", "TRIGGER PENDING"]
            failed_statuses = ["CANCELLED", "REJECTED", "REJECTED BY EXCHANGE"]

            for order_id, pending_order in list(self.pending_orders.items()):
                # Find matching order in Dhan response
                dhan_order = next(
                    (o for o in all_orders if o.get("orderId") == order_id), None)

                if dhan_order:
                    order_status = dhan_order.get("orderStatus", "").upper()
                    logger.info(f"Checking order {order_id}: Status={order_status}")

                    if order_status in executed_statuses:
                        # Order executed successfully
                        executed_price = float(dhan_order.get("price", pending_order["price"]))
                        executed_qty = int(dhan_order.get("tradedQuantity", pending_order["quantity"]))

                        logger.info(f"✅ Order EXECUTED: {order_id} | {pending_order['side']} {executed_qty} {pending_order['symbol']} at Rs.{executed_price:.2f}")

                        # Update portfolio with execution details
                        self._update_portfolio_after_execution(
                            pending_order["symbol"],
                            pending_order["side"],
                            executed_qty,
                            executed_price,
                            is_short_sell=pending_order.get("is_short_sell", False),
                            is_cover_short=pending_order.get("is_cover_short", False)
                        )

                        # Record trade in database
                        try:
                            signal_data = pending_order.get("signal_data", {})
                            self.portfolio_manager.record_trade(
                                ticker=pending_order["symbol"],
                                action=pending_order["side"].lower(),
                                quantity=executed_qty,
                                price=executed_price,
                                stop_loss=signal_data.get("stop_loss"),
                                take_profit=signal_data.get("take_profit"),
                                product_type=signal_data.get("product_type", "CNC"),
                                is_short_sell=pending_order.get("is_short_sell", False),
                                is_cover_short=pending_order.get("is_cover_short", False)
                            )
                            logger.info(f"✅ Trade recorded in database: {order_id}")
                        except Exception as record_error:
                            logger.error(f"❌ Error recording trade: {record_error}")

                        # Move to executed orders
                        self.executed_orders[order_id] = {
                            **pending_order,
                            "executed_price": executed_price,
                            "executed_quantity": executed_qty,
                            "execution_time": datetime.now().isoformat(),
                            "status": "EXECUTED"
                        }

                        updated_orders.append(self.executed_orders[order_id])
                        del self.pending_orders[order_id]

                    elif order_status in failed_statuses:
                        # Order failed
                        logger.warning(f"❌ Order {order_status}: {order_id} | {pending_order['side']} {pending_order['symbol']}")
                        
                        self.executed_orders[order_id] = {
                            **pending_order,
                            "status": order_status,
                            "failure_time": datetime.now().isoformat(),
                            "failure_reason": dhan_order.get("message", "Unknown reason")
                        }

                        updated_orders.append(self.executed_orders[order_id])
                        del self.pending_orders[order_id]

                    elif order_status in pending_statuses:
                        # Still pending - just log
                        logger.debug(f"⏳ Order still pending: {order_id} | Status: {order_status}")

                    else:
                        # Unknown status - log warning
                        logger.warning(f"⚠️ Unknown order status: {order_status} | Order ID: {order_id}")
                        logger.debug(f"   Full order data: {dhan_order}")

            return updated_orders

        except Exception as e:
            logger.error(f"❌ Failed to check order status: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return []
"""

# ============================================================================
# FIX 6: Add Order Retry Logic
# ============================================================================

ADD_RETRY_LOGIC = """
Add this new method to LiveTradingExecutor class (after check_and_update_orders):

    def retry_failed_orders(self, max_retries: int = 2) -> List[Dict]:
        \"\"\"Retry orders that failed due to temporary issues
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of retry results
        \"\"\"
        retry_results = []
        
        # Get failed orders from last hour
        one_hour_ago = datetime.now().timestamp() - 3600
        
        failed_orders_to_retry = []
        for order_id, order_data in self.executed_orders.items():
            if order_data.get("status") in ["CANCELLED", "REJECTED"]:
                failure_time_str = order_data.get("failure_time", "")
                if failure_time_str:
                    try:
                        failure_time = datetime.fromisoformat(failure_time_str).timestamp()
                        if failure_time > one_hour_ago:
                            failed_orders_to_retry.append((order_id, order_data))
                    except:
                        pass
        
        if not failed_orders_to_retry:
            logger.info("No recent failed orders to retry")
            return retry_results
        
        logger.info(f"Found {len(failed_orders_to_retry)} failed orders to retry")
        
        for order_id, order_data in failed_orders_to_retry:
            try:
                retry_count = order_data.get("retry_count", 0)
                if retry_count >= max_retries:
                    logger.info(f"Skipping {order_id} - max retries ({max_retries}) reached")
                    continue
                
                logger.info(f"Retrying order {order_id} (attempt {retry_count + 1}/{max_retries})")
                
                symbol = order_data["symbol"]
                side = order_data["side"]
                quantity = order_data["quantity"]
                signal_data = order_data.get("signal_data", {})
                
                # Re-execute based on side
                if side == "BUY":
                    result = self.execute_buy_order(symbol, signal_data)
                elif side == "SELL":
                    result = self.execute_sell_order(symbol, signal_data)
                else:
                    logger.warning(f"Unknown side for retry: {side}")
                    continue
                
                result["original_order_id"] = order_id
                result["retry_attempt"] = retry_count + 1
                retry_results.append(result)
                
                if result.get("success"):
                    # Update retry count
                    order_data["retry_count"] = retry_count + 1
                    logger.info(f"✅ Retry successful for {order_id}")
                else:
                    logger.warning(f"❌ Retry failed for {order_id}: {result.get('message')}")
                    
            except Exception as e:
                logger.error(f"❌ Error retrying order {order_id}: {e}")
        
        return retry_results
"""

# ============================================================================
# FIX 7: Enhanced Portfolio Sync After Execution
# ============================================================================

ENHANCED_PORTFOLIO_SYNC = """
Add this method to improve portfolio synchronization (after _update_portfolio_after_execution):

    def force_portfolio_sync(self) -> Dict:
        \"\"\"Force sync portfolio with Dhan holdings after execution
        
        This should be called periodically to ensure database matches broker
        \"\"\"
        logger.info("🔄 Forcing portfolio sync with Dhan...")
        
        try:
            # Get holdings from Dhan
            dhan_holdings = self.dhan_client.get_holdings()
            
            if not dhan_holdings:
                logger.warning("No holdings data from Dhan")
                return {"success": False, "message": "No holdings data from Dhan"}
            
            sync_results = {
                "success": True,
                "synced_holdings": 0,
                "discrepancies": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Sync each holding
            for dhan_holding in dhan_holdings:
                symbol = dhan_holding.get("tradingSymbol", "")
                dhan_qty = int(dhan_holding.get("quantity", 0))
                dhan_avg_price = float(dhan_holding.get("averagePrice", 0))
                
                if dhan_qty == 0:
                    continue
                
                # Check against database
                db_holding = self.portfolio_manager.get_holding(symbol)
                
                if not db_holding or db_holding.quantity != dhan_qty:
                    discrepancy = {
                        "symbol": symbol,
                        "dhan_quantity": dhan_qty,
                        "db_quantity": db_holding.quantity if db_holding else 0,
                        "dhan_avg_price": dhan_avg_price,
                        "db_avg_price": db_holding.avg_price if db_holding else 0
                    }
                    sync_results["discrepancies"].append(discrepancy)
                    logger.warning(f"⚠️ Discrepancy found: {discrepancy}")
                    
                    # Auto-fix discrepancy
                    try:
                        self.portfolio_manager.sync_holding(
                            symbol=symbol,
                            quantity=dhan_qty,
                            avg_price=dhan_avg_price
                        )
                        sync_results["synced_holdings"] += 1
                        logger.info(f"✅ Synced {symbol}: qty={dhan_qty}, price={dhan_avg_price}")
                    except Exception as sync_error:
                        logger.error(f"❌ Failed to sync {symbol}: {sync_error}")
            
            logger.info(f"✅ Portfolio sync complete: {sync_results['synced_holdings']} holdings synced, {len(sync_results['discrepancies'])} discrepancies")
            return sync_results
            
        except Exception as e:
            logger.error(f"❌ Portfolio sync failed: {e}")
            return {"success": False, "message": str(e)}
"""

# ============================================================================
# SUMMARY OF ALL FIXES
# ============================================================================

FIXES_SUMMARY = """
EXECUTION LAYER FIXES SUMMARY
==============================

✅ Fix 1: Order Status Checking (execute_buy_order)
   - Handle multiple Dhan status codes: TRADED, FILLED, COMPLETE, PENDING, OPEN, CANCELLED, REJECTED
   - Use actual execution price from Dhan instead of requested price
   - Only record trades after confirmed execution
   - Better error handling with traceback logging

✅ Fix 2: Order Status Checking (execute_short_sell_order)  
   - Same improvements as Fix 1 for short selling

✅ Fix 3: Order Status Checking (execute_buy_to_cover_order)
   - Same improvements as Fix 1 for buy-to-cover

✅ Fix 4: Order Status Checking (execute_sell_order)
   - Same improvements as Fix 1 for selling

✅ Fix 5: Enhanced check_and_update_orders
   - Comprehensive status category handling
   - Record trades in database only after execution confirmation
   - Better logging with order IDs and status
   - Capture failure reasons from Dhan

✅ Fix 6: Add retry_failed_orders method
   - Retry orders that failed due to temporary issues
   - Configurable max retries (default: 2)
   - Only retry recent failures (last hour)
   - Track retry attempts

✅ Fix 7: Add force_portfolio_sync method
   - Periodic synchronization with Dhan holdings
   - Auto-fix discrepancies between database and broker
   - Detailed sync report with discrepancies

BENEFITS:
=========
1. No more premature trade recording
2. Accurate execution prices from broker
3. Better error handling and recovery
4. Comprehensive logging for debugging
5. Automatic retry for temporary failures
6. Portfolio stays in sync with broker
7. Support for all Dhan order statuses

TESTING:
========
1. Test buy order execution and recording
2. Test sell order execution and recording  
3. Test order status transitions (PENDING -> TRADED)
4. Test failed order handling (REJECTED, CANCELLED)
5. Test portfolio sync after multiple executions
6. Test retry logic with simulated failures
"""

if __name__ == "__main__":
    print(FIXES_SUMMARY)
    print("\n📝 Apply these fixes to: backend/live_executor.py")
    print("🔍 Search for the method names mentioned in each fix")
    print("🛠️  Replace the existing code with the improved versions")
