#!/usr/bin/env python3
"""
Test script to simulate a trade execution and verify real-time WebSocket updates
"""

import sys
import os
import json
from datetime import datetime

# Add the backend directory to the Python path
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from testindia import VirtualPortfolio

def test_trade_execution():
    """Test trade execution with WebSocket callbacks"""
    
    # Load configuration
    config = {
        "starting_balance": 10000,
        "mode": "paper",
        "dhan_client_id": None,
        "dhan_access_token": None
    }
    
    # Create portfolio instance (data is loaded automatically)
    portfolio = VirtualPortfolio(config)
    
    print("=== Current Portfolio State ===")
    print(f"Cash: Rs.{portfolio.cash:.2f}")
    print(f"Holdings: {portfolio.holdings}")
    print(f"Realized P&L: Rs.{portfolio.realized_pnl:.2f}")
    print(f"Unrealized P&L: Rs.{portfolio.unrealized_pnl:.2f}")
    
    # Add a test callback to see if it works
    def test_callback(trade_data):
        print(f"\n🔔 TRADE CALLBACK TRIGGERED:")
        print(f"   Action: {trade_data['action'].upper()}")
        print(f"   Asset: {trade_data['asset']}")
        print(f"   Quantity: {trade_data['qty']}")
        print(f"   Price: Rs.{trade_data['price']:.2f}")
        print(f"   Mode: {trade_data['mode']}")
        if 'realized_pnl' in trade_data:
            print(f"   Realized P&L: Rs.{trade_data['realized_pnl']:.2f}")
    
    portfolio.add_trade_callback(test_callback)
    
    # Test selling a small portion if we have holdings
    if portfolio.holdings:
        # Get the first stock in holdings
        ticker = list(portfolio.holdings.keys())[0]
        holding = portfolio.holdings[ticker]
        
        print(f"\n=== Testing SELL Transaction ===")
        print(f"Attempting to sell 0.5 units of {ticker}")
        print(f"Current holding: {holding['qty']} units at avg price Rs.{holding['avg_price']:.2f}")
        
        # Get current price (simulate)
        current_price = holding['avg_price'] * 1.02  # 2% higher than avg price
        
        # Execute sell order
        success = portfolio.sell(ticker, 0.5, current_price)
        
        if success:
            print(f"✅ Sell order executed successfully!")
            print(f"\n=== Updated Portfolio State ===")
            print(f"Cash: Rs.{portfolio.cash:.2f}")
            print(f"Holdings: {portfolio.holdings}")
            print(f"Realized P&L: Rs.{portfolio.realized_pnl:.2f}")
            print(f"Unrealized P&L: Rs.{portfolio.unrealized_pnl:.2f}")
        else:
            print(f"❌ Sell order failed!")
    else:
        print("\n=== Testing BUY Transaction ===")
        print("No holdings found, testing buy transaction instead")
        
        # Test buying a small amount
        ticker = "RELIANCE.NS"
        price = 2500.0
        qty = 1
        
        print(f"Attempting to buy {qty} units of {ticker} at Rs.{price:.2f}")
        
        success = portfolio.buy(ticker, qty, price)
        
        if success:
            print(f"✅ Buy order executed successfully!")
            print(f"\n=== Updated Portfolio State ===")
            print(f"Cash: Rs.{portfolio.cash:.2f}")
            print(f"Holdings: {portfolio.holdings}")
            print(f"Realized P&L: Rs.{portfolio.realized_pnl:.2f}")
            print(f"Unrealized P&L: Rs.{portfolio.unrealized_pnl:.2f}")
        else:
            print(f"❌ Buy order failed!")

if __name__ == "__main__":
    test_trade_execution()
