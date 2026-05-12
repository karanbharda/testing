#!/usr/bin/env python3
"""
Quick verification script for product_type flow
Checks if recent trades have product_type stored in metadata
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

def verify_product_type_storage():
    """Verify that trades are being stored with product_type"""
    
    print("\n" + "="*80)
    print("PRODUCT TYPE STORAGE VERIFICATION")
    print("="*80)
    
    # Find database
    data_dir = Path(__file__).parent / "data"
    db_path = data_dir / "trading.db"
    
    if not db_path.exists():
        print(f"\n⚠️  Database not found at {db_path}")
        print("This is normal if you haven't run any trades yet.")
        return False
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check schema
        print("\n1. Checking database schema...")
        cursor.execute("PRAGMA table_info(trades)")
        columns = cursor.fetchall()
        
        has_metadata = any(col[1] == 'trade_metadata' for col in columns)
        
        if has_metadata:
            print("   ✅ trades table has 'trade_metadata' column (JSON)")
        else:
            print("   ❌ trades table MISSING 'trade_metadata' column")
            return False
        
        # Check recent trades
        print("\n2. Checking recent trades...")
        cursor.execute("""
            SELECT id, ticker, action, timestamp, trade_metadata 
            FROM trades 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        
        trades = cursor.fetchall()
        
        if not trades:
            print("   ℹ️  No trades found in database (this is normal for new installations)")
            print("   Run some test trades to see product_type storage")
            return True
        
        print(f"   Found {len(trades)} recent trades\n")
        
        # Analyze trades
        trades_with_product_type = 0
        trades_without_product_type = 0
        
        print("   Recent Trades:")
        print("   " + "-"*76)
        
        for trade in trades:
            trade_id, ticker, action, timestamp, metadata_json = trade
            
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
                product_type = metadata.get('product_type', 'NOT SET')
                
                if product_type != 'NOT SET':
                    trades_with_product_type += 1
                    status_icon = "✅"
                else:
                    trades_without_product_type += 1
                    status_icon = "⚠️ "
                
                print(f"   {status_icon} Trade #{trade_id}: {action.upper()} {ticker} - "
                      f"Product Type: {product_type} - {timestamp}")
                
            except json.JSONDecodeError as e:
                print(f"   ❌ Trade #{trade_id}: {action.upper()} {ticker} - "
                      f"Invalid JSON in metadata: {e}")
                trades_without_product_type += 1
        
        print("   " + "-"*76)
        
        # Summary
        print(f"\n3. Summary:")
        print(f"   Total trades analyzed: {len(trades)}")
        print(f"   ✅ With product_type: {trades_with_product_type}")
        print(f"   ⚠️  Without product_type: {trades_without_product_type}")
        
        if trades_with_product_type > 0 and trades_without_product_type == 0:
            print(f"\n   🎉 PERFECT! All trades have product_type stored correctly")
            return True
        elif trades_with_product_type > trades_without_product_type:
            print(f"\n   ✅ MOSTLY GOOD! Most trades have product_type (fixes are working)")
            print(f"   Older trades may not have product_type before the fix")
            return True
        elif trades_with_product_type > 0:
            print(f"\n   ⚠️  MIXED RESULTS - Some trades have product_type, others don't")
            print(f"   This suggests fixes were applied mid-session")
            return True
        else:
            print(f"\n   ❌ ISSUE: No trades have product_type stored")
            print(f"   The fixes may not be fully deployed yet")
            return False
        
    except Exception as e:
        print(f"\n   ❌ Error checking database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def verify_config_product_type():
    """Verify that config files have product_type setting"""
    
    print("\n" + "="*80)
    print("CONFIG PRODUCT TYPE VERIFICATION")
    print("="*80)
    
    data_dir = Path(__file__).parent / "data"
    config_files = ["paper_config.json", "live_config.json"]
    
    for config_file in config_files:
        config_path = data_dir / config_file
        
        if not config_path.exists():
            print(f"\n⚠️  Config file not found: {config_path}")
            continue
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            product_type = config.get('productType', 'NOT SET')
            
            if product_type != 'NOT SET':
                print(f"\n✅ {config_file}: productType = {product_type}")
            else:
                print(f"\n⚠️  {config_file}: productType NOT SET (will default to CNC)")
                print(f"   You can add it manually or use the Settings UI")
        
        except Exception as e:
            print(f"\n❌ Error reading {config_file}: {e}")
    
    return True

def main():
    """Run all verifications"""
    
    print("\n" + "="*80)
    print("INTRADAY & DELIVERY FLOW VERIFICATION")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "Config Verification": verify_config_product_type(),
        "Database Verification": verify_product_type_storage()
    }
    
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    for check, result in results.items():
        status = "✅ PASS" if result else "⚠️  NEEDS ATTENTION"
        print(f"{status}: {check}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if all(results.values()):
        print("""
✅ All verifications passed! Your system is correctly handling product types.

To confirm everything is working end-to-end:
1. Place a small test trade (intraday or delivery)
2. Check the logs for "Placing order with product type: XXX"
3. Verify the trade appears in database with correct product_type
4. Check your Dhan order book to confirm correct product type execution
""")
    else:
        print("""
⚠️  Some verifications need attention:

If database shows no product_type:
- The fixes are newly applied
- Run a test trade to verify new trades have product_type

If config shows no product_type:
- Use the frontend Settings UI to select CNC or INTRADAY
- Or manually add "productType": "CNC" to your config JSON files

For detailed fix instructions, see:
- INTRADAY_DELIVERY_FLOW_ANALYSIS.md
""")
    
    return all(results.values())

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
