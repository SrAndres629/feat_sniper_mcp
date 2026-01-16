"""Quick MT5 diagnostic script."""
import MetaTrader5 as mt5

def main():
    print("=" * 60)
    print("FEAT SNIPER - MT5 DIAGNOSTIC")
    print("=" * 60)
    
    if not mt5.initialize():
        print(f"[FAIL] MT5 Init Failed: {mt5.last_error()}")
        return
    
    print("[OK] MT5 Initialized")
    
    acc = mt5.account_info()
    if acc:
        print(f"[OK] Account: {acc.login}")
        print(f"     Mode: {'DEMO' if acc.trade_mode == 1 else 'REAL'}")
        print(f"     Balance: ${acc.balance:.2f}")
        print(f"     Equity: ${acc.equity:.2f}")
        print(f"     Free Margin: ${acc.margin_free:.2f}")
    else:
        print(f"[FAIL] No account info: {mt5.last_error()}")
    
    # Check XAUUSD
    symbol = "XAUUSD"
    if mt5.symbol_select(symbol, True):
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            print(f"[OK] {symbol}: Bid={tick.bid:.2f} Ask={tick.ask:.2f}")
        else:
            print(f"[WARN] {symbol}: No tick data")
    else:
        print(f"[FAIL] {symbol}: Not available")
    
    print("=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    
    mt5.shutdown()

if __name__ == "__main__":
    main()
