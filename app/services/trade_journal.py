"""
MODULO 4 FASE 14: Trade Journal
Registro detallado de cada trade para análisis post-mortem.

Datos capturados:
- Timestamp de entrada/salida
- Features al momento del trade
- Física del mercado
- Resultado y P&L
- Razón de entrada/salida
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("feat.trade_journal")

class TradeJournal:
    """
    Comprehensive trade journal for post-mortem analysis.
    Records all details of each trade for continuous improvement.
    """
    
    def __init__(self, journal_path: str = "trade_journal.json"):
        self.journal_path = Path(journal_path)
        self.entries: List[Dict[str, Any]] = []
        self.current_trade: Optional[Dict] = None
        self._load_existing()
        
    def _load_existing(self):
        """Load existing journal entries if file exists."""
        if self.journal_path.exists():
            try:
                with open(self.journal_path, 'r', encoding='utf-8') as f:
                    self.entries = json.load(f)
                logger.info(f"Loaded {len(self.entries)} journal entries")
            except Exception as e:
                logger.warning(f"Could not load journal: {e}")
                self.entries = []
    
    def _save(self):
        """Persist journal to disk."""
        try:
            with open(self.journal_path, 'w', encoding='utf-8') as f:
                json.dump(self.entries, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save journal: {e}")
    
    def open_trade(self, 
                   ticket: int,
                   symbol: str,
                   direction: str,
                   lot_size: float,
                   entry_price: float,
                   stop_loss: float,
                   take_profit: float,
                   features: List[float],
                   physics: Dict[str, Any],
                   entry_reason: str) -> Dict:
        """
        Record opening of a new trade.
        """
        self.current_trade = {
            "id": len(self.entries) + 1,
            "ticket": ticket,
            "symbol": symbol,
            "direction": direction,
            "lot_size": lot_size,
            "entry_time": datetime.now().isoformat(),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_features": features,
            "entry_physics": physics,
            "entry_reason": entry_reason,
            "status": "OPEN"
        }
        
        logger.info(f"📝 Trade #{self.current_trade['id']} opened: {direction} {symbol}")
        return self.current_trade
    
    def close_trade(self,
                    ticket: int,
                    exit_price: float,
                    exit_reason: str,
                    exit_features: Optional[List[float]] = None,
                    exit_physics: Optional[Dict] = None) -> Dict:
        """
        Record closing of a trade.
        """
        trade = self._find_trade(ticket)
        if not trade:
            logger.warning(f"Trade {ticket} not found in journal")
            return {}
        
        trade["exit_time"] = datetime.now().isoformat()
        trade["exit_price"] = exit_price
        trade["exit_reason"] = exit_reason
        trade["exit_features"] = exit_features
        trade["exit_physics"] = exit_physics
        trade["status"] = "CLOSED"
        
        # Calculate P&L
        if trade["direction"] == "BUY":
            pnl_pips = (exit_price - trade["entry_price"]) * 10000
        else:
            pnl_pips = (trade["entry_price"] - exit_price) * 10000
        
        trade["pnl_pips"] = round(pnl_pips, 1)
        trade["result"] = "WIN" if pnl_pips > 0 else "LOSS"
        
        # Calculate duration
        entry = datetime.fromisoformat(trade["entry_time"])
        exit_dt = datetime.fromisoformat(trade["exit_time"])
        trade["duration_minutes"] = round((exit_dt - entry).total_seconds() / 60, 1)
        
        self._save()
        logger.info(f"📝 Trade #{trade['id']} closed: {trade['result']} ({trade['pnl_pips']} pips)")
        
        return trade
    
    def _find_trade(self, ticket: int) -> Optional[Dict]:
        """Find a trade by ticket number."""
        for entry in self.entries:
            if entry.get("ticket") == ticket:
                return entry
        if self.current_trade and self.current_trade.get("ticket") == ticket:
            self.entries.append(self.current_trade)
            return self.current_trade
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get journal statistics."""
        closed = [e for e in self.entries if e.get("status") == "CLOSED"]
        
        if not closed:
            return {"trades": 0}
        
        wins = [e for e in closed if e.get("result") == "WIN"]
        total_pnl = sum(e.get("pnl_pips", 0) for e in closed)
        
        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(closed) - len(wins),
            "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
            "total_pnl_pips": round(total_pnl, 1),
            "avg_pnl_pips": round(total_pnl / len(closed), 1) if closed else 0,
            "avg_duration_min": round(sum(e.get("duration_minutes", 0) for e in closed) / len(closed), 1) if closed else 0
        }
    
    def get_exit_analysis(self) -> Dict[str, int]:
        """Analyze exit reasons."""
        closed = [e for e in self.entries if e.get("status") == "CLOSED"]
        reasons = {}
        for trade in closed:
            reason = trade.get("exit_reason", "UNKNOWN")
            reasons[reason] = reasons.get(reason, 0) + 1
        return reasons

# Singleton
trade_journal = TradeJournal()

def test_trade_journal():
    """Test the trade journal."""
    print("=" * 60)
    print("📓 FEAT SYSTEM - MODULE 4 PHASE 14: TRADE JOURNAL")
    print("=" * 60)
    
    tj = TradeJournal("test_journal.json")
    
    # Record a sample trade
    tj.open_trade(
        ticket=12345,
        symbol="XAUUSD",
        direction="BUY",
        lot_size=0.05,
        entry_price=1850.50,
        stop_loss=1848.00,
        take_profit=1855.00,
        features=[1850.5, 0.025, 1.2],
        physics={"regime": "LAMINAR", "l4_slope": 0.025},
        entry_reason="AI Signal: p_win=0.68"
    )
    
    # Close the trade
    tj.close_trade(
        ticket=12345,
        exit_price=1853.00,
        exit_reason="TP_HIT",
        exit_features=[1852.8, 0.02, 1.1],
        exit_physics={"regime": "LAMINAR"}
    )
    
    stats = tj.get_statistics()
    print(f"\n📊 Journal Statistics:")
    print(f"   Total Trades: {stats['total_trades']}")
    print(f"   Win Rate: {stats['win_rate']}%")
    print(f"   Total P&L: {stats['total_pnl_pips']} pips")
    
    # Cleanup test file
    Path("test_journal.json").unlink(missing_ok=True)

if __name__ == "__main__":
    test_trade_journal()
