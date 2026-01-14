import asyncio
import logging
import unittest
from unittest.mock import MagicMock, patch

# Unit test targeting RiskEngine integration
from app.services.risk_engine import RiskEngine
from app.services.volatility_guard import volatility_guard
from app.services.spread_filter import spread_filter
from app.services.circuit_breaker import circuit_breaker

# Configure logging
logging.basicConfig(level=logging.ERROR)

class TestInstitutionalGuardsUnit(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.risk_engine = RiskEngine()

    @patch.object(circuit_breaker, 'get_lot_multiplier')
    @patch.object(spread_filter, 'is_spread_toxic')
    @patch.object(volatility_guard, 'can_trade')
    @patch('app.core.mt5_conn.mt5_conn.execute')
    async def test_volatility_veto_unit(self, mock_execute, mock_can_trade, mock_is_toxic, mock_get_mult):
        """Verify RiskEngine respects VolatilityGuard veto."""
        mock_can_trade.return_value = (False, "Flash Crash")
        mock_is_toxic.return_value = False
        mock_get_mult.return_value = 1.0 # Safe
        
        lot = await self.risk_engine.calculate_dynamic_lot(
            confidence=0.9, 
            volatility=0.5, 
            symbol="XAUUSD", 
            market_data={}
        )
        
        print(f"UNIT TEST: Volatility Veto -> Lot: {lot}")
        self.assertEqual(lot, 0.0)

    @patch.object(circuit_breaker, 'get_lot_multiplier')
    @patch.object(spread_filter, 'is_spread_toxic')
    @patch.object(volatility_guard, 'can_trade')
    @patch('app.core.mt5_conn.mt5_conn.execute')
    async def test_spread_veto_unit(self, mock_execute, mock_can_trade, mock_is_toxic, mock_get_mult):
        """Verify RiskEngine respects SpreadFilter veto."""
        mock_can_trade.return_value = (True, "Safe")
        mock_is_toxic.return_value = True
        mock_get_mult.return_value = 1.0 # Safe
        
        # Mock tick
        mock_tick = MagicMock()
        mock_tick.ask = 2000.10
        mock_tick.bid = 2000.05
        mock_execute.return_value = mock_tick

        lot = await self.risk_engine.calculate_dynamic_lot(
            confidence=0.9, 
            volatility=0.5, 
            symbol="XAUUSD", 
            market_data={}
        )
        
        print(f"UNIT TEST: Spread Veto -> Lot: {lot}")
        self.assertEqual(lot, 0.0)

    @patch.object(circuit_breaker, 'get_lot_multiplier')
    @patch.object(volatility_guard, 'can_trade')
    @patch.object(spread_filter, 'is_spread_toxic')
    @patch('app.core.mt5_conn.mt5_conn.execute')
    async def test_cb_veto_unit(self, mock_execute, mock_is_toxic, mock_can_trade, mock_get_mult):
        """Verify RiskEngine respects Circuit Breaker multiplier 0.0."""
        mock_can_trade.return_value = (True, "Safe")
        mock_is_toxic.return_value = False
        mock_get_mult.return_value = 0.0
        
        # Mock tick
        mock_tick = MagicMock()
        mock_tick.ask = 2000.10
        mock_tick.bid = 2000.05
        mock_execute.return_value = mock_tick

        lot = await self.risk_engine.calculate_dynamic_lot(
            confidence=0.9, 
            volatility=0.5, 
            symbol="XAUUSD", 
            market_data={}
        )
        
        print(f"UNIT TEST: CB Veto -> Lot: {lot}")
        self.assertEqual(lot, 0.0)

if __name__ == "__main__":
    unittest.main()
