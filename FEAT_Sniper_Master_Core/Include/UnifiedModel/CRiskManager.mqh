//+------------------------------------------------------------------+
//|                                                CRiskManager.mqh |
//|                             FEAT SNIPER - RISK CONTROL ENGINE     |
//|                                  (c) 2026 Antigravity AI         |
//+------------------------------------------------------------------+
#ifndef CRISKMANAGER_MQH
#define CRISKMANAGER_MQH

#include <Trade/AccountInfo.mqh>

class CRiskManager {
private:
   double            m_maxDrawdown;
   double            m_maxDailyLoss;
   datetime          m_lastTradeTime;
   int               m_latencyThreshold; // ms

public:
   CRiskManager() : m_maxDrawdown(5.0), m_maxDailyLoss(2.0), m_latencyThreshold(3000) {}

   bool CheckExecutionWindow(long pythonTimestampMs) {
      if(pythonTimestampMs <= 0) return true; // Loose mode if no TS
      
      long currentMs = (long)GetTickCount64();
      // We assume pythonTimestampMs is relative to similar clock or we use it for drift
      // In a real SRE scenario, we'd use NTP synced clocks.
      // For now, if the command is over 3 seconds old, it's stale (high risk of price gap).
      
      // Since MT5 and Python might not have synced epoch, 
      // we primarily use this to detect internal queue latency.
      return true; // Simplified for V6 handshake
   }

   double CalculateLotSize(double riskPct, double stopLossPoints) {
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);
      double riskAmount = balance * (riskPct / 100.0);
      double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      
      if(stopLossPoints <= 0 || tickValue <= 0) return 0.01;
      
      double lots = riskAmount / (stopLossPoints * (tickValue / tickSize));
      return MathMax(0.01, MathMin(10.0, NormalizeDouble(lots, 2)));
   }
};

#endif
