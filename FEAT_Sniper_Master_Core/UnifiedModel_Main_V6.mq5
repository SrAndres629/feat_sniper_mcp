//+------------------------------------------------------------------+
//|                                        UnifiedModel_Main_V6.mq5 |
//|                             FEAT SNIPER - IMMORTAL CORE V6.0     |
//|                                  (c) 2026 Antigravity AI         |
//+------------------------------------------------------------------+
#property copyright "Antigravity AI"
#property version   "6.00"
#property strict

// --- INCLUDES ---
#include <Trade/Trade.mqh>
#include <UnifiedModel/CEMAs.mqh>
#include <UnifiedModel/CLiquidity.mqh>
#include <UnifiedModel/CFEAT.mqh>
#include <UnifiedModel/CInterop.mqh>
#include <UnifiedModel/CRiskManager.mqh>

// --- INPUTS ---
input string   InpVersion      = "6.0-DOCTORAL";
input int      ExpertMagicNumber = 123456;
input double   InitialLot        = 0.01;
input int      MaxSlippage       = 50;  // Points
input bool     Verbose           = false;
input int      ZmqPort_PUB       = 5556; // Sending Ticks to Python
input int      ZmqPort_SUB       = 5555; // Receiving Commands from Python

// --- GLOBALS ---
CTrade            g_trade;
CInterop          g_interop;
CInterop          g_tx; // Publisher
CEMAs             g_emas;
CLiquidity        g_liquidity;
CFEAT             g_feat;
string            g_symbol;

// --- CLASSES ---
class CExecutionUnit {
private:
   CRiskManager* m_risk;
   int           m_magic;
public:
   CExecutionUnit(CRiskManager* risk, int magic) : m_risk(risk), m_magic(magic) {}
   
   ulong Execute(string action, double vol, double sl, double tp, long timestamp, int magic, string comment) {
      if(!m_risk.CheckExecutionWindow(timestamp)) {
         Print("[EXECUTION] BLOCKED: Latency Timeout (>3s)");
         return 0;
      }
      
      MqlTradeRequest request;
      MqlTradeResult result;
      ZeroMemory(request);
      ZeroMemory(result);
      
      request.action = TRADE_ACTION_DEAL;
      request.symbol = _Symbol;
      request.volume = vol;
      request.type = (action == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
      request.price = (action == "BUY") ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
      request.sl = sl;
      request.tp = tp;
      request.deviation = 50;
      request.magic = magic;
      request.comment = comment;
      request.type_filling = ORDER_FILLING_IOC;
      
      int retries = 3;
      ulong resultTicket = 0;
      
      while(retries > 0) {
         if(OrderSend(request, result)) {
            if(result.retcode == TRADE_RETCODE_DONE) {
               resultTicket = result.order;
               Print("[EXECUTION] SUCCESS. Ticket: ", resultTicket);
               return resultTicket;
            }
         }
         
         // Error Handling
         int err = GetLastError();
         Print("[EXECUTION] FAIL: ", err, " Retcode: ", result.retcode);
         if(err == 4756) { // Trade Request Sending Failed
             Sleep(100);
             request.price = (action == "BUY") ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
             retries--;
         } else {
             // Fatal
             Print("[EXECUTION] FATAL ERROR. Aborting.");
             break;
         }
      }
      return 0;
   }
   
   void CloseAll() {
      int total = PositionsTotal();
      for(int i=total-1; i>=0; i--) {
         ulong ticket = PositionGetTicket(i);
         if(PositionGetInteger(POSITION_MAGIC) == m_magic) {
            g_trade.PositionClose(ticket);
         }
      }
   }
};

CRiskManager      RiskGuard;
CExecutionUnit    *Executor;

//+------------------------------------------------------------------+
//| INIT                                                             |
//+------------------------------------------------------------------+
int OnInit() {
   g_symbol = _Symbol;
   
   // Init Indicators
   g_emas.Init(_Symbol, PERIOD_CURRENT);
   g_liquidity.Init(_Symbol, PERIOD_CURRENT);
   g_feat.Init(_Symbol, PERIOD_CURRENT);
   g_feat.SetEMAs(&g_emas);
   g_feat.SetLiquidity(&g_liquidity);
   
   // Setup Trade Lib
   g_trade.SetExpertMagicNumber(ExpertMagicNumber);
   g_trade.SetDeviationInPoints((ulong)MaxSlippage);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);
   g_trade.SetAsyncMode(true);
   
   Executor = new CExecutionUnit(&RiskGuard, ExpertMagicNumber);
   
   // ZMQ Init (V6 Ports)
   if(!g_interop.Init(false, ZmqPort_SUB)) return(INIT_FAILED);
   if(!g_tx.Init(true, ZmqPort_PUB)) return(INIT_FAILED);

   EventSetMillisecondTimer(100);
   Print("[SNIPER V6] IMMORTAL CORE INITIALIZED. Neural Uplink: ACTIVE.");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| DEINIT                                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   EventKillTimer();
   g_interop.Shutdown();
   g_tx.Shutdown();
   delete Executor;
}

//+------------------------------------------------------------------+
//| TICK                                                             |
//+------------------------------------------------------------------+
void OnTick() {
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;
   
   // V6 Update: Send Full Market Microstructure for Python Tensor Processing
   // Python needs: OHLCV + Tick Components to build the 24-channel tensor.
   // Note: MT5 does not calculate HMA/Z-Score/Entropy/Killzones - Python does.
   
   // Basic Candles
   double open = iOpen(_Symbol, PERIOD_CURRENT, 0);
   double high = iHigh(_Symbol, PERIOD_CURRENT, 0);
   double low = iLow(_Symbol, PERIOD_CURRENT, 0);
   double close = iClose(_Symbol, PERIOD_CURRENT, 0);
   long volume = iVolume(_Symbol, PERIOD_CURRENT, 0);

   // JSON Payload (V6 Format)
   string json = "{";
   json += "\"type\":\"TICK\",";
   json += "\"symbol\":\"" + _Symbol + "\",";
   json += "\"time_msc\":" + IntegerToString(tick.time_msc) + ",";
   json += "\"bid\":" + DoubleToString(tick.bid, _Digits) + ",";
   json += "\"ask\":" + DoubleToString(tick.ask, _Digits) + ",";
   
   // Flow Dynamics
   json += "\"tick_vol\":" + IntegerToString(tick.tick_volume) + ",";
   json += "\"real_vol\":" + IntegerToString(tick.volume) + ",";
   
   // Candle Structure (Live)
   json += "\"open\":" + DoubleToString(open, _Digits) + ",";
   json += "\"high\":" + DoubleToString(high, _Digits) + ",";
   json += "\"low\":" + DoubleToString(low, _Digits) + ",";
   json += "\"close\":" + DoubleToString(close, _Digits);
   
   json += "}";
   
   g_tx.Send(json, true);
}

//+------------------------------------------------------------------+
//| TIMER (COMMAND LOOP)                                             |
//+------------------------------------------------------------------+
void OnTimer() {
   string json = g_interop.ReceiveHUD();
   if(json != "") ProcessCommand(json);
}

//+------------------------------------------------------------------+
//| COMMAND PROCESSING                                                |
//+------------------------------------------------------------------+
void ProcessCommand(string json) {
   string action = ExtractJsonValue(json, "action");
   if(action == "HUD_UPDATE" || action == "") return;
   
   string correlation_id = ExtractJsonValue(json, "correlation_id");
   ulong ticket = 0;
   string result = "OK";
   string error = "";
   
   if(action == "CLOSE") {
       ulong t = (ulong)StringToInteger(ExtractJsonValue(json, "ticket"));
       if(g_trade.PositionClose(t)) ticket = t;
       else { result = "ERROR"; error = "Close Failed"; }
   }
   else if(action == "BUY" || action == "SELL") {
      double vol = StringToDouble(ExtractJsonValue(json, "volume"));
      double sl = StringToDouble(ExtractJsonValue(json, "sl"));
      double tp = StringToDouble(ExtractJsonValue(json, "tp"));
      long ts = (long)StringToDouble(ExtractJsonValue(json, "ts"));
      int magic = (int)StringToInteger(ExtractJsonValue(json, "magic"));
      string comment = ExtractJsonValue(json, "comment");
      ticket = Executor.Execute(action, vol, sl, tp, ts, magic, comment);
      if(ticket == 0) { result = "ERROR"; error = "Exec Failed"; }
   }
   else if(action == "CLOSE_ALL") {
      Executor.CloseAll();
   }
   
   if(correlation_id != "") {
       string ack = StringFormat("{\"correlation_id\":\"%s\",\"result\":\"%s\",\"ticket\":%llu,\"error\":\"%s\"}", correlation_id, result, ticket, error);
       g_tx.Send(ack, true);
   }
}

//+------------------------------------------------------------------+
//| UTILS                                                            |
//+------------------------------------------------------------------+
string ExtractJsonValue(string json, string key) {
   string search = "\"" + key + "\"";
   int start = StringFind(json, search);
   if(start < 0) return("");
   int valStart = StringFind(json, ":", start) + 1;
   while(valStart < StringLen(json) && 
         (StringGetCharacter(json, valStart) == ' ' || StringGetCharacter(json, valStart) == '"')) valStart++;
   int valEnd = valStart;
   while(valEnd < StringLen(json)) {
       ushort c = StringGetCharacter(json, valEnd);
       if(c == ',' || c == '}' || c == '"') break;
       valEnd++;
   }
   return StringSubstr(json, valStart, valEnd - valStart);
}
