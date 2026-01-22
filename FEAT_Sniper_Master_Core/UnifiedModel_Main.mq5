//+------------------------------------------------------------------+
//|                                           UnifiedModel_Main.mq5 |
//|                             FEAT SNIPER - IMMORTAL CORE V5.1     |
//|                                  (c) 2026 Antigravity AI         |
//+------------------------------------------------------------------+
#property copyright "Antigravity AI"
#property version   "5.10"
#property strict

// --- INCLUDES ---
#include <Trade/Trade.mqh>
#include <UnifiedModel/CEMAs.mqh>
#include <UnifiedModel/CLiquidity.mqh>
#include <UnifiedModel/CFEAT.mqh>
#include <UnifiedModel/CInterop.mqh>
#include <UnifiedModel/CRiskManager.mqh>

// --- INPUTS ---
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
   
   // ZMQ Init
   if(!g_interop.Init(false, ZmqPort_SUB)) return(INIT_FAILED);
   if(!g_tx.Init(true, ZmqPort_PUB)) return(INIT_FAILED);

   EventSetMillisecondTimer(100);
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
   
   // Update FEAT Core
   g_feat.Calculate(PERIOD_CURRENT, tick.time, tick.bid, tick.bid, tick.ask, tick.bid, (double)tick.tick_volume);
   
   // Get Rich Data
   SFEATResult res = g_feat.GetResult();
   
   // "Hydrodynamic" JSON Payload
   string json = StringFormat(
      "{\"type\":\"TICK\",\"symbol\":\"%s\",\"bid\":%.5f,\"ask\":%.5f,\"time\":%llu,\"vol\":%llu,\"feat_score\":%.1f,\"accel\":%.2f,\"titanium\":\"%s\",\"zone\":\"%s\"}",
      _Symbol, tick.bid, tick.ask, tick.time_msc, tick.volume,
      res.compositeScore, res.accel.velocity, 
      (res.form.curvatureScore > 0.8 ? "TITANIUM_SUPPORT" : (res.form.curvatureScore < -0.8 ? "TITANIUM_RESISTANCE" : "NEUTRAL")),
      res.space.activeZoneType
   );
   
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
