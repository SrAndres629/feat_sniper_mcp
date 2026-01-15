//+------------------------------------------------------------------+
//|                                          UnifiedModel_Main.mq5   |
//|                        FEAT NEXUS: EXECUTION UNIT (The Hand)     |
//|                  Listens for ZMQ Commands -> Executes Trades     |
//+------------------------------------------------------------------+
#property copyright "FEAT Systems AI"
#property version   "2.00"
#property strict

#include <Trade/Trade.mqh>
#include <UnifiedModel/CInterop.mqh>

// --- INPUTS ---
input int      MagicNumber = 123456;
input double   MaxSlippage = 10;
input bool     Verbose     = true;

// --- GLOBALS ---
CTrade      g_trade;
CInterop    g_interop;
string      g_symbol;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   g_symbol = _Symbol;
   g_trade.SetExpertMagicNumber(MagicNumber);
   g_trade.SetDeviationInPoints((ulong)MaxSlippage);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC); 
   g_trade.SetAsyncMode(true); // Non-blocking execution
   
   // Init ZMQ Subscriber (Read from 5556)
   if(!g_interop.Init(false)) {
      Print("CRITICAL: ZMQ Execution Bridge Failed.");
      return(INIT_FAILED);
   }
   
   EventSetMillisecondTimer(100); // High-Frequency Poll
   Print("FEAT EXECUTION UNIT ONLINE. Waiting for commands...");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   g_interop.Shutdown();
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Ticks drive internal MQL5 events, but Logic is ZMQ driven.
}

//+------------------------------------------------------------------+
//| Timer function (The Nerve Loop)                                  |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Read pending commands from Python
   string json = g_interop.ReceiveHUD(); // Reusing the read function
   if(json != "") {
      ProcessCommand(json);
   }
}

//+------------------------------------------------------------------+
//| Command Processor                                                |
//+------------------------------------------------------------------+
void ProcessCommand(string json)
{
   string action = ExtractJsonValue(json, "action");
   
   // Ignore HUD updates
   if(action == "HUD_UPDATE" || action == "") return;
   
   Print("RECEIVED COMMAND: ", action);
   
   if(action == "BUY") {
      double vol = StringToDouble(ExtractJsonValue(json, "volume"));
      double sl  = StringToDouble(ExtractJsonValue(json, "sl"));
      double tp  = StringToDouble(ExtractJsonValue(json, "tp"));
      
      if(vol <= 0) vol = 0.01;
      g_trade.Buy(vol, g_symbol, 0, sl, tp, "FEAT_AI_BUY");
   }
   else if(action == "SELL") {
      double vol = StringToDouble(ExtractJsonValue(json, "volume"));
      double sl  = StringToDouble(ExtractJsonValue(json, "sl"));
      double tp  = StringToDouble(ExtractJsonValue(json, "tp"));
      
      if(vol <= 0) vol = 0.01;
      g_trade.Sell(vol, g_symbol, 0, sl, tp, "FEAT_AI_SELL");
   }
   else if(action == "CLOSE_ALL") {
      CloseAll();
   }
}

//+------------------------------------------------------------------+
//| Helper: Close All Positions                                      |
//+------------------------------------------------------------------+
void CloseAll()
{
   int total = PositionsTotal();
   for(int i=total-1; i>=0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(PositionGetInteger(POSITION_MAGIC) == MagicNumber) {
         g_trade.PositionClose(ticket);
      }
   }
}

//+------------------------------------------------------------------+
//| Helper: JSON Extractor                                           |
//+------------------------------------------------------------------+
string ExtractJsonValue(string json, string key) {
   string search = "\"" + key + "\"";
   int start = StringFind(json, search);
   if(start < 0) return("");
   
   int valStart = StringFind(json, ":", start);
   if(valStart < 0) return("");
   
   // Clean value start
   valStart++;
   while(valStart < StringLen(json) && 
         (StringGetCharacter(json, valStart) == ' ' || StringGetCharacter(json, valStart) == '"')) {
      valStart++;
   }
   
   // Find end
   int valEnd = valStart;
   bool inQuote = false;
   while(valEnd < StringLen(json)) {
      ushort c = StringGetCharacter(json, valEnd);
      if(c == ',' || c == '}') break;
      valEnd++;
   }
   
   // Trim quotes if string
   string res = StringSubstr(json, valStart, valEnd - valStart);
   StringReplace(res, "\"", "");
   return(res);
}
