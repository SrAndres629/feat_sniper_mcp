//+------------------------------------------------------------------+
//|                                              FEAT_Visualizer.mq5 |
//|                                  Copyright 2024, FEAT Systems AI |
//|                                             https://featsystems.ai |
//+------------------------------------------------------------------+
#property copyright "FEAT Systems AI"
#property link      "https://featsystems.ai"
#property version   "5.00"
#property indicator_chart_window
#property indicator_buffers 6
#property indicator_plots   4

//--- Plot 1: Clouds (Fills)
#property indicator_label1  "CloudUpper"
#property indicator_type1   DRAW_FILLING
#property indicator_color1  clrMediumSeaGreen, clrCrimson
//--- Plot 2: FEAT Score Line (Purple)
#property indicator_label2  "FEAT Score"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrMediumPurple
#property indicator_style2  STYLE_SOLID
#property indicator_width2  2
//--- Plot 3: Projection High
#property indicator_label3  "ProjHigh"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrDeepSkyBlue
#property indicator_style3  STYLE_DOT
//--- Plot 4: Projection Low
#property indicator_label4  "ProjLow"
#property indicator_type4   DRAW_LINE
#property indicator_color4  clrOrangeRed
#property indicator_style4  STYLE_DOT

// Includes
#include <UnifiedModel/CInterop.mqh> // ZMQ Logic
#include <Canvas/Canvas.mqh>

//--- Buffers
double CloudUpperBuffer[];
double CloudLowerBuffer[];
double FeatScoreBuffer[];
double ProjHighBuffer[];
double ProjLowBuffer[];
double PlaceholderBuffer[]; 

//--- Global Objects
CInterop g_interop; // Receiver (SUB)
CInterop g_sender;  // Sender (PUSH)
string   g_objPrefix = "FEAT_HUD_";

//--- HUD State
string g_systemState = "WAITING";
double g_confidence = 0.0;
double g_lastFeatScore = 0.0;
bool   g_vaultActive = false;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- Indicator Buffers
   SetIndexBuffer(0, CloudUpperBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, CloudLowerBuffer, INDICATOR_DATA);
   SetIndexBuffer(2, FeatScoreBuffer,  INDICATOR_DATA);
   SetIndexBuffer(3, ProjHighBuffer,   INDICATOR_DATA);
   SetIndexBuffer(4, ProjLowBuffer,    INDICATOR_DATA);
   SetIndexBuffer(5, PlaceholderBuffer, INDICATOR_CALCULATED);
   
   PlotIndexSetInteger(0, PLOT_LINE_COLOR, 0, clrMediumSeaGreen); // Bullish
   PlotIndexSetInteger(0, PLOT_LINE_COLOR, 1, clrCrimson);        // Bearish
   
   //--- ZMQ Init
   if(!g_interop.Init(false)) // Subscribe Mode
      Print("ZMQ Receiver Init Failed");
      
   if(!g_sender.Init(true))   // Push Mode (Sender)
      Print("ZMQ Sender Init Failed");
   
   EventSetMillisecondTimer(50); // High frequency poll
   
   DrawHUD();
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   ObjectsDeleteAll(0, g_objPrefix);
   g_interop.Shutdown();
   g_sender.Shutdown();
  }

//+------------------------------------------------------------------+
//| Timer event handler                                              |
//+------------------------------------------------------------------+
void OnTimer()
  {
   string msg = g_interop.ReceiveHUD();
   if(msg != "")
     {
      ParseState(msg);
      UpdateHUD();
      ChartRedraw();
     }
  }

//+------------------------------------------------------------------+
//| Parse JSON State from Python                                     |
//+------------------------------------------------------------------+
void ParseState(string json)
  {
   if (StringFind(json, "PING") >= 0)
     {
      g_sender.Send("PONG");
      return; 
     }
  
   // Extract values
   g_systemState  = ExtractJsonString(json, "regime");
   g_confidence   = StringToDouble(ExtractJsonValue(json, "ai_confidence")) / 100.0;
   g_lastFeatScore = StringToDouble(ExtractJsonValue(json, "feat_index"));
   string vaultStr = ExtractJsonValue(json, "vault_active");
   g_vaultActive  = (vaultStr == "true");
   
   // Apply visuals per tick
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   if(g_lastFeatScore > 0)
     {
      // The user wants a purple line for PVP/POI. 
      // We'll use the price level where the High probability was detected or actual POI if sent.
      FeatScoreBuffer[0] = currentPrice; 
     }
     
   // Impulse Projections extraction
   double pHigh = StringToDouble(ExtractJsonValue(json, "proj_high"));
   double pLow = StringToDouble(ExtractJsonValue(json, "proj_low"));
   
   if(pHigh > 0) {
      ProjHighBuffer[0] = pHigh;
      ProjLowBuffer[0]  = pLow;
   }
  }

string ExtractJsonString(string json, string key)
{
   int pos = StringFind(json, "\""+key+"\"");
   if(pos < 0) return "";
   int start = StringFind(json, ":", pos) + 1;
   while(StringSubstr(json, start, 1) == " " || StringSubstr(json, start, 1) == "\"") start++;
   int end = StringFind(json, "\"", start);
   if (end < 0) end = StringFind(json, ",", start);
   if (end < 0) end = StringFind(json, "}", start);
   return StringSubstr(json, start, end-start);
}

string ExtractJsonValue(string json, string key)
{
   int pos = StringFind(json, "\""+key+"\"");
   if(pos < 0) return "0";
   int start = StringFind(json, ":", pos) + 1;
   while(StringSubstr(json, start, 1) == " ") start++;
   int end = StringFind(json, ",", start);
   if (end < 0) end = StringFind(json, "}", start);
   return StringSubstr(json, start, end-start);
}

//+------------------------------------------------------------------+
//| Draw Head-Up Display                                             |
//+------------------------------------------------------------------+
void DrawHUD()
  {
   CreateRectLabel("BG", 10, 30, 240, 120, C'20,20,30', 200);
   CreateLabel("TITLE", "FEAT NEXUS UPLINK V5", 20, 40, clrWhite, 10, true);
   CreateLabel("PILOT", "REGIME: WAITING", 20, 60, clrGray, 9);
   CreateLabel("CONF", "NEURAL CONF: 0%", 20, 80, clrGray, 9);
   CreateLabel("FEAT", "FEAT SCORE: 0.0", 20, 100, clrMediumPurple, 9);
   CreateLabel("VAULT", "VAULT: LOCKED", 20, 120, clrGray, 9);
  }

void UpdateHUD()
  {
   color stateColor = clrGray;
   if(g_systemState == "TREND_GRAVITY") stateColor = clrLime;
   else if(g_systemState == "EXPANSION") stateColor = clrGold;
   else if(g_systemState == "COMPRESSION") stateColor = clrAqua;
   
   ObjectSetString(0, g_objPrefix+"PILOT", OBJPROP_TEXT, "REGIME: " + g_systemState);
   ObjectSetInteger(0, g_objPrefix+"PILOT", OBJPROP_COLOR, stateColor);
   
   int confPct = (int)(g_confidence * 100);
   ObjectSetString(0, g_objPrefix+"CONF", OBJPROP_TEXT, "NEURAL CONF: " + IntegerToString(confPct) + "%");
   
   ObjectSetString(0, g_objPrefix+"FEAT", OBJPROP_TEXT, "FEAT SCORE: " + DoubleToString(g_lastFeatScore, 1));
   
   color vaultColor = g_vaultActive ? clrAqua : clrSilver;
   string vaultTxt = g_vaultActive ? "VAULT: ACTIVE" : "VAULT: SAFE";
   ObjectSetString(0, g_objPrefix+"VAULT", OBJPROP_TEXT, vaultTxt);
   ObjectSetInteger(0, g_objPrefix+"VAULT", OBJPROP_COLOR, vaultColor);
  }

void CreateLabel(string name, string text, int x, int y, color clr, int fontSize, bool isBold=false)
{
   string objName = g_objPrefix + name;
   if(ObjectFind(0, objName) < 0)
      ObjectCreate(0, objName, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, objName, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetString(0, objName, OBJPROP_TEXT, text);
   ObjectSetInteger(0, objName, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, objName, OBJPROP_FONTSIZE, fontSize);
}

void CreateRectLabel(string name, int x, int y, int w, int h, color bg, int alpha)
{
   string objName = g_objPrefix + name;
   if(ObjectFind(0, objName) < 0)
      ObjectCreate(0, objName, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, objName, OBJPROP_XSIZE, w);
   ObjectSetInteger(0, objName, OBJPROP_YSIZE, h);
   ObjectSetInteger(0, objName, OBJPROP_BGCOLOR, bg);
   ObjectSetInteger(0, objName, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   ArraySetAsSeries(FeatScoreBuffer, true);
   ArraySetAsSeries(ProjHighBuffer, true);
   ArraySetAsSeries(ProjLowBuffer, true);
   ArraySetAsSeries(CloudUpperBuffer, true);
   ArraySetAsSeries(CloudLowerBuffer, true);
   
   return(rates_total);
  }
