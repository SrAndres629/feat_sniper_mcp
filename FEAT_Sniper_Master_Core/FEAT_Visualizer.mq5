//+------------------------------------------------------------------+
//|                        FEAT_Visualizer_Master_v4.mq5             |
//|               Cerebro Analítico FEAT - Institutional GUI         |
//|           Sincronización: ZMQ BRIDGE | IA Nexus Integration      |
//+------------------------------------------------------------------+
#property copyright "FEAT System - Nexus Architecture"
#property description "Visualizador Institucional: Forma, Espacio, Aceleración, Tiempo"
#property version   "4.00"
#property indicator_chart_window
#property indicator_buffers 6
#property indicator_plots   4

//--- PLOTS VISUALES EN EL GRÁFICO
#property indicator_label1  "PvP_Zone_High"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrMediumPurple
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

#property indicator_label2  "PvP_Zone_Low"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrMediumPurple
#property indicator_style2  STYLE_DOT
#property indicator_width2  1

#property indicator_label3  "Structure_BOS"
#property indicator_type3   DRAW_ARROW
#property indicator_color3  clrDodgerBlue

#property indicator_label4  "Structure_CHOCH"
#property indicator_type4   DRAW_ARROW
#property indicator_color4  clrCrimson

//--- INCLUDES
#include <UnifiedModel/CInterop.mqh> 
#include <Canvas/Canvas.mqh>         

//--- GLOBALES
CInterop g_interop;        
CCanvas  Dashboard;         
string   g_objPrefix = "FEAT_KZ_";

//--- BUFFERS
double BufferPvP_H[];
double BufferPvP_L[];
double BufferStruct_BOS[];
double BufferStruct_CHOCH[];
double BufferVol[];
double BufferCalc[];

//--- ESTADO DEL SISTEMA (Recibido desde Python)
struct FeatState {
   double score;           // -100 a +100
   string ai_prediction;   // BUY FLOW / SELL FLOW / WAIT
   double ai_confidence;   // %
   string active_zone;     // Killzone status
   string regime;          // Market Regime
   double pvp_level;       // Central PVP
   bool   acceleration;    // Momentum flag
};
FeatState CurrentState;

//+------------------------------------------------------------------+
//| INIT                                                             |
//+------------------------------------------------------------------+
int OnInit() {
   if(!SetIndexBuffer(0, BufferPvP_H)) return(INIT_FAILED);
   if(!SetIndexBuffer(1, BufferPvP_L)) return(INIT_FAILED);
   if(!SetIndexBuffer(2, BufferStruct_BOS)) return(INIT_FAILED);
   if(!SetIndexBuffer(3, BufferStruct_CHOCH)) return(INIT_FAILED);
   if(!SetIndexBuffer(4, BufferVol)) return(INIT_FAILED);
   if(!SetIndexBuffer(5, BufferCalc)) return(INIT_FAILED);

   PlotIndexSetInteger(2, PLOT_ARROW, 159); 
   PlotIndexSetInteger(3, PLOT_ARROW, 158); 
   
   if(!g_interop.Init(false)) {
      Print("CRITICAL: ZMQ Bridge Failed.");
      return(INIT_FAILED);
   }

   if(!Dashboard.CreateBitmapLabel("FEAT_HUD", 5, 25, 350, 180, COLOR_FORMAT_ARGB_NORMALIZE)) {
      Print("Error creando Dashboard GUI");
      return(INIT_FAILED);
   }
   
   EventSetMillisecondTimer(100); 
   DrawLoadingScreen();
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
   EventKillTimer();
   g_interop.Shutdown();
   Dashboard.Destroy();
   ObjectsDeleteAll(0, g_objPrefix);
}

void OnTimer() {
   string json = g_interop.ReceiveHUD();
   if(json != "") {
      ParseTelemetry(json); 
      UpdateDashboard();
      UpdateChartVisuals(); 
   }
}

void ParseTelemetry(string json) {
   CurrentState.regime     = ExtractJsonValue(json, "regime");
   CurrentState.ai_confidence = StringToDouble(ExtractJsonValue(json, "ai_confidence"));
   CurrentState.score      = StringToDouble(ExtractJsonValue(json, "feat_score_val"));
   CurrentState.pvp_level  = StringToDouble(ExtractJsonValue(json, "feat_pvp_price"));
   CurrentState.active_zone = ExtractJsonValue(json, "session");
   
   if(CurrentState.score > 50) CurrentState.ai_prediction = "BUY FLOW >>";
   else if(CurrentState.score < -50) CurrentState.ai_prediction = "<< SELL FLOW";
   else CurrentState.ai_prediction = "-- WAIT --";
   
   CurrentState.acceleration = (StringFind(CurrentState.regime, "TREND") >= 0);
}

void UpdateChartVisuals() {
   int total = ArraySize(BufferPvP_H);
   if(total > 0) {
      BufferPvP_H[0] = CurrentState.pvp_level + (SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 50);
      BufferPvP_L[0] = CurrentState.pvp_level - (SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 50);
   }
}

void UpdateDashboard() {
   Dashboard.Erase(0);
   Dashboard.FillRectangle(0, 0, 350, 180, ColorToARGB(clrBlack, 230));
   Dashboard.Rectangle(0, 0, 349, 179, ColorToARGB(clrGray, 255));
   
   Dashboard.TextOut(15, 10, "FEAT SYSTEM v4 | INSTITUTIONAL", ColorToARGB(clrLightGray, 255), TA_LEFT);
   
   int barX = 15, barY = 40, barW = 320, barH = 15;
   Dashboard.FillRectangle(barX, barY, barX+barW, barY+barH, ColorToARGB(clrDarkSlateGray, 255));
   
   double pct = (CurrentState.score + 100.0) / 200.0; 
   if(pct < 0) pct = 0; if(pct > 1) pct = 1;
   int fillW = (int)(barW * pct);
   
   uint sColor = ColorToARGB(clrGray, 255);
   if(CurrentState.score > 30) sColor = ColorToARGB(clrSpringGreen, 255);
   if(CurrentState.score < -30) sColor = ColorToARGB(clrCrimson, 255);
   
   Dashboard.FillRectangle(barX, barY, barX+fillW, barY+barH, sColor);
   Dashboard.TextOut(barX, barY+20, StringFormat("FEAT SCORE: %.2f", CurrentState.score), sColor, TA_LEFT);

   Dashboard.TextOut(15, 80, "NEXUS AI PREDICTION:", ColorToARGB(clrWhite, 255), TA_LEFT);
   
   uint predColor = ColorToARGB(clrYellow, 255);
   if(CurrentState.ai_prediction == "BUY FLOW >>") predColor = ColorToARGB(clrLime, 255);
   else if(CurrentState.ai_prediction == "<< SELL FLOW") predColor = ColorToARGB(clrRed, 255);
   
   Dashboard.TextOut(15, 100, CurrentState.ai_prediction, predColor, TA_LEFT);
   Dashboard.TextOut(200, 105, StringFormat("Conf: %.1f%%", CurrentState.ai_confidence), ColorToARGB(clrSilver, 255), TA_LEFT);

   Dashboard.TextOut(15, 140, "MARKET: " + (CurrentState.acceleration ? "HIGH MOMENTUM" : "LOW VOL"), ColorToARGB(clrCyan, 255), TA_LEFT);
   Dashboard.TextOut(200, 140, "SESSION: " + CurrentState.active_zone, ColorToARGB(clrGold, 255), TA_LEFT);

   // --- INSTITUTIONAL ACCOUNT INFO ---
   string accMode = (AccountInfoInteger(ACCOUNT_TRADE_MODE) == ACCOUNT_TRADE_MODE_DEMO) ? "DEMO" : "REAL";
   uint accColor = (accMode == "REAL") ? ColorToARGB(clrRed, 255) : ColorToARGB(clrLimeGreen, 255);
   
   Dashboard.TextOut(15, 160, StringFormat("ACC: %s | BAL: $%.2f", accMode, AccountInfoDouble(ACCOUNT_BALANCE)), accColor, TA_LEFT);
   Dashboard.TextOut(200, 160, StringFormat("EQ: $%.2f", AccountInfoDouble(ACCOUNT_EQUITY)), ColorToARGB(clrWhiteSmoke, 255), TA_LEFT);

   Dashboard.Update();
}

void DrawLoadingScreen() {
   if(Dashboard.ChartObjectName() != "") {
      Dashboard.Erase(ColorToARGB(clrBlack, 255));
      Dashboard.TextOut(175, 75, "BOOTING FEAT...", ColorToARGB(clrLime, 255), TA_CENTER);
      Dashboard.Update();
   }
}

string ExtractJsonValue(string json, string key) {
   int pos = StringFind(json, "\""+key+"\"");
   if(pos < 0) return("");
   int s = StringFind(json, ":", pos) + 1;
   while(s < StringLen(json) && (StringSubstr(json, s, 1) == " " || StringSubstr(json, s, 1) == "\"")) s++;
   int e = StringFind(json, ",", s);
   if (e < 0) e = StringFind(json, "}", s);
   int e2 = StringFind(json, "\"", s);
   if (e2 > 0 && e2 < e) e = e2;
   if (e < s) return("");
   return(StringSubstr(json, s, e-s));
}

int OnCalculate(const int rates_total, const int prev_calculated, const datetime &time[], const double &open[], const double &high[], const double &low[], const double &close[], const long &tick_volume[], const long &volume[], const int &spread[])
{
   ArraySetAsSeries(BufferPvP_H, true);
   ArraySetAsSeries(BufferPvP_L, true);
   return(rates_total);
}
