//+------------------------------------------------------------------+
//|                                     FEAT_Visualizer_Master_v5.mq5|
//|                   Cerebro Analítico FEAT - Institutional Master  |
//|               Engineering: OOP | ZMQ Hyperbridge | Canvas GPU    |
//+------------------------------------------------------------------+
#property copyright "FEAT Systems AI | Quant Arch: Omega"
#property description "Institutional Grade: Volatility Envelopes, Delta-Neutral GUI, HFT Bridge"
#property version   "5.00"
#property indicator_chart_window
#property indicator_buffers 7
#property indicator_plots   5

//--- PLOTS: DYNAMIC PVP ENVELOPES (Violin Cloud)
#property indicator_label1  "PvP_Cloud_High"
#property indicator_type1   DRAW_FILLING
#property indicator_color1  clrMediumPurple, clrMediumPurple
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1

#property indicator_label2  "PvP_Cloud_Low"
#property indicator_type2   DRAW_NONE 

//--- PLOTS: STRUCTURE (BOS/CHOCH)
#property indicator_label3  "Structure_BOS"
#property indicator_type3   DRAW_ARROW
#property indicator_color3  clrDodgerBlue
#property indicator_width3  2

#property indicator_label4  "Structure_CHOCH"
#property indicator_type4   DRAW_ARROW
#property indicator_color4  clrCrimson
#property indicator_width4  2

//--- PLOTS: HFT ACTIVITY (Volume/Tick Heat)
#property indicator_label5  "HFT_Pulse"
#property indicator_type5   DRAW_NONE

//--- INCLUDES
#include <UnifiedModel/CInterop.mqh> 
#include <Canvas/Canvas.mqh>         

//--- INPUTS
input int      InpZMQPort        = 5556;        // ZMQ Sub Port
input color    InpColorBuy       = clrLimeGreen;// Buy Flow Color
input color    InpColorSell      = clrCrimson;  // Sell Flow Color
input int      InpSparklineHist  = 20;          // Sparkline Depth

//--- BUFFERS
double BufferPvP_H[];
double BufferPvP_L[];
double BufferStruct_BOS[];
double BufferStruct_CHOCH[];
double BufferHFT[];
// Shadow Buffers for calculation
double BufferCalc_1[];
double BufferCalc_2[];

//+------------------------------------------------------------------+
//| CLASS: CMarketLogic                                              |
//| Kernel: Mathematical Modeling & State Management                 |
//+------------------------------------------------------------------+
class CMarketLogic
{
public:
   struct State {
      double score;
      string prediction;
      double confidence;
      string session_state;
      string regime;
      double pvp_level;
      bool   acceleration;
      // New: Volatility Factors
      double vol_factor;
      string raw_json;
      ulong  last_update_ms;
   };

   State m_state;
   double m_sparkline[];
   
   CMarketLogic() {
      ArrayResize(m_sparkline, InpSparklineHist);
      ArrayInitialize(m_sparkline, 0.0);
      m_state.score = 0;
      m_state.vol_factor = 1.0;
   }
   
   // --- PARSING & STATE UPDATE ---
   bool UpdateState(string json) {
      // 1. Hash Check (Simple String Compare for Low Latency)
      if(json == m_state.raw_json) return false;
      m_state.raw_json = json;
      m_state.last_update_ms = GetTickCount64();
      
      // 2. Parse JSON
      m_state.regime     = ExtractJsonValue(json, "regime");
      m_state.confidence = StringToDouble(ExtractJsonValue(json, "ai_confidence"));
      m_state.score      = StringToDouble(ExtractJsonValue(json, "feat_score_val"));
      m_state.pvp_level  = StringToDouble(ExtractJsonValue(json, "feat_pvp_price"));
      m_state.session_state = ExtractJsonValue(json, "session");
      
      string acc_str     = ExtractJsonValue(json, "acceleration"); // Assuming bool or number
      m_state.acceleration = (StringFind(m_state.regime, "TREND") >= 0); // Logic override based on reg

      // 3. Update Sparkline (FIFO)
      for(int i=InpSparklineHist-1; i>0; i--) {
         m_sparkline[i] = m_sparkline[i-1];
      }
      m_sparkline[0] = m_state.score;
      
      // 4. Volatility Dynamics (Breathing Function)
      // If Accelerating -> Compress (Focus). If Laminar -> Expand (Discovery).
      if(m_state.acceleration) m_state.vol_factor = 0.5; // Tight
      else m_state.vol_factor = 1.5; // Wide for consolidation
      
      // Prediction String
      if(m_state.score > 50) m_state.prediction = "BUY FLOW >>";
      else if(m_state.score < -50) m_state.prediction = "<< SELL FLOW";
      else m_state.prediction = "-- NEUTRAL --";
      
      return true; // State changed
   }

private:
    string ExtractJsonValue(string json, string key) {
       string search = "\"" + key + "\"";
       int start = StringFind(json, search);
       if(start < 0) return("");
       
       int valStart = StringFind(json, ":", start);
       if(valStart < 0) return("");
       
       valStart++;
       while(valStart < StringLen(json) && 
             (StringGetCharacter(json, valStart) == ' ' || StringGetCharacter(json, valStart) == '"')) {
          valStart++;
       }
       
       int valEnd = valStart;
       while(valEnd < StringLen(json)) {
          ushort c = StringGetCharacter(json, valEnd);
          if(c == ',' || c == '}') break;
          valEnd++;
       }
       
       string res = StringSubstr(json, valStart, valEnd - valStart);
       StringReplace(res, "\"", "");
       return(res);
    }
};

//+------------------------------------------------------------------+
//| CLASS: CVisualEngine                                             |
//| Kernel: GPU Canvas Rendering & Glassmorphism                     |
//+------------------------------------------------------------------+
class CVisualEngine
{
private:
   CCanvas m_canvas;
   int     m_width;
   int     m_height;
   
public:
   CVisualEngine() : m_width(400), m_height(220) {}
   ~CVisualEngine() { m_canvas.Destroy(); }
   
   bool Init() {
      if(!m_canvas.CreateBitmapLabel("FEAT_HUD_V5", 5, 25, m_width, m_height, COLOR_FORMAT_ARGB_NORMALIZE))
         return false;
      return true;
   }
   
   void Render(CMarketLogic &logic) {
      // 1. Glassmorphism Background
      // Dark Semi-transparent base
      m_canvas.FillRectangle(0, 0, m_width, m_height, ColorToARGB(clrBlack, 235));
      // Subtle Glow Border (Neon Cyan or Purple based on PVP)
      uint borderColor = logic.m_state.acceleration ? ColorToARGB(clrCyan, 120) : ColorToARGB(clrMediumPurple, 100);
      m_canvas.Rectangle(0, 0, m_width-1, m_height-1, borderColor);
      
      // 2. Header
      m_canvas.TextOut(15, 10, "FEAT INSTITUTIONAL // V5", ColorToARGB(clrWhiteSmoke, 255), TA_LEFT);
      
      // 3. FEAT Gradient Bar
      RenderGradientBar(logic.m_state.score);
      
      // 4. Telemetry Grid
      RenderTelemetry(logic.m_state);
      
      // 5. Sparkline (AI Trend)
      RenderSparkline(logic.m_sparkline);
      
      // 6. Account Status (Bottom)
      RenderAccountStatus();
      
      m_canvas.Update();
   }
   
private:
   void RenderGradientBar(double score) {
      int x=15, y=40, w=370, h=12;
      
      // Background track
      m_canvas.FillRectangle(x, y, x+w, y+h, ColorToARGB(clrDarkSlateGray, 255));
      
      // Gradient Fill (Simulated)
      // Center (0) is Gray, Left (-100) Red, Right (+100) Green
      double norm = (score + 100.0) / 200.0; 
      if(norm < 0) norm=0; if(norm>1) norm=1;
      
      int fillW = (int)(w * norm);
      
      uint c1 = ColorToARGB(clrCrimson, 255);
      uint c2 = ColorToARGB(clrLime, 255);
      // Simple coloring for reliability: Red if negative, Green if positive
      uint barColor = score >= 0 ? c2 : c1;
      
      // Draw from center? Or left-to-right? Standard progress bar:
      m_canvas.FillRectangle(x, y, x+fillW, y+h, barColor);
      
      // Center Marker
      m_canvas.Line(x+(w/2), y-2, x+(w/2), y+h+2, ColorToARGB(clrWhite, 255));
      
      // Triangle Marker at current position
      int triX = x + fillW;
      int triY = y + h + 2;
      m_canvas.FillTriangle(triX, triY, triX-5, triY+5, triX+5, triY+5, ColorToARGB(clrWhite, 255));
      
      m_canvas.TextOut(x, y+22, StringFormat("NEURAL SCORE: %.2f", score), barColor, TA_LEFT);
   }
   
   void RenderTelemetry(CMarketLogic::State &state) {
      int col1_x = 15;
      int col2_x = 220;
      int row_y  = 90;
      int step   = 20;
      
      // AI Signal
      uint sigColor = clrSilver;
      if(StringFind(state.prediction, "BUY") >= 0) sigColor = clrLime;
      if(StringFind(state.prediction, "SELL") >= 0) sigColor = clrRed;
      
      m_canvas.TextOut(col1_x, row_y, "SIGNAL:", ColorToARGB(clrGray, 200));
      m_canvas.TextOut(col1_x+60, row_y, state.prediction, ColorToARGB(sigColor, 255));
      
      m_canvas.TextOut(col2_x, row_y, StringFormat("CONF: %.1f%%", state.confidence), ColorToARGB(clrWhite, 255));
      
      row_y += step;
      m_canvas.TextOut(col1_x, row_y, "REGIME:", ColorToARGB(clrGray, 200));
      m_canvas.TextOut(col1_x+60, row_y, state.regime, ColorToARGB(clrCyan, 255));
      
      m_canvas.TextOut(col2_x, row_y, state.session_state, ColorToARGB(clrGold, 255));
   }
   
   void RenderSparkline(double &data[]) {
      int x=15, y=140, w=370, h=40;
      int count = ArraySize(data);
      if(count < 2) return;
      
      double maxV = 100, minV = -100;
      double dx = (double)w / (count - 1);
      
      // Draw Frame
      m_canvas.Rectangle(x, y, x+w, y+h, ColorToARGB(clrGray, 50));
      m_canvas.Line(x, y+(h/2), x+w, y+(h/2), ColorToARGB(clrGray, 50)); // Zero line
      
      for(int i=1; i<count; i++) {
         int x1 = x + (int)((count - 1 - i) * dx); // Reverse visual order (0 is newest on right)
         int x2 = x + (int)((count - 1 - (i-1)) * dx);
         
         // Normalize Y (-100..100 -> h..0)
         int y1 = y + h - (int)((data[i] + 100) / 200.0 * h);
         int y2 = y + h - (int)((data[i-1] + 100) / 200.0 * h);
         
         uint lc = (data[i-1] >= 0) ? ColorToARGB(clrLime, 200) : ColorToARGB(clrRed, 200);
         m_canvas.Line(x1, y1, x2, y2, lc);
      }
   }
   
   void RenderAccountStatus() {
      int y = 195;
      string mode = (AccountInfoInteger(ACCOUNT_TRADE_MODE)==ACCOUNT_TRADE_MODE_DEMO) ? "DEMO" : "REAL";
      uint c = (mode=="REAL") ? ColorToARGB(clrRed, 255) : ColorToARGB(clrLime, 255);
      
      string txt = StringFormat("[%s] BAL: $%.2f | EQ: $%.2f", 
         mode, AccountInfoDouble(ACCOUNT_BALANCE), AccountInfoDouble(ACCOUNT_EQUITY));
      
      m_canvas.TextOut(m_width/2, y, txt, c, TA_CENTER);
      m_canvas.TextOut(m_width-10, y+5, "5.0", ColorToARGB(clrGray, 100), TA_RIGHT); // Ver
   }
};

//+------------------------------------------------------------------+
//| CLASS: CNexusBridge (ZMQ)                                        |
//+------------------------------------------------------------------+
class CNexusBridge
{
private:
   CInterop m_zmq;
   bool     m_active;
   
public:
   bool Init() {
      m_active = m_zmq.Init(false);
      return m_active;
   }
   
   void Shutdown() {
      m_zmq.Shutdown();
   }
   
   string GetPacket() {
      if(!m_active) return "";
      return m_zmq.ReceiveHUD();
   }
};

//--- GLOBALS (Instances)
CNexusBridge   Bridge;
CVisualEngine  Gui;
CMarketLogic   Logic;

//+------------------------------------------------------------------+
//| Init                                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   // Buffers mapping
   SetIndexBuffer(0, BufferPvP_H, INDICATOR_DATA);
   SetIndexBuffer(1, BufferPvP_L, INDICATOR_DATA);
   SetIndexBuffer(2, BufferStruct_BOS, INDICATOR_DATA);
   SetIndexBuffer(3, BufferStruct_CHOCH, INDICATOR_DATA);
   SetIndexBuffer(4, BufferHFT, INDICATOR_CALCULATIONS);
   SetIndexBuffer(5, BufferCalc_1, INDICATOR_CALCULATIONS);
   SetIndexBuffer(6, BufferCalc_2, INDICATOR_CALCULATIONS);
   
   // ZMQ
   if(!Bridge.Init()) {
      Print("WARNING: Nexus Bridge Failed - Running in OFFLINE mode.");
      // Don't fail - allow indicator to show offline status
   }
   
   // GUI
   if(!Gui.Init()) {
      Print("GUI Init Failed");
      return(INIT_FAILED);
   }
   
   // CRITICAL: Render initial frame immediately
   Logic.m_state.regime = "OFFLINE";
   Logic.m_state.confidence = 0.0;
   Logic.m_state.score = 0.0;
   Gui.Render(Logic);
   
   EventSetMillisecondTimer(100); 
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Deinit                                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Bridge.Shutdown();
   // VisualEngine handles canvas destroy in destructor
}

//+------------------------------------------------------------------+
//| Timer (The Nerve Loop)                                           |
//+------------------------------------------------------------------+
void OnTimer()
{
   string json = Bridge.GetPacket();
   
   if(json != "") {
      // Logic Core handles Hash Check internally
      if(Logic.UpdateState(json)) {
         // Only Render if state changed
         Gui.Render(Logic);
         UpdateChartObjects();
      }
   } else {
      // No data - still render to show offline status periodically
      static int tick_count = 0;
      tick_count++;
      if(tick_count % 50 == 0) { // Every 5 seconds
         Logic.m_state.regime = "WAITING";
         Gui.Render(Logic);
      }
   }
}

//+------------------------------------------------------------------+
//| Chart Updates (PvP Envelopes)                                    |
//+------------------------------------------------------------------+
void UpdateChartObjects()
{
   // Here we update the indicator buffers for the Cloud
   
   double pvp = Logic.m_state.pvp_level;
   if(pvp <= 0) return;
   
   // Dynamic Envelopes: "Breathing"
   // Base width: 50 points. 
   // Volatility Multiplier: Logic.m_state.vol_factor
   double points = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double baseWidth = 50 * points;
   double dynamicWidth = baseWidth * Logic.m_state.vol_factor;
   
   // Update current bar
   // (In a real scenario we might update a rolling window, but for HUD we focus on current state visualization)
   BufferPvP_H[0] = pvp + dynamicWidth;
   BufferPvP_L[0] = pvp - dynamicWidth;
   
   // For history, we just hold the level (simplification for HUD indicator)
   // Or we could shift the buffer if we were tracking history in MQL5 arrays
}

//+------------------------------------------------------------------+
//| OnCalculate                                                      |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated, const datetime &time[], const double &open[], const double &high[], const double &low[], const double &close[], const long &tick_volume[], const long &volume[], const int &spread[])
{
   ArraySetAsSeries(BufferPvP_H, true);
   ArraySetAsSeries(BufferPvP_L, true);
   ArraySetAsSeries(BufferStruct_BOS, true);
   ArraySetAsSeries(BufferStruct_CHOCH, true);
   
   return(rates_total);
}
