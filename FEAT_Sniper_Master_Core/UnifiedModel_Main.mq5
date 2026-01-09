//+------------------------------------------------------------------+
//|                                          UnifiedModel_Main.mq5 |
//|                    Unified Institutional Model                  |
//|    Senior Quantitative Engineer Edition (Gas, Water, Wall, Bedrock)|
//+------------------------------------------------------------------+
#property copyright "Institutional Trading Systems"
#property link      "https://github.com/SrAndres629/feat_sniper_mcp"
#property version   "2.25"
#property indicator_chart_window
#property indicator_buffers 31
#property indicator_plots   31

// --- EMA VISUAL HIERARCHY (SENIOR ENGINEER PHYSICS) ---

// 1. MICRO (GAS) - Yellow/Gold (Momentum)
#property indicator_label1 "M1"
#property indicator_type1 DRAW_LINE
#property indicator_color1 clrGold
#property indicator_width1 1
#property indicator_label2 "M2"
#property indicator_type2 DRAW_LINE
#property indicator_color2 clrGold
#property indicator_width2 1
#property indicator_label3 "M3"
#property indicator_type3 DRAW_LINE
#property indicator_color3 clrGold
#property indicator_width3 1
#property indicator_label4 "M4"
#property indicator_type4 DRAW_LINE
#property indicator_color4 clrYellow
#property indicator_width4 1
#property indicator_label5 "M5 (WIND/GAS)"
#property indicator_type5 DRAW_LINE
#property indicator_color5 clrYellow
#property indicator_width5 2    // Core Gas
#property indicator_label6 "M6"
#property indicator_type6 DRAW_LINE
#property indicator_color6 clrYellow
#property indicator_width6 1
#property indicator_label7 "M7"
#property indicator_type7 DRAW_LINE
#property indicator_color7 clrKhaki
#property indicator_width7 1
#property indicator_label8 "M8"
#property indicator_type8 DRAW_LINE
#property indicator_color8 clrKhaki
#property indicator_width8 1
#property indicator_label9 "M9"
#property indicator_type9 DRAW_LINE
#property indicator_color9 clrPaleGoldenrod
#property indicator_width9 1
#property indicator_label10 "M10"
#property indicator_type10 DRAW_LINE
#property indicator_color10 clrPaleGoldenrod
#property indicator_width10 1

// 2. OPERATIONAL (WATER) - Orange/Red (Friction)
#property indicator_label11 "O1 (RIVER Start)"
#property indicator_type11 DRAW_LINE
#property indicator_color11 clrOrangeRed
#property indicator_width11 1
#property indicator_label12 "O2 (RIVER Core)"
#property indicator_type12 DRAW_LINE
#property indicator_color12 clrRed
#property indicator_width12 2 // EMA 21
#property indicator_label13 "O3"
#property indicator_type13 DRAW_LINE
#property indicator_color13 clrOrangeRed
#property indicator_width13 1

// 3. STRUCTURAL (WALL) - Green (Brick Wall)
#property indicator_label14 "O4 (WALL Core)"
#property indicator_type14 DRAW_LINE
#property indicator_color14 clrLimeGreen
#property indicator_width14 3 // EMA 50 (Thick Wall)
#property indicator_label15 "O5"
#property indicator_type15 DRAW_LINE
#property indicator_color15 clrLime
#property indicator_width15 1
#property indicator_label16 "O6"
#property indicator_type16 DRAW_LINE
#property indicator_color16 clrLime
#property indicator_width16 1
#property indicator_label17 "O7"
#property indicator_type17 DRAW_LINE
#property indicator_color17 clrSpringGreen
#property indicator_width17 1
#property indicator_label18 "O8"
#property indicator_type18 DRAW_LINE
#property indicator_color18 clrSpringGreen
#property indicator_width18 1
#property indicator_label19 "O9"
#property indicator_type19 DRAW_LINE
#property indicator_color19 clrMediumSpringGreen
#property indicator_width19 1
#property indicator_label20 "O10 (MAGNET Lead)"
#property indicator_type20 DRAW_LINE
#property indicator_color20 clrMediumPurple
#property indicator_width20 1

// 4. MACRO (BEDROCK) - Purple/Blue (Gravity)
#property indicator_label21 "Ma1 (BEDROCK Core)"
#property indicator_type21 DRAW_LINE
#property indicator_color21 clrDarkViolet // EMA 200+
#property indicator_width21 2
#property indicator_label22 "Ma2"
#property indicator_type22 DRAW_LINE
#property indicator_color22 clrBlueViolet
#property indicator_width22 1
#property indicator_label23 "Ma3"
#property indicator_type23 DRAW_LINE
#property indicator_color23 clrMediumSlateBlue
#property indicator_width23 1
#property indicator_label24 "Ma4"
#property indicator_type24 DRAW_LINE
#property indicator_color24 clrSlateBlue
#property indicator_width24 1
#property indicator_label25 "Ma5"
#property indicator_type25 DRAW_LINE
#property indicator_color25 clrDodgerBlue
#property indicator_width25 1
#property indicator_label26 "Ma6"
#property indicator_type26 DRAW_LINE
#property indicator_color26 clrDeepSkyBlue
#property indicator_width26 1
#property indicator_label27 "Ma7"
#property indicator_type27 DRAW_LINE
#property indicator_color27 clrDeepSkyBlue
#property indicator_width27 1
#property indicator_label28 "Ma8"
#property indicator_type28 DRAW_LINE
#property indicator_color28 clrSkyBlue
#property indicator_width28 1
#property indicator_label29 "Ma9"
#property indicator_type29 DRAW_LINE
#property indicator_color29 clrSkyBlue
#property indicator_width29 1
#property indicator_label30 "Ma10"
#property indicator_type30 DRAW_LINE
#property indicator_color30 clrLightBlue
#property indicator_width30 1

// Bias: Absolute (Gray)
#property indicator_label31 "Bias"
#property indicator_type31 DRAW_LINE
#property indicator_color31 clrDimGray
#property indicator_width31 1 
#property indicator_style31 STYLE_DOT

#include <UnifiedModel\CEMAs.mqh>
#include <UnifiedModel\CFEAT.mqh>
#include <UnifiedModel\CLiquidity.mqh>
#include <UnifiedModel\CFSM.mqh>
#include <UnifiedModel\CVisuals.mqh>
#include <UnifiedModel\CInterop.mqh>
#include <UnifiedModel\CMultitemporal.mqh>

input group "--- Configuration ---"
input int ATR_Period = 14;
input int Lookback = 100;
input bool ShowDashboard = true; // Toggle Senior Engineer HUD
input bool ExportData = true;    // Export Data for Neural Training
input string Bridge_Path = "c:\\Users\\acord\\OneDrive\\Desktop\\Bot\\feat_sniper_mcp\\FEAT_Sniper_Master_Core\\Python\\start_bridge.bat";

double b0[], b1[], b2[], b3[], b4[], b5[], b6[], b7[], b8[], b9[];
double b10[], b11[], b12[], b13[], b14[], b15[], b16[], b17[], b18[], b19[];
double b20[], b21[], b22[], b23[], b24[], b25[], b26[], b27[], b28[], b29[];
double b30[];

CEMAs       g_emas;
CFEAT       g_feat;
CLiquidity  g_liq;
CFSM        g_fsm;
CVisuals    g_vis;
CInterop    g_io;
CMultitemporal g_mtf;

int OnInit() {
   // Mapping buffers...
   SetIndexBuffer(0, b0); SetIndexBuffer(1, b1); SetIndexBuffer(2, b2); SetIndexBuffer(3, b3);
   SetIndexBuffer(4, b4); SetIndexBuffer(5, b5); SetIndexBuffer(6, b6); SetIndexBuffer(7, b7);
   SetIndexBuffer(8, b8); SetIndexBuffer(9, b9); SetIndexBuffer(10, b10); SetIndexBuffer(11, b11);
   SetIndexBuffer(12, b12); SetIndexBuffer(13, b13); SetIndexBuffer(14, b14); SetIndexBuffer(15, b15);
   SetIndexBuffer(16, b16); SetIndexBuffer(17, b17); SetIndexBuffer(18, b18); SetIndexBuffer(19, b19);
   SetIndexBuffer(20, b20); SetIndexBuffer(21, b21); SetIndexBuffer(22, b22); SetIndexBuffer(23, b23);
   SetIndexBuffer(24, b24); SetIndexBuffer(25, b25); SetIndexBuffer(26, b26); SetIndexBuffer(27, b27);
   SetIndexBuffer(28, b28); SetIndexBuffer(29, b29); SetIndexBuffer(30, b30);

   if(!g_emas.Init(_Symbol, _Period, ATR_Period)) return INIT_FAILED;
   g_feat.SetEMAs(&g_emas);
   g_feat.SetLiquidity(&g_liq);
   if(!g_feat.Init(_Symbol, _Period)) return INIT_FAILED;
   g_liq.Init(_Symbol, _Period, 50, Lookback, 5.0, 0.5);
   g_fsm.SetComponents(&g_emas, &g_feat, &g_liq);
   g_fsm.SetBufferSize(Lookback);
   
   g_vis.Init("UM_", ChartID());
   g_vis.SetComponents(&g_emas, &g_feat, &g_liq, &g_fsm);
   g_vis.SetDrawOptions(true, ShowDashboard, true, true);
   
   g_io.SetEnabled(ExportData);
   g_io.SetDataPath(""); 
   g_io.SetFilename("UnifiedModel_LiveFeed.csv");
   
   g_mtf.Init(_Symbol);
   
   return INIT_SUCCEEDED;
}

void OnDeinit(const int r) { g_feat.Deinit(); g_emas.Deinit(); g_vis.Clear(); }

int OnCalculate(const int total, const int prev, const datetime &time[], const double &open[],
                const double &high[], const double &low[], const double &close[],
                const long &tick[], const long &real[], const int &spread[]) {
   
   if(total < Lookback) return 0;
   
   int limit = (prev == 0) ? total - Lookback : prev - 1;
   for(int i = total - 1 - limit; i >= 0; i--) {
      int idx = total - 1 - i;
      double buff[1];
      // Micro
      if(CopyBuffer(g_emas.GetHandle(0), 0, i, 1, buff) > 0) b0[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(1), 0, i, 1, buff) > 0) b1[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(2), 0, i, 1, buff) > 0) b2[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(3), 0, i, 1, buff) > 0) b3[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(4), 0, i, 1, buff) > 0) b4[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(5), 0, i, 1, buff) > 0) b5[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(6), 0, i, 1, buff) > 0) b6[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(7), 0, i, 1, buff) > 0) b7[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(8), 0, i, 1, buff) > 0) b8[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(9), 0, i, 1, buff) > 0) b9[idx] = buff[0];
      // Operational
      if(CopyBuffer(g_emas.GetHandle(10), 0, i, 1, buff) > 0) b10[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(11), 0, i, 1, buff) > 0) b11[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(12), 0, i, 1, buff) > 0) b12[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(13), 0, i, 1, buff) > 0) b13[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(14), 0, i, 1, buff) > 0) b14[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(15), 0, i, 1, buff) > 0) b15[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(16), 0, i, 1, buff) > 0) b16[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(17), 0, i, 1, buff) > 0) b17[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(18), 0, i, 1, buff) > 0) b18[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(19), 0, i, 1, buff) > 0) b19[idx] = buff[0];
      // Macro
      if(CopyBuffer(g_emas.GetHandle(20), 0, i, 1, buff) > 0) b20[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(21), 0, i, 1, buff) > 0) b21[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(22), 0, i, 1, buff) > 0) b22[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(23), 0, i, 1, buff) > 0) b23[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(24), 0, i, 1, buff) > 0) b24[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(25), 0, i, 1, buff) > 0) b25[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(26), 0, i, 1, buff) > 0) b26[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(27), 0, i, 1, buff) > 0) b27[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(28), 0, i, 1, buff) > 0) b28[idx] = buff[0];
      if(CopyBuffer(g_emas.GetHandle(29), 0, i, 1, buff) > 0) b29[idx] = buff[0];
      // Bias
      if(CopyBuffer(g_emas.GetHandle(30), 0, i, 1, buff) > 0) b30[idx] = buff[0];
   }
   
   static datetime last = 0;
   if(time[total-1] != last) {
      last = time[total-1];
      
      // 1. UPDATE CALCULATIONS
      g_emas.Calculate(0); 
      g_liq.Calculate(high, low, open, close, time, total, close[total-1]);
      g_feat.Calculate((ENUM_TIMEFRAMES)_Period, time[total-1], open[total-1], high[total-1], low[total-1], close[total-1], (double)tick[total-1]);
      g_fsm.Calculate(close[total-1], close[total-2], (double)tick[total-1]);
      g_mtf.Calculate();  
      
      // 2. DRAW VISUALS
      if(ShowDashboard) g_vis.Draw(time[total-1], close[total-1]);
      else g_vis.Clear();
      
      // 3. EXPORT DATA
      if(ExportData) {
         SBarDataExport data;
         data.time = time[total-1];
         data.open = open[total-1];
         data.high = high[total-1];
         data.low = low[total-1];
         data.close = close[total-1];
         data.volume = (double)tick[total-1];
         
         SEMAGroupMetrics mic = g_emas.GetMicroMetrics();
         SEMAGroupMetrics opr = g_emas.GetOperationalMetrics();
         SEMAGroupMetrics mac = g_emas.GetMacroMetrics();
         SFanMetrics fan = g_emas.GetFanMetrics();
         CFEAT::SResult feat = g_feat.GetResult(); // Use scoped resolution for embedded struct
         SMultitemporalResult mtf = g_mtf.GetResult();
         
         data.microComp = mic.compression;
         data.microSlope = mic.avgSlope;
         data.microCurvature = feat.form.curvatureScore;
         data.operComp = opr.compression;
         data.operSlope = opr.avgSlope;
         data.macroSlope = mac.avgSlope;
         data.biasSlope = g_emas.GetBiasSlope();
         
         data.layerSep12 = feat.space.fastMediumGap;
         data.layerSep23 = feat.space.mediumSlowGap;
         data.fanBullish = fan.bullishOrder;
         data.fanBearish = !fan.bullishOrder;
         
         data.hasBOS = feat.form.hasBOS;
         data.hasCHoCH = feat.form.hasCHoCH;
         data.hasHCH = feat.form.hasHCH;
         data.isIntentCandle = feat.form.isIntentCandle;
         data.curvatureScore = feat.form.curvatureScore;
         data.compressionRatio = feat.form.compressionRatio;
         
         data.atZone = feat.space.atZone;
         data.proximityScore = feat.space.proximityScore;
         data.activeZoneType = feat.space.activeZoneType;
         
         data.velocity = feat.accel.velocity;
         data.momentum = feat.accel.momentum;
         data.deltaFlow = feat.accel.deltaFlow;
         data.rsi = feat.accel.rsi;
         data.macdHist = feat.accel.macdHist;
         data.ao = feat.accel.ao;
         data.ac = feat.accel.ac;
         data.isInstitutional = feat.accel.isInstitutional;
         data.isExhausted = feat.accel.isExhausted;
         
         data.isKillzone = feat.time.isKillzone;
         data.isLondonKZ = (StringFind(feat.time.activeSession, "LONDON") >= 0); 
         data.isNYKZ = (StringFind(feat.time.activeSession, "NY") >= 0); 
         data.isAgainstH4 = false; // Deprecated but kept for struct
         data.h4Direction = 0;
         data.activeSession = feat.time.activeSession;
         
         data.marketState = g_fsm.GetStateString();
         data.compositeScore = feat.compositeScore;
         
         data.dominantTrend = mtf.dominantTrend;
         data.mtfConfluence = mtf.confluenceScore;
         data.mtfAgainstBias = mtf.isAgainstBias;
         data.m5Bias = mtf.states[0].bias;
         data.h1Bias = mtf.states[1].bias;
         data.h4Bias = mtf.states[2].bias;
         data.d1Bias = mtf.states[3].bias;
         
         g_io.ExportFeatures(data);
      }
   }
   
   return total;
}
