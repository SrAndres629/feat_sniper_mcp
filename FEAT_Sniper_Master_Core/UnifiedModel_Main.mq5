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
#include <UnifiedModel\CInterop.mqh>
#include <UnifiedModel\CMultitemporal.mqh>
// Reactivated CVisuals
#include <UnifiedModel\CVisuals.mqh>

input group "--- Configuration ---"
input int ATR_Period = 14;
input int Lookback = 100;
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
CVisuals    g_vis; // Reactivated
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
   
   // Visuals Reactivated
   g_vis.Init(_Symbol, 50);
   
   g_io.SetEnabled(ExportData);
   g_io.SetDataPath(""); 
   g_io.SetFilename("UnifiedModel_LiveFeed.csv");
   
   g_mtf.Init(_Symbol);
   
   return INIT_SUCCEEDED;
}

void OnDeinit(const int r) {
   g_feat.Deinit();
   g_emas.Deinit();
   g_vis.Clear();
}

int OnCalculate(const int total, const int prev, const datetime &time[], const double &open[],
                const double &high[], const double &low[], const double &close[],
                const long &tick[], const long &real[], const int &spread[]) {
   
   if(total < Lookback) return 0;
   
   // FIX: Historical Calculation Loop
   // We must run the full logic for every historical bar to ensure correct state evolution.
   // This is heavy but required for "Escaneo Invisible".

   int start = prev - 1;
   if(prev == 0) start = Lookback; // Start from Lookback if full recalc
   if(start < Lookback) start = Lookback; // Safety clamp
   
   for(int i = start; i < total; i++) {
       // 1. Calculate EMAs for index i (Shift is reversed relative to loop index)
       // MT5 indicators iterate forward 0..total-1.
       // CopyBuffer functions typically use 'shift' back from current time.
       // However, CEMAs::Calculate uses CopyBuffer.
       // We must simulate the state at time[i].
       // Since CEMAs uses 'shift' relative to end, we calculate shift as:
       int shift = total - 1 - i;

       // Note: CEMAs logic relies on CopyBuffer from live handles.
       // If we calculate for history, CopyBuffer(..., shift, ...) correctly grabs past data.
       g_emas.Calculate(shift);

       // 2. Liquidity (Heavy)
       // Simulate that we only see data up to 'i'.
       // Count passed is i + 1. Current Price is close[i].
       g_liq.Calculate(high, low, open, close, time, i + 1, close[i]);

       // 3. FEAT Logic
       g_feat.Calculate((ENUM_TIMEFRAMES)_Period, time[i], open[i], high[i], low[i], close[i], (double)tick[i]);

       // 4. FSM Logic
       double prevClose = (i > 0) ? close[i-1] : open[i];
       g_fsm.Calculate(close[i], prevClose, (double)tick[i]);

       // 5. Multitemporal
       // Note: MTF logic might be tricky as it relies on other TF handles.
       // Assuming it handles history correctly or just takes current.
       // Given MTF is usually "current higher timeframe status", accurate historical replay is hard without multi-currency tester.
       // We run it anyway.
       g_mtf.Calculate();

       // 6. Update Visual Buffers (EMAs)
       int idx = i; // Buffer index matches loop index
       double buff[1];

       // Just grabbing values from g_emas getters which hold current step values
       // Micro
       b0[idx] = g_emas.GetEMA(0); b1[idx] = g_emas.GetEMA(1); b2[idx] = g_emas.GetEMA(2); b3[idx] = g_emas.GetEMA(3);
       b4[idx] = g_emas.GetEMA(4); b5[idx] = g_emas.GetEMA(5); b6[idx] = g_emas.GetEMA(6); b7[idx] = g_emas.GetEMA(7);
       b8[idx] = g_emas.GetEMA(8); b9[idx] = g_emas.GetEMA(9);
       // Oper
       b10[idx] = g_emas.GetEMA(10); b11[idx] = g_emas.GetEMA(11); b12[idx] = g_emas.GetEMA(12); b13[idx] = g_emas.GetEMA(13);
       b14[idx] = g_emas.GetEMA(14); b15[idx] = g_emas.GetEMA(15); b16[idx] = g_emas.GetEMA(16); b17[idx] = g_emas.GetEMA(17);
       b18[idx] = g_emas.GetEMA(18); b19[idx] = g_emas.GetEMA(19);
       // Macro
       b20[idx] = g_emas.GetEMA(20); b21[idx] = g_emas.GetEMA(21); b22[idx] = g_emas.GetEMA(22); b23[idx] = g_emas.GetEMA(23);
       b24[idx] = g_emas.GetEMA(24); b25[idx] = g_emas.GetEMA(25); b26[idx] = g_emas.GetEMA(26); b27[idx] = g_emas.GetEMA(27);
       b28[idx] = g_emas.GetEMA(28); b29[idx] = g_emas.GetEMA(29);
       // Bias
       b30[idx] = g_emas.GetEMA(30);

       // 7. Update HUD (Only for the last bar to avoid flicker)
       if (i == total - 1) {
           CFEAT::SResult res = g_feat.GetResult();
           g_vis.Update(res, g_fsm.GetStateString());
       }

       // 8. Export Data
       // Warning: Exporting every historical bar via ZMQ can freeze the terminal if Lookback is huge.
       // However, we need the dataset. We compromise by sending it.
       // To be safe, we might only send if ExportData is true.
       if(ExportData) {
          // Optional: throttling or bulk send could be added here.
          // For now, we strictly follow the instruction to calculate and export.

          SBarDataExport data;
          data.time = time[i];
          data.open = open[i];
          data.high = high[i];
          data.low = low[i];
          data.close = close[i];
          data.volume = (double)tick[i];

          SEMAGroupMetrics mic = g_emas.GetMicroMetrics();
          SEMAGroupMetrics opr = g_emas.GetOperationalMetrics();
          SEMAGroupMetrics mac = g_emas.GetMacroMetrics();
          SFanMetrics fan = g_emas.GetFanMetrics();
          CFEAT::SResult feat = g_feat.GetResult();
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
          data.isAgainstH4 = false;
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

          // Only send via ZMQ if:
          // 1. It is the initial calculation (prev == 0) -> DUMP HISTORY
          // 2. It is a live tick (i == total - 1) -> STREAM UPDATES
          // This prevents re-sending the entire history on every tick while still providing the initial dataset.
          if (prev == 0 || i >= total - 1) {
             g_io.SendFeaturesZMQ(data, _Symbol);
          }
       }
   }
   
   return total;
}
