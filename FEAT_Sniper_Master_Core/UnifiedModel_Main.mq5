//+------------------------------------------------------------------+
//|                                          UnifiedModel_Main.mq5 |
//|                    Unified Institutional Model                  |
//|              30 EMAs + FEAT + Liquidity + FSM                   |
//+------------------------------------------------------------------+
#property copyright "Institutional Trading Systems"
#property link      "https://github.com/SrAndres629/feat_sniper_mcp"
#property version   "1.00"
#property description "Unified Institutional Model: 30 EMAs + FEAT + Liquidity"
#property description "Framework de lectura total del mercado"
#property indicator_chart_window
#property indicator_buffers 12
#property indicator_plots   12

//--- EMA plots (12 EMAs for visual ribbon)
#property indicator_label1  "EMA3"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrYellow
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1

#property indicator_label2  "EMA5"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrYellow
#property indicator_style2  STYLE_SOLID
#property indicator_width2  1

#property indicator_label3  "EMA8"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrGold
#property indicator_style3  STYLE_SOLID
#property indicator_width3  1

#property indicator_label4  "EMA13"
#property indicator_type4   DRAW_LINE
#property indicator_color4  clrGold
#property indicator_style4  STYLE_SOLID
#property indicator_width4  1

#property indicator_label5  "EMA21"
#property indicator_type5   DRAW_LINE
#property indicator_color5  clrOrange
#property indicator_style5  STYLE_SOLID
#property indicator_width5  1

#property indicator_label6  "EMA34"
#property indicator_type6   DRAW_LINE
#property indicator_color6  clrOrange
#property indicator_style6  STYLE_SOLID
#property indicator_width6  1

#property indicator_label7  "EMA55"
#property indicator_type7   DRAW_LINE
#property indicator_color7  clrOrangeRed
#property indicator_style7  STYLE_SOLID
#property indicator_width7  1

#property indicator_label8  "EMA89"
#property indicator_type8   DRAW_LINE
#property indicator_color8  clrOrangeRed
#property indicator_style8  STYLE_SOLID
#property indicator_width8  1

#property indicator_label9  "EMA144"
#property indicator_type9   DRAW_LINE
#property indicator_color9  clrRed
#property indicator_style9  STYLE_SOLID
#property indicator_width9  2

#property indicator_label10 "EMA233"
#property indicator_type10  DRAW_LINE
#property indicator_color10 clrRed
#property indicator_style10 STYLE_SOLID
#property indicator_width10 2

#property indicator_label11 "EMA377"
#property indicator_type11  DRAW_LINE
#property indicator_color11 clrDarkRed
#property indicator_style11 STYLE_SOLID
#property indicator_width11 2

#property indicator_label12 "EMA610"
#property indicator_type12  DRAW_LINE
#property indicator_color12 clrDarkRed
#property indicator_style12 STYLE_SOLID
#property indicator_width12 2

//+------------------------------------------------------------------+
//| INCLUDES                                                         |
//+------------------------------------------------------------------+
#include <UnifiedModel\CEMAs.mqh>
#include <UnifiedModel\CFEAT.mqh>
#include <UnifiedModel\CLiquidity.mqh>
#include <UnifiedModel\CFSM.mqh>
#include <UnifiedModel\CVisuals.mqh>
#include <UnifiedModel\CInterop.mqh>

//+------------------------------------------------------------------+
//| DLL IMPORTS                                                      |
//+------------------------------------------------------------------+
#import "shell32.dll"
int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);
#import

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                 |
//+------------------------------------------------------------------+
input group "=== General Settings ==="
input int                 ATR_Period         = 14;            // ATR Period
input int                 LookbackBars       = 100;           // Lookback Bars

input group "=== Bridge Integration ==="
input bool                AutoStart_Bridge   = true;          // Auto-Start Python Bridge
input string              Bridge_Bat_Path    = "c:\\Users\\acord\\OneDrive\\Desktop\\Bot\\feat_sniper_mcp\\FEAT_Sniper_Master_Core\\Python\\start_bridge.bat"; // Path to Bridge .bat

input group "=== Sessions (Server Time HH:MM) ==="
input string              Session_Asia_Start = "00:00";       // Asia Start
input string              Session_Asia_End   = "09:00";       // Asia End
input string              Session_London_Start = "08:00";     // London Start
input string              Session_London_End = "17:00";       // London End
input string              Session_NY_Start   = "13:00";       // NY Start
input string              Session_NY_End     = "22:00";       // NY End

input group "=== Killzones (Server Time HH:MM) ==="
input string              KZ_London_Start    = "08:00";       // London KZ Start
input string              KZ_London_End      = "10:00";       // London KZ End
input string              KZ_NY_Start        = "13:00";       // NY KZ Start
input string              KZ_NY_End          = "15:00";       // NY KZ End

input group "=== FEAT Thresholds ==="
input double              FEAT_Curvature     = 0.3;           // Curvature Threshold
input double              FEAT_Compression   = 0.7;           // Compression Threshold
input double              FEAT_Accel         = 0.5;           // Acceleration Threshold
input double              FEAT_Gap           = 1.5;           // Gap Threshold (ATR mult)

input group "=== FSM Settings ==="
input double              FSM_AccumCompress  = 0.7;           // Accumulation Compression
input double              FSM_ExpanSlope     = 0.3;           // Expansion Slope
input double              FSM_DistMomentum   = -0.2;          // Distribution Momentum
input double              FSM_ResetSpeed     = 2.0;           // Reset Speed
input int                 FSM_MinBars        = 3;             // Min Bars in State
input int                 FSM_BufferSize     = 100;           // Percentile Buffer Size

input group "=== Liquidity ==="
input double              LIQ_EqualPips      = 5.0;           // Equal Level Tolerance (pips)
input double              LIQ_FVGMinATR      = 0.5;           // Min FVG Size (ATR mult)
input int                 LIQ_MaxLevels      = 50;            // Max Liquidity Levels

input group "=== Visualization ==="
input bool                Draw_EMAs          = true;          // Draw EMA Ribbon
input bool                Draw_Panel         = true;          // Draw Info Panel
input bool                Draw_Liquidity     = true;          // Draw Liquidity Levels
input bool                Draw_FEAT          = true;          // Draw FEAT Info
input int                 Panel_X            = 10;            // Panel X Position
input int                 Panel_Y            = 30;            // Panel Y Position

input group "=== Python Interop ==="
input bool                Use_Python_Calib   = false;         // Use Python Calibration
input bool                Export_Data        = false;         // Export Data for Training
input string              Python_DataPath    = "UnifiedModel\\"; // Data Path

input group "=== Debug ==="
input bool                Debug_Mode         = false;         // Enable Debug Logs

//+------------------------------------------------------------------+
//| GLOBAL OBJECTS                                                   |
//+------------------------------------------------------------------+
CEMAs       g_emas;
CFEAT       g_feat;
CLiquidity  g_liquidity;
CFSM        g_fsm;
CVisuals    g_visuals;
CInterop    g_interop;

//+------------------------------------------------------------------+
//| BUFFERS                                                          |
//+------------------------------------------------------------------+
double      g_emaBuffer1[], g_emaBuffer2[], g_emaBuffer3[], g_emaBuffer4[];
double      g_emaBuffer5[], g_emaBuffer6[], g_emaBuffer7[], g_emaBuffer8[];
double      g_emaBuffer9[], g_emaBuffer10[], g_emaBuffer11[], g_emaBuffer12[];

double      g_emaBuffer9[], g_emaBuffer10[], g_emaBuffer11[], g_emaBuffer12[];

// NOTE: g_emaHandles removed to optimize redundancy. Handles are managed by CEMAs.

//+------------------------------------------------------------------+
//| INITIALIZATION                                                   |
//+------------------------------------------------------------------+
int OnInit() {
   // Warning about DLLs
   if(AutoStart_Bridge && !MQLInfoInteger(MQL_DLLS_ALLOWED)) {
      Alert("[UnifiedModel] Critical: DLL imports must be allowed for Auto-Start Bridge!");
      Print("[UnifiedModel] Please enable 'Allow DLL imports' in Indicator Properties > Dependencies");
   }

   // Auto-Start Bridge
   if(AutoStart_Bridge && MQLInfoInteger(MQL_DLLS_ALLOWED)) {
      string operation = "open";
      string file = Bridge_Bat_Path;
      string params = "";
      string directory = "";
      
      string parts[];
      if(StringSplit(file, '\\', parts) > 0) {
         for(int i=0; i<ArraySize(parts)-1; i++) {
             directory += parts[i] + "\\";
         }
      }
      
      if(!GlobalVariableCheck("UM_Bridge_Started")) {
         ShellExecuteW(0, operation, file, params, directory, 2); // 2 = Minimized
         GlobalVariableTemp("UM_Bridge_Started"); // Session-only variable
         GlobalVariableSet("UM_Bridge_Started", 1);
         Print("[UnifiedModel] Bridge Auto-Started: ", file);
      }
   }

   // Set buffers
   SetIndexBuffer(0, g_emaBuffer1, INDICATOR_DATA);
   SetIndexBuffer(1, g_emaBuffer2, INDICATOR_DATA);
   SetIndexBuffer(2, g_emaBuffer3, INDICATOR_DATA);
   SetIndexBuffer(3, g_emaBuffer4, INDICATOR_DATA);
   SetIndexBuffer(4, g_emaBuffer5, INDICATOR_DATA);
   SetIndexBuffer(5, g_emaBuffer6, INDICATOR_DATA);
   SetIndexBuffer(6, g_emaBuffer7, INDICATOR_DATA);
   SetIndexBuffer(7, g_emaBuffer8, INDICATOR_DATA);
   SetIndexBuffer(8, g_emaBuffer9, INDICATOR_DATA);
   SetIndexBuffer(9, g_emaBuffer10, INDICATOR_DATA);
   SetIndexBuffer(10, g_emaBuffer11, INDICATOR_DATA);
   SetIndexBuffer(11, g_emaBuffer12, INDICATOR_DATA);
   
   // Initialize EMAs engine (Creates handles internally)
   if(!g_emas.Init(_Symbol, PERIOD_CURRENT, ATR_Period)) {
      Print("[UnifiedModel] Failed to initialize CEMAs");
      return INIT_FAILED;
   }
   
   // Initialize FEAT
   g_feat.SetEMAs(&g_emas);
   g_feat.SetThresholds(FEAT_Curvature, FEAT_Compression, FEAT_Accel, FEAT_Gap);
   
   // Parse session times
   int asiaS = ParseTimeToMinutes(Session_Asia_Start);
   int asiaE = ParseTimeToMinutes(Session_Asia_End);
   int londonS = ParseTimeToMinutes(Session_London_Start);
   int londonE = ParseTimeToMinutes(Session_London_End);
   int nyS = ParseTimeToMinutes(Session_NY_Start);
   int nyE = ParseTimeToMinutes(Session_NY_End);
   int kzLS = ParseTimeToMinutes(KZ_London_Start);
   int kzLE = ParseTimeToMinutes(KZ_London_End);
   int kzNS = ParseTimeToMinutes(KZ_NY_Start);
   int kzNE = ParseTimeToMinutes(KZ_NY_End);
   
   g_feat.SetSessionTimes(asiaS, asiaE, londonS, londonE, nyS, nyE);
   g_feat.SetKillzones(kzLS, kzLE, kzNS, kzNE);
   
   // Initialize Liquidity
   g_liquidity.Init(_Symbol, PERIOD_CURRENT, LIQ_MaxLevels, LookbackBars, LIQ_EqualPips, LIQ_FVGMinATR);
   
   // Initialize FSM
   g_fsm.SetComponents(&g_emas, &g_feat, &g_liquidity);
   g_fsm.SetBufferSize(FSM_BufferSize);
   
   SFSMThresholds fsmThresh;
   fsmThresh.accumulationCompression = FSM_AccumCompress;
   fsmThresh.expansionSlope = FSM_ExpanSlope;
   fsmThresh.distributionMomentum = FSM_DistMomentum;
   fsmThresh.resetSpeed = FSM_ResetSpeed;
   fsmThresh.minBarsInState = FSM_MinBars;
   fsmThresh.hysteresisMargin = 0.1;
   g_fsm.SetThresholds(fsmThresh);
   
   // Initialize Python interop
   g_interop.SetEnabled(Use_Python_Calib || Export_Data);
   g_interop.SetDataPath(Python_DataPath);
   
   if(Use_Python_Calib) {
      if(g_interop.LoadCalibrationAuto(_Symbol, PERIOD_CURRENT)) {
         SFSMThresholds pyThresh;
         if(g_interop.ExportToFSMThresholds(pyThresh)) {
            g_fsm.SetThresholds(pyThresh);
            Print("[UnifiedModel] Using Python calibration");
         }
      }
   }
   
   // Initialize Visuals
   g_visuals.Init("UM_", ChartID());
   g_visuals.SetComponents(&g_emas, &g_feat, &g_liquidity, &g_fsm);
   g_visuals.SetDrawOptions(Draw_EMAs, Draw_Panel, Draw_Liquidity, Draw_FEAT);
   g_visuals.SetPanelPosition(Panel_X, Panel_Y);
   
   if(Debug_Mode) Print("[UnifiedModel] Initialized on ", _Symbol, " ", EnumToString(PERIOD_CURRENT));
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| DEINITIALIZATION                                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   g_emas.Deinit();
   g_visuals.Clear();
   
   
   if(Debug_Mode) Print("[UnifiedModel] Deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| MAIN CALCULATION                                                 |
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
                const int &spread[]) {
   
   if(rates_total < LookbackBars + 1) return 0;
   
   uint startTime = GetTickCount();
   
   // Calculate from start or from last calculated
   int calcFrom = (prev_calculated == 0) ? rates_total - LookbackBars : prev_calculated - 1;
   if(calcFrom < 0) calcFrom = 0;
   
   // Copy EMA buffers
   double buffer[];
   ArraySetAsSeries(buffer, true);
   
   for(int shift = rates_total - 1 - calcFrom; shift >= 0; shift--) {
      int idx = rates_total - 1 - shift;
      
      // Copy each EMA from CEMAs handles to Indicator Buffers
      if(CopyBuffer(g_emas.GetHandle(0), 0, shift, 1, buffer) > 0) g_emaBuffer1[idx] = buffer[0];
      if(CopyBuffer(g_emas.GetHandle(1), 0, shift, 1, buffer) > 0) g_emaBuffer2[idx] = buffer[0];
      if(CopyBuffer(g_emas.GetHandle(2), 0, shift, 1, buffer) > 0) g_emaBuffer3[idx] = buffer[0];
      if(CopyBuffer(g_emas.GetHandle(3), 0, shift, 1, buffer) > 0) g_emaBuffer4[idx] = buffer[0];
      if(CopyBuffer(g_emas.GetHandle(4), 0, shift, 1, buffer) > 0) g_emaBuffer5[idx] = buffer[0];
      if(CopyBuffer(g_emas.GetHandle(5), 0, shift, 1, buffer) > 0) g_emaBuffer6[idx] = buffer[0];
      if(CopyBuffer(g_emas.GetHandle(6), 0, shift, 1, buffer) > 0) g_emaBuffer7[idx] = buffer[0];
      if(CopyBuffer(g_emas.GetHandle(7), 0, shift, 1, buffer) > 0) g_emaBuffer8[idx] = buffer[0];
      if(CopyBuffer(g_emas.GetHandle(8), 0, shift, 1, buffer) > 0) g_emaBuffer9[idx] = buffer[0];
      if(CopyBuffer(g_emas.GetHandle(9), 0, shift, 1, buffer) > 0) g_emaBuffer10[idx] = buffer[0];
      if(CopyBuffer(g_emas.GetHandle(10), 0, shift, 1, buffer) > 0) g_emaBuffer11[idx] = buffer[0];
      if(CopyBuffer(g_emas.GetHandle(11), 0, shift, 1, buffer) > 0) g_emaBuffer12[idx] = buffer[0];
   }
   
   // Only calculate layers on bar close (efficiency)
   static datetime lastBarTime = 0;
   datetime currentBarTime = time[rates_total - 1];
   bool isNewBar = (currentBarTime != lastBarTime);
   
   if(isNewBar) {
      lastBarTime = currentBarTime;
      
      // Set as series for layer calculations
      ArraySetAsSeries(time, true);
      ArraySetAsSeries(open, true);
      ArraySetAsSeries(high, true);
      ArraySetAsSeries(low, true);
      ArraySetAsSeries(close, true);
      ArraySetAsSeries(tick_volume, true);
      
      // Calculate EMAs engine
      g_emas.Calculate(0);
      
      // Calculate FEAT
      g_feat.Calculate(PERIOD_CURRENT, time[0]);
      
      // Calculate Liquidity
      g_liquidity.SetATR(g_emas.GetATR());
      g_liquidity.Calculate(high, low, open, close, time, MathMin(rates_total, LookbackBars), close[0]);
      
      // Calculate FSM
      double prevClose = (rates_total > 1) ? close[1] : close[0];
      g_fsm.Calculate(close[0], prevClose, (double)tick_volume[0]);
      
      // Data Export Logic (Bridge Integration)
      if(Export_Data) {
         SFSMMetrics m = g_fsm.GetMetrics();
         string exportFile = "UnifiedModel_Export_" + _Symbol + "_" + EnumToString(PERIOD_CURRENT) + ".csv";
         
         g_interop.AppendBarData(exportFile, time[0], open[0], high[0], low[0], close[0],
                                 m.effort, m.result, m.compression, m.slope, m.speed,
                                 g_fsm.GetConfidence(), g_fsm.GetStateString());
      }
      
      // Draw visuals
      g_visuals.Draw(time[0], close[0]);
      
      // Performance logging
      if(Debug_Mode) {
         uint elapsed = GetTickCount() - startTime;
         if(elapsed > 1) Print("[UnifiedModel] OnCalculate: ", elapsed, "ms | State: ", g_fsm.GetStateString());
      }
   }
   
   return rates_total;
}

//+------------------------------------------------------------------+
//| Parse Time String to Minutes from Midnight                       |
//+------------------------------------------------------------------+
int ParseTimeToMinutes(string timeStr) {
   string parts[];
   int count = StringSplit(timeStr, ':', parts);
   if(count < 2) return 0;
   
   int hour = (int)StringToInteger(parts[0]);
   int minute = (int)StringToInteger(parts[1]);
   
   return hour * 60 + minute;
}
//+------------------------------------------------------------------+
