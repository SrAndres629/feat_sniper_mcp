//+------------------------------------------------------------------+
//|                                           InstitutionalPVP.mq5 |
//|                    Institutional Volume Profile Indicator       |
//|                 PVP + Temporal Cycles + Liquidity + State ML    |
//+------------------------------------------------------------------+
#property copyright "Institutional Trading Systems"
#property link      "https://github.com/SrAndres629/feat_sniper_mcp"
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

//+------------------------------------------------------------------+
//| ENUMERATIONS                                                     |
//+------------------------------------------------------------------+
enum ENUM_AGGREGATION_MODE {
   AGG_CLOSE,              // Close Price Only
   AGG_HIGH_LOW_DIST       // High-Low Distribution
};

enum ENUM_MARKET_STATE {
   STATE_ACCUMULATION,     // Accumulation
   STATE_EXPANSION,        // Expansion
   STATE_DISTRIBUTION,     // Distribution
   STATE_MANIPULATION      // Manipulation
};

enum ENUM_PROFILE_SHAPE {
   SHAPE_P,                // P-Shape (Buy Pressure)
   SHAPE_B,                // B-Shape (Balance)
   SHAPE_D,                // D-Shape (Sell Pressure)
   SHAPE_UNDEFINED         // Undefined
};

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                 |
//+------------------------------------------------------------------+
input group "=== HTF & PVP Settings ==="
input ENUM_TIMEFRAMES     HTF_Timeframe       = PERIOD_H4;    // HTF Timeframe
input double              ValueAreaPercent    = 70.0;         // Value Area % (50-90)
input int                 NumBins             = 100;          // Number of Price Bins (20-500)
input int                 LookbackBars        = 50;           // Lookback Bars (10-500)
input ENUM_AGGREGATION_MODE AggregationMode   = AGG_HIGH_LOW_DIST; // Volume Aggregation

input group "=== Sessions (Server Time HH:MM) ==="
input string              SessionAsia_Start   = "00:00";      // Asia Start
input string              SessionAsia_End     = "09:00";      // Asia End
input string              SessionLondon_Start = "08:00";      // London Start
input string              SessionLondon_End   = "17:00";      // London End
input string              SessionNY_Start     = "13:00";      // NY Start
input string              SessionNY_End       = "22:00";      // NY End

input group "=== Killzones (Server Time HH:MM) ==="
input string              KZ_LondonOpen_Start = "08:00";      // London Open KZ Start
input string              KZ_LondonOpen_End   = "10:00";      // London Open KZ End
input string              KZ_NYOpen_Start     = "13:00";      // NY Open KZ Start
input string              KZ_NYOpen_End       = "15:00";      // NY Open KZ End

input group "=== Score Weights (sum = 1.0) ==="
input double              ScoreWeight_Volume  = 0.25;         // Volume Weight
input double              ScoreWeight_Accept  = 0.25;         // Acceptance Weight
input double              ScoreWeight_POC     = 0.25;         // POC Shift Weight
input double              ScoreWeight_Shape   = 0.25;         // Profile Shape Weight

input group "=== Thresholds ==="
input int                 AcceptanceThreshold = 3;            // Acceptance Minutes
input double              VolatilityThreshold = 1.5;          // Volatility ATR Mult
input double              HVN_Threshold       = 1.5;          // HVN Relative Threshold
input double              LVN_Threshold       = 0.3;          // LVN Relative Threshold

input group "=== Visualization ==="
input bool                DrawPOC             = true;         // Draw POC Line
input bool                DrawVA              = true;         // Draw Value Area
input bool                DrawHVN_LVN         = true;         // Draw HVN/LVN
input bool                DrawPanel           = true;         // Draw Info Panel
input color               POC_Color           = clrMagenta;   // POC Color
input color               VA_Color            = clrGray;      // VA Color
input color               HVN_Color           = clrLimeGreen; // HVN Color
input color               LVN_Color           = clrCrimson;   // LVN Color

input group "=== Debug ==="
input bool                DebugMode           = false;        // Enable Debug Logs

//+------------------------------------------------------------------+
//| STRUCTURES                                                       |
//+------------------------------------------------------------------+
struct SPriceLevel {
   double   price;
   double   volume;
   bool     isHVN;
   bool     isLVN;
};

struct SPVPResult {
   double   poc;
   double   vah;
   double   val;
   double   totalVolume;
   double   rangeHigh;
   double   rangeLow;
   double   binSize;
   SPriceLevel levels[];
   int      pocBin;
   int      vahBin;
   int      valBin;
};

struct SSessionTime {
   int      startHour;
   int      startMin;
   int      endHour;
   int      endMin;
};

struct SMarketContext {
   SPVPResult        pvpHTF;
   SPVPResult        pvpSession;
   SPVPResult        pvpKillzone;
   ENUM_MARKET_STATE state;
   ENUM_PROFILE_SHAPE shape;
   double            score;
   double            prevPOC;
   double            deltaPOC;
   string            activeSession;
   string            activeKillzone;
   datetime          lastCycleTime;
   bool              isValid;
};

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                 |
//+------------------------------------------------------------------+
SMarketContext    g_context;
SSessionTime      g_asia, g_london, g_ny;
SSessionTime      g_kzLondon, g_kzNY;
double            g_volumeHistogram[];
double            g_scoreEMA;
int               g_htfHandle;
string            g_objPrefix;
datetime          g_lastHTFBar;
datetime          g_lastSessionStart;
double            g_tickSize;
int               g_digits;

//+------------------------------------------------------------------+
//| INITIALIZATION                                                   |
//+------------------------------------------------------------------+
int OnInit() {
   // Validate inputs
   if(!ValidateInputs()) return INIT_PARAMETERS_INCORRECT;
   
   // Parse session times
   if(!ParseTimeString(SessionAsia_Start, g_asia.startHour, g_asia.startMin)) return INIT_PARAMETERS_INCORRECT;
   if(!ParseTimeString(SessionAsia_End, g_asia.endHour, g_asia.endMin)) return INIT_PARAMETERS_INCORRECT;
   if(!ParseTimeString(SessionLondon_Start, g_london.startHour, g_london.startMin)) return INIT_PARAMETERS_INCORRECT;
   if(!ParseTimeString(SessionLondon_End, g_london.endHour, g_london.endMin)) return INIT_PARAMETERS_INCORRECT;
   if(!ParseTimeString(SessionNY_Start, g_ny.startHour, g_ny.startMin)) return INIT_PARAMETERS_INCORRECT;
   if(!ParseTimeString(SessionNY_End, g_ny.endHour, g_ny.endMin)) return INIT_PARAMETERS_INCORRECT;
   if(!ParseTimeString(KZ_LondonOpen_Start, g_kzLondon.startHour, g_kzLondon.startMin)) return INIT_PARAMETERS_INCORRECT;
   if(!ParseTimeString(KZ_LondonOpen_End, g_kzLondon.endHour, g_kzLondon.endMin)) return INIT_PARAMETERS_INCORRECT;
   if(!ParseTimeString(KZ_NYOpen_Start, g_kzNY.startHour, g_kzNY.startMin)) return INIT_PARAMETERS_INCORRECT;
   if(!ParseTimeString(KZ_NYOpen_End, g_kzNY.endHour, g_kzNY.endMin)) return INIT_PARAMETERS_INCORRECT;
   
   // Initialize globals
   g_objPrefix = "IPVP_" + IntegerToString(ChartID()) + "_";
   g_tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   g_digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   g_scoreEMA = 50.0;
   g_lastHTFBar = 0;
   g_lastSessionStart = 0;
   
   ArrayResize(g_volumeHistogram, NumBins);
   ArrayInitialize(g_volumeHistogram, 0);
   
   ZeroMemory(g_context);
   g_context.state = STATE_ACCUMULATION;
   g_context.score = 50.0;
   
   if(DebugMode) Print("[IPVP] Initialized. Symbol=", _Symbol, " HTF=", EnumToString(HTF_Timeframe));
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| DEINITIALIZATION                                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   ObjectsDeleteAll(0, g_objPrefix);
   if(DebugMode) Print("[IPVP] Deinitialized. Reason=", reason);
}

//+------------------------------------------------------------------+
//| MAIN CALCULATION                                                 |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated,
                const datetime &time[], const double &open[],
                const double &high[], const double &low[],
                const double &close[], const long &tick_volume[],
                const long &volume[], const int &spread[]) {
   
   if(rates_total < LookbackBars + 1) return 0;
   
   uint startTime = GetTickCount();
   
   // Set as series
   ArraySetAsSeries(time, true);
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(tick_volume, true);
   
   datetime currentTime = time[0];
   
   // Check for new HTF bar
   datetime htfBarTime = iTime(_Symbol, HTF_Timeframe, 0);
   bool newHTFBar = (htfBarTime != g_lastHTFBar);
   if(newHTFBar) {
      g_context.prevPOC = g_context.pvpHTF.poc;
      g_lastHTFBar = htfBarTime;
   }
   
   // Detect active session and killzone
   DetectActiveCycle(currentTime);
   
   // Calculate PVP for HTF context
   CalculatePVP(g_context.pvpHTF, high, low, close, tick_volume, 0, LookbackBars);
   
   // Calculate delta POC
   if(g_context.prevPOC > 0) {
      g_context.deltaPOC = (g_context.pvpHTF.poc - g_context.prevPOC) / g_tickSize;
   }
   
   // Detect profile shape
   g_context.shape = DetectProfileShape(g_context.pvpHTF);
   
   // Classify market state
   ClassifyMarketState(close[0]);
   
   // Draw visuals
   DrawVisuals();
   
   // Performance logging
   if(DebugMode) {
      uint elapsed = GetTickCount() - startTime;
      if(elapsed > 1) Print("[IPVP] OnCalculate took ", elapsed, "ms");
   }
   
   return rates_total;
}

//+------------------------------------------------------------------+
//| INPUT VALIDATION                                                 |
//+------------------------------------------------------------------+
bool ValidateInputs() {
   if(ValueAreaPercent < 50 || ValueAreaPercent > 90) {
      Print("[IPVP] Error: ValueAreaPercent must be 50-90");
      return false;
   }
   if(NumBins < 20 || NumBins > 500) {
      Print("[IPVP] Error: NumBins must be 20-500");
      return false;
   }
   if(LookbackBars < 10 || LookbackBars > 500) {
      Print("[IPVP] Error: LookbackBars must be 10-500");
      return false;
   }
   double weightSum = ScoreWeight_Volume + ScoreWeight_Accept + ScoreWeight_POC + ScoreWeight_Shape;
   if(MathAbs(weightSum - 1.0) > 0.01) {
      Print("[IPVP] Warning: Score weights sum to ", weightSum, " (should be 1.0)");
   }
   if(HVN_Threshold <= 1.0) {
      Print("[IPVP] Error: HVN_Threshold must be > 1.0");
      return false;
   }
   if(LVN_Threshold >= 1.0 || LVN_Threshold <= 0) {
      Print("[IPVP] Error: LVN_Threshold must be 0 < x < 1.0");
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| PARSE TIME STRING (HH:MM)                                        |
//+------------------------------------------------------------------+
bool ParseTimeString(const string timeStr, int &hour, int &minute) {
   string parts[];
   int count = StringSplit(timeStr, ':', parts);
   if(count != 2) {
      Print("[IPVP] Error: Invalid time format: ", timeStr);
      return false;
   }
   hour = (int)StringToInteger(parts[0]);
   minute = (int)StringToInteger(parts[1]);
   if(hour < 0 || hour > 23 || minute < 0 || minute > 59) {
      Print("[IPVP] Error: Invalid time values: ", timeStr);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| DETECT ACTIVE CYCLE (Session/Killzone)                           |
//+------------------------------------------------------------------+
void DetectActiveCycle(datetime currentTime) {
   MqlDateTime dt;
   TimeToStruct(currentTime, dt);
   int currentMinutes = dt.hour * 60 + dt.min;
   
   // Reset session names
   g_context.activeSession = "";
   g_context.activeKillzone = "";
   
   // Check sessions
   if(IsInTimeRange(currentMinutes, g_asia)) g_context.activeSession = "Asia";
   else if(IsInTimeRange(currentMinutes, g_london)) g_context.activeSession = "London";
   else if(IsInTimeRange(currentMinutes, g_ny)) g_context.activeSession = "NY";
   
   // Check killzones
   if(IsInTimeRange(currentMinutes, g_kzLondon)) g_context.activeKillzone = "LO";
   else if(IsInTimeRange(currentMinutes, g_kzNY)) g_context.activeKillzone = "NYO";
}

//+------------------------------------------------------------------+
//| CHECK IF TIME IS IN RANGE                                        |
//+------------------------------------------------------------------+
bool IsInTimeRange(int currentMinutes, const SSessionTime &session) {
   int startMinutes = session.startHour * 60 + session.startMin;
   int endMinutes = session.endHour * 60 + session.endMin;
   
   if(startMinutes <= endMinutes) {
      return (currentMinutes >= startMinutes && currentMinutes < endMinutes);
   } else {
      // Overnight session
      return (currentMinutes >= startMinutes || currentMinutes < endMinutes);
   }
}

//+------------------------------------------------------------------+
//| CALCULATE PVP (Volume Profile by Price)                          |
//+------------------------------------------------------------------+
void CalculatePVP(SPVPResult &result, const double &high[], const double &low[],
                  const double &close[], const long &tick_volume[],
                  int startBar, int numBars) {
   
   // Find price range
   double rangeHigh = high[startBar];
   double rangeLow = low[startBar];
   
   for(int i = startBar; i < startBar + numBars; i++) {
      if(high[i] > rangeHigh) rangeHigh = high[i];
      if(low[i] < rangeLow) rangeLow = low[i];
   }
   
   // Handle edge case
   if(rangeHigh <= rangeLow) {
      rangeHigh = rangeLow + g_tickSize * 10;
   }
   
   result.rangeHigh = rangeHigh;
   result.rangeLow = rangeLow;
   result.binSize = (rangeHigh - rangeLow) / NumBins;
   
   // Initialize histogram
   ArrayInitialize(g_volumeHistogram, 0);
   
   // Accumulate volume
   result.totalVolume = 0;
   
   for(int i = startBar; i < startBar + numBars; i++) {
      double vol = (double)tick_volume[i];
      result.totalVolume += vol;
      
      if(AggregationMode == AGG_CLOSE) {
         // Assign all volume to close price bin
         int bin = PriceToBin(close[i], result.rangeLow, result.binSize);
         if(bin >= 0 && bin < NumBins) g_volumeHistogram[bin] += vol;
      } else {
         // Distribute volume across high-low range
         int binLow = PriceToBin(low[i], result.rangeLow, result.binSize);
         int binHigh = PriceToBin(high[i], result.rangeLow, result.binSize);
         
         binLow = MathMax(0, MathMin(NumBins-1, binLow));
         binHigh = MathMax(0, MathMin(NumBins-1, binHigh));
         
         int numBinsInRange = binHigh - binLow + 1;
         double volPerBin = vol / numBinsInRange;
         
         for(int b = binLow; b <= binHigh; b++) {
            g_volumeHistogram[b] += volPerBin;
         }
      }
   }
   
   // Find POC (bin with max volume)
   double maxVol = 0;
   result.pocBin = 0;
   for(int i = 0; i < NumBins; i++) {
      if(g_volumeHistogram[i] > maxVol) {
         maxVol = g_volumeHistogram[i];
         result.pocBin = i;
      }
   }
   result.poc = BinToPrice(result.pocBin, result.rangeLow, result.binSize);
   
   // Calculate Value Area
   CalculateValueArea(result);
   
   // Detect HVN/LVN
   DetectVolumeNodes(result);
   
   result.isValid = true;
}

//+------------------------------------------------------------------+
//| PRICE TO BIN INDEX                                               |
//+------------------------------------------------------------------+
int PriceToBin(double price, double rangeLow, double binSize) {
   if(binSize <= 0) return 0;
   return (int)MathFloor((price - rangeLow) / binSize);
}

//+------------------------------------------------------------------+
//| BIN INDEX TO PRICE (center of bin)                               |
//+------------------------------------------------------------------+
double BinToPrice(int bin, double rangeLow, double binSize) {
   return rangeLow + (bin + 0.5) * binSize;
}

//+------------------------------------------------------------------+
//| CALCULATE VALUE AREA                                             |
//+------------------------------------------------------------------+
void CalculateValueArea(SPVPResult &result) {
   double targetVolume = result.totalVolume * ValueAreaPercent / 100.0;
   double accumulatedVolume = g_volumeHistogram[result.pocBin];
   
   int upperBin = result.pocBin;
   int lowerBin = result.pocBin;
   
   // Expand from POC until we reach target volume
   while(accumulatedVolume < targetVolume && (upperBin < NumBins-1 || lowerBin > 0)) {
      double upperVol = (upperBin < NumBins-1) ? g_volumeHistogram[upperBin+1] : 0;
      double lowerVol = (lowerBin > 0) ? g_volumeHistogram[lowerBin-1] : 0;
      
      if(upperVol >= lowerVol && upperBin < NumBins-1) {
         upperBin++;
         accumulatedVolume += g_volumeHistogram[upperBin];
      } else if(lowerBin > 0) {
         lowerBin--;
         accumulatedVolume += g_volumeHistogram[lowerBin];
      } else if(upperBin < NumBins-1) {
         upperBin++;
         accumulatedVolume += g_volumeHistogram[upperBin];
      } else {
         break;
      }
   }
   
   result.vahBin = upperBin;
   result.valBin = lowerBin;
   result.vah = BinToPrice(upperBin, result.rangeLow, result.binSize) + result.binSize/2;
   result.val = BinToPrice(lowerBin, result.rangeLow, result.binSize) - result.binSize/2;
}

//+------------------------------------------------------------------+
//| DETECT HVN/LVN                                                   |
//+------------------------------------------------------------------+
void DetectVolumeNodes(SPVPResult &result) {
   // Calculate mean volume
   double meanVol = 0;
   int validBins = 0;
   for(int i = 0; i < NumBins; i++) {
      if(g_volumeHistogram[i] > 0) {
         meanVol += g_volumeHistogram[i];
         validBins++;
      }
   }
   if(validBins > 0) meanVol /= validBins;
   
   double hvnThresh = meanVol * HVN_Threshold;
   double lvnThresh = meanVol * LVN_Threshold;
   
   // Resize levels array
   ArrayResize(result.levels, NumBins);
   
   for(int i = 0; i < NumBins; i++) {
      result.levels[i].price = BinToPrice(i, result.rangeLow, result.binSize);
      result.levels[i].volume = g_volumeHistogram[i];
      result.levels[i].isHVN = (g_volumeHistogram[i] >= hvnThresh);
      result.levels[i].isLVN = (g_volumeHistogram[i] > 0 && g_volumeHistogram[i] <= lvnThresh);
   }
}

//+------------------------------------------------------------------+
//| DETECT PROFILE SHAPE                                             |
//+------------------------------------------------------------------+
ENUM_PROFILE_SHAPE DetectProfileShape(const SPVPResult &result) {
   if(!result.isValid || NumBins < 3) return SHAPE_UNDEFINED;
   
   // Divide into thirds
   int thirdSize = NumBins / 3;
   double lowerVol = 0, middleVol = 0, upperVol = 0;
   
   for(int i = 0; i < thirdSize; i++) lowerVol += g_volumeHistogram[i];
   for(int i = thirdSize; i < thirdSize*2; i++) middleVol += g_volumeHistogram[i];
   for(int i = thirdSize*2; i < NumBins; i++) upperVol += g_volumeHistogram[i];
   
   double totalVol = lowerVol + middleVol + upperVol;
   if(totalVol == 0) return SHAPE_UNDEFINED;
   
   double lowerPct = lowerVol / totalVol;
   double middlePct = middleVol / totalVol;
   double upperPct = upperVol / totalVol;
   
   // P-Shape: concentration in upper third (bullish)
   if(upperPct > 0.45 && lowerPct < 0.25) return SHAPE_P;
   
   // D-Shape: concentration in lower third (bearish)
   if(lowerPct > 0.45 && upperPct < 0.25) return SHAPE_D;
   
   // B-Shape: balanced/bell curve
   if(middlePct > 0.35) return SHAPE_B;
   
   return SHAPE_UNDEFINED;
}

//+------------------------------------------------------------------+
//| CLASSIFY MARKET STATE                                            |
//+------------------------------------------------------------------+
void ClassifyMarketState(double currentPrice) {
   if(!g_context.pvpHTF.isValid) {
      g_context.state = STATE_ACCUMULATION;
      g_context.score = 50.0;
      return;
   }
   
   double poc = g_context.pvpHTF.poc;
   double vah = g_context.pvpHTF.vah;
   double val = g_context.pvpHTF.val;
   double vaRange = vah - val;
   
   if(vaRange <= 0) vaRange = g_tickSize * 10;
   
   // Feature 1: Position relative to VA (-1 to 1)
   double positionScore = 0;
   if(currentPrice > vah) positionScore = MathMin(1.0, (currentPrice - vah) / vaRange);
   else if(currentPrice < val) positionScore = MathMax(-1.0, (currentPrice - val) / vaRange);
   else positionScore = (currentPrice - poc) / (vaRange / 2);
   
   // Feature 2: POC displacement (normalized)
   double deltaPOCNorm = 0;
   if(MathAbs(g_context.deltaPOC) > 0) {
      deltaPOCNorm = MathMin(1.0, MathAbs(g_context.deltaPOC) / 50.0);
   }
   
   // Feature 3: Profile shape score
   double shapeScore = 0;
   switch(g_context.shape) {
      case SHAPE_P: shapeScore = 0.8; break;
      case SHAPE_D: shapeScore = 0.8; break;
      case SHAPE_B: shapeScore = 0.3; break;
      default: shapeScore = 0.5;
   }
   
   // Feature 4: Volume relative (simplified - using bin concentration)
   double maxBinVol = g_volumeHistogram[g_context.pvpHTF.pocBin];
   double avgVol = g_context.pvpHTF.totalVolume / NumBins;
   double volumeScore = (avgVol > 0) ? MathMin(1.0, maxBinVol / (avgVol * 3)) : 0.5;
   
   // Calculate composite score
   double rawScore = ScoreWeight_Volume * volumeScore + 
                     ScoreWeight_Accept * (1.0 - MathAbs(positionScore)) +
                     ScoreWeight_POC * deltaPOCNorm +
                     ScoreWeight_Shape * shapeScore;
   
   // Apply EMA smoothing
   double emaPeriod = 5.0;
   double alpha = 2.0 / (emaPeriod + 1.0);
   g_scoreEMA = alpha * rawScore * 100 + (1 - alpha) * g_scoreEMA;
   g_context.score = g_scoreEMA;
   
   // State classification with hysteresis
   bool insideVA = (currentPrice >= val && currentPrice <= vah);
   bool breakingUp = (currentPrice > vah + vaRange * 0.1);
   bool breakingDown = (currentPrice < val - vaRange * 0.1);
   bool balancedProfile = (g_context.shape == SHAPE_B);
   bool directionalProfile = (g_context.shape == SHAPE_P || g_context.shape == SHAPE_D);
   
   ENUM_MARKET_STATE prevState = g_context.state;
   
   // State transition logic
   if(insideVA && balancedProfile && deltaPOCNorm < 0.2) {
      if(prevState == STATE_EXPANSION) {
         g_context.state = STATE_DISTRIBUTION;
      } else {
         g_context.state = STATE_ACCUMULATION;
      }
   }
   else if((breakingUp || breakingDown) && directionalProfile && deltaPOCNorm > 0.3) {
      g_context.state = STATE_EXPANSION;
   }
   else if(insideVA && directionalProfile && prevState == STATE_EXPANSION) {
      g_context.state = STATE_DISTRIBUTION;
   }
   else if((breakingUp || breakingDown) && !directionalProfile && deltaPOCNorm < 0.2) {
      // Quick break without follow-through = manipulation
      g_context.state = STATE_MANIPULATION;
   }
   
   // Hysteresis: require stronger signal to change state
   if(g_context.state != prevState) {
      if(g_context.score < 40 || g_context.score > 60) {
         // Keep new state
      } else {
         // Revert to previous state (not enough confidence)
         g_context.state = prevState;
      }
   }
}

//+------------------------------------------------------------------+
//| DRAW VISUALS                                                     |
//+------------------------------------------------------------------+
void DrawVisuals() {
   // Delete old objects
   ObjectsDeleteAll(0, g_objPrefix);
   
   if(!g_context.pvpHTF.isValid) return;
   
   datetime timeStart = iTime(_Symbol, HTF_Timeframe, LookbackBars);
   datetime timeEnd = TimeCurrent() + PeriodSeconds(HTF_Timeframe);
   
   // Draw POC line
   if(DrawPOC) {
      string pocName = g_objPrefix + "POC";
      ObjectCreate(0, pocName, OBJ_TREND, 0, timeStart, g_context.pvpHTF.poc, timeEnd, g_context.pvpHTF.poc);
      ObjectSetInteger(0, pocName, OBJPROP_COLOR, POC_Color);
      ObjectSetInteger(0, pocName, OBJPROP_WIDTH, 2);
      ObjectSetInteger(0, pocName, OBJPROP_STYLE, STYLE_SOLID);
      ObjectSetInteger(0, pocName, OBJPROP_RAY_RIGHT, false);
      ObjectSetString(0, pocName, OBJPROP_TOOLTIP, "POC: " + DoubleToString(g_context.pvpHTF.poc, g_digits));
   }
   
   // Draw Value Area
   if(DrawVA) {
      string vaName = g_objPrefix + "VA";
      ObjectCreate(0, vaName, OBJ_RECTANGLE, 0, timeStart, g_context.pvpHTF.vah, timeEnd, g_context.pvpHTF.val);
      ObjectSetInteger(0, vaName, OBJPROP_COLOR, VA_Color);
      ObjectSetInteger(0, vaName, OBJPROP_FILL, true);
      ObjectSetInteger(0, vaName, OBJPROP_BACK, true);
      ObjectSetString(0, vaName, OBJPROP_TOOLTIP, StringFormat("VA: %."+IntegerToString(g_digits)+"f - %."+IntegerToString(g_digits)+"f", g_context.pvpHTF.val, g_context.pvpHTF.vah));
   }
   
   // Draw HVN/LVN
   if(DrawHVN_LVN) {
      int lineCount = 0;
      for(int i = 0; i < ArraySize(g_context.pvpHTF.levels) && lineCount < 20; i++) {
         if(g_context.pvpHTF.levels[i].isHVN) {
            string hvnName = g_objPrefix + "HVN_" + IntegerToString(i);
            double price = g_context.pvpHTF.levels[i].price;
            ObjectCreate(0, hvnName, OBJ_TREND, 0, timeStart, price, timeEnd, price);
            ObjectSetInteger(0, hvnName, OBJPROP_COLOR, HVN_Color);
            ObjectSetInteger(0, hvnName, OBJPROP_WIDTH, 1);
            ObjectSetInteger(0, hvnName, OBJPROP_STYLE, STYLE_DOT);
            lineCount++;
         }
         if(g_context.pvpHTF.levels[i].isLVN) {
            string lvnName = g_objPrefix + "LVN_" + IntegerToString(i);
            double price = g_context.pvpHTF.levels[i].price;
            ObjectCreate(0, lvnName, OBJ_TREND, 0, timeStart, price, timeEnd, price);
            ObjectSetInteger(0, lvnName, OBJPROP_COLOR, LVN_Color);
            ObjectSetInteger(0, lvnName, OBJPROP_WIDTH, 1);
            ObjectSetInteger(0, lvnName, OBJPROP_STYLE, STYLE_DASHDOT);
            lineCount++;
         }
      }
   }
   
   // Draw Info Panel
   if(DrawPanel) {
      DrawInfoPanel();
   }
   
   ChartRedraw(0);
}

//+------------------------------------------------------------------+
//| DRAW INFO PANEL                                                  |
//+------------------------------------------------------------------+
void DrawInfoPanel() {
   int x = 10;
   int y = 30;
   int lineHeight = 18;
   
   string stateStr = "";
   color stateColor = clrWhite;
   switch(g_context.state) {
      case STATE_ACCUMULATION:
         stateStr = "ACCUMULATION";
         stateColor = clrDodgerBlue;
         break;
      case STATE_EXPANSION:
         stateStr = "EXPANSION";
         stateColor = clrLime;
         break;
      case STATE_DISTRIBUTION:
         stateStr = "DISTRIBUTION";
         stateColor = clrOrange;
         break;
      case STATE_MANIPULATION:
         stateStr = "MANIPULATION";
         stateColor = clrRed;
         break;
   }
   
   string shapeStr = "";
   switch(g_context.shape) {
      case SHAPE_P: shapeStr = "P-Shape"; break;
      case SHAPE_B: shapeStr = "B-Shape"; break;
      case SHAPE_D: shapeStr = "D-Shape"; break;
      default: shapeStr = "---";
   }
   
   // Background
   string bgName = g_objPrefix + "PanelBG";
   ObjectCreate(0, bgName, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, bgName, OBJPROP_XDISTANCE, x - 5);
   ObjectSetInteger(0, bgName, OBJPROP_YDISTANCE, y - 5);
   ObjectSetInteger(0, bgName, OBJPROP_XSIZE, 200);
   ObjectSetInteger(0, bgName, OBJPROP_YSIZE, lineHeight * 8 + 10);
   ObjectSetInteger(0, bgName, OBJPROP_BGCOLOR, clrBlack);
   ObjectSetInteger(0, bgName, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, bgName, OBJPROP_COLOR, clrGray);
   ObjectSetInteger(0, bgName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, bgName, OBJPROP_BACK, false);
   
   // Title
   CreateLabel(g_objPrefix + "L_Title", x, y, "=== INSTITUTIONAL PVP ===", clrGold, 9);
   y += lineHeight;
   
   // State
   CreateLabel(g_objPrefix + "L_State", x, y, "State: " + stateStr, stateColor, 9);
   y += lineHeight;
   
   // Score
   CreateLabel(g_objPrefix + "L_Score", x, y, "Score: " + DoubleToString(g_context.score, 1) + "%", clrWhite, 9);
   y += lineHeight;
   
   // Shape
   CreateLabel(g_objPrefix + "L_Shape", x, y, "Profile: " + shapeStr, clrSilver, 9);
   y += lineHeight;
   
   // POC
   CreateLabel(g_objPrefix + "L_POC", x, y, "POC: " + DoubleToString(g_context.pvpHTF.poc, g_digits), POC_Color, 9);
   y += lineHeight;
   
   // VA
   CreateLabel(g_objPrefix + "L_VA", x, y, StringFormat("VA: %."+IntegerToString(g_digits)+"f-%."+IntegerToString(g_digits)+"f", g_context.pvpHTF.val, g_context.pvpHTF.vah), clrSilver, 9);
   y += lineHeight;
   
   // Session/Killzone
   string cycleStr = "Cycle: " + (g_context.activeSession != "" ? g_context.activeSession : "Off");
   if(g_context.activeKillzone != "") cycleStr += " [" + g_context.activeKillzone + "]";
   CreateLabel(g_objPrefix + "L_Cycle", x, y, cycleStr, clrAqua, 9);
   y += lineHeight;
   
   // Delta POC
   CreateLabel(g_objPrefix + "L_Delta", x, y, "ΔPOC: " + DoubleToString(g_context.deltaPOC, 1) + " ticks", clrSilver, 9);
}

//+------------------------------------------------------------------+
//| CREATE LABEL HELPER                                              |
//+------------------------------------------------------------------+
void CreateLabel(string name, int x, int y, string text, color clr, int fontSize) {
   ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fontSize);
   ObjectSetString(0, name, OBJPROP_FONT, "Consolas");
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
}
//+------------------------------------------------------------------+
