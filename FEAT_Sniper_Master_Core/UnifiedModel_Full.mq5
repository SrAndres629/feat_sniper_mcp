//+------------------------------------------------------------------+
//|                                                      CEMAs.mqh |
//|                    Multifractal EMA Engine - 31 Layers          |
//|         Micro (Red), Operational (Green), Macro (Blue), Bias (Grey) |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| EMA GROUP DEFINITIONS                                            |
//+------------------------------------------------------------------+

enum ENUM_EMA_GROUP {
   EMA_GROUP_MICRO,       // 1-14 periods (Red)
   EMA_GROUP_OPERATIONAL, // 16-224 periods (Green)
   EMA_GROUP_MACRO,       // 256-1280 periods (Blue)
   EMA_GROUP_BIAS         // 2048 period (Grey)
};

//+------------------------------------------------------------------+
//| EMA DATA STRUCTURE                                               |
//+------------------------------------------------------------------+
struct SEMAData {
   int      period;
   double   value;
   double   prevValue;
   double   slope;         // Normalized slope
   double   curvature;     // Second derivative
   ENUM_EMA_GROUP group;
};

//+------------------------------------------------------------------+
//| EMA GROUP METRICS                                                |
//+------------------------------------------------------------------+
struct SEMAGroupMetrics {
   double   avgValue;
   double   avgSlope;
   double   spread;        // Max - Min in group
   double   compression;   // Normalized compression (0-1)
   bool     aligned;       // All slopes same direction
};

//+------------------------------------------------------------------+
//| FAN METRICS (Global EMA State)                                   |
//+------------------------------------------------------------------+
struct SFanMetrics {
   double   totalSpread;      // Distance between fastest and slowest
   double   compression;      // Global compression (0=max spread, 1=compressed)
   double   openingSpeed;     // Rate of spread change
   bool     bullishOrder;     // Micro > Operational > Macro
   bool     bearishOrder;     // Micro < Operational < Macro
   bool     isConverging;     // Layers coming together
   bool     isDiverging;      // Layers spreading apart
};

//+------------------------------------------------------------------+
//| CEMAs CLASS                                                      |
//+------------------------------------------------------------------+
class CEMAs {
private:
   SEMAData          m_ptrEmas[EMA_COUNT];
   SEMAGroupMetrics  m_microMetrics;
   SEMAGroupMetrics  m_operationalMetrics;
   SEMAGroupMetrics  m_macroMetrics;
   SEMAGroupMetrics  m_biasMetrics;
   SFanMetrics       m_fanMetrics;
   
   int               m_emaPeriods[EMA_COUNT];
   int               m_emaHandles[EMA_COUNT];
   double            m_emaBuffers[EMA_COUNT];
   double            m_emaPrevBuffers[EMA_COUNT];
   
   double            m_atr;
   int               m_atrHandle;
   int               m_atrPeriod;
   
   string            m_symbol;
   ENUM_TIMEFRAMES   m_timeframe;
   bool              m_initialized;
   
   // Private methods
   void              AssignGroups();
   double            NormalizeSlope(double slope);
   void              CalculateGroupMetrics(ENUM_EMA_GROUP group, SEMAGroupMetrics &metrics);
   void              CalculateFanMetrics();
   
public:
   CEMAs();
   ~CEMAs();
   
   // Initialization
   bool              Init(string symbol, ENUM_TIMEFRAMES tf, int atrPeriod = 14);
   void              Deinit();
   
   // Calculation
   bool              Calculate(int shift = 0);
   bool              IsReady() const { return m_initialized; }
   
   // Getters - Individual EMAs
   double            GetEMA(int index) const;
   double            GetEMASlope(int index) const;
   double            GetEMACurvature(int index) const;
   int               GetEMAPeriod(int index) const;
   ENUM_EMA_GROUP    GetEMAGroup(int index) const;
   int               GetHandle(int index) const;
   
   // Getters - Group Metrics
   SEMAGroupMetrics  GetMicroMetrics() const { return m_microMetrics; }
   SEMAGroupMetrics  GetOperationalMetrics() const { return m_operationalMetrics; }
   SEMAGroupMetrics  GetMacroMetrics() const { return m_macroMetrics; }
   SEMAGroupMetrics  GetBiasMetrics() const { return m_biasMetrics; }
   
   // Getters - Fan Metrics
   SFanMetrics       GetFanMetrics() const { return m_fanMetrics; }
   
   // Getters - Price Position
   double            GetPricePosition(double price) const;  // -1 to 1 relative to EMA cloud
   bool              IsPriceAboveCloud(double price) const;
   bool              IsPriceBelowCloud(double price) const;
   bool              IsPriceInCloud(double price) const;
   
   // Getters - ATR
   double            GetATR() const { return m_atr; }
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEMAs::CEMAs() {
   m_initialized = false;
   m_atr = 0;
   m_atrHandle = INVALID_HANDLE;
   
   // MICRO Periods (Red)
   m_emaPeriods[0] = 1; m_emaPeriods[1] = 2; m_emaPeriods[2] = 3; m_emaPeriods[3] = 6;
   m_emaPeriods[4] = 7; m_emaPeriods[5] = 8; m_emaPeriods[6] = 9; m_emaPeriods[7] = 12;
   m_emaPeriods[8] = 13; m_emaPeriods[9] = 14;
   
   // OPERATIONAL Periods (Green)
   m_emaPeriods[10] = 16; m_emaPeriods[11] = 24; m_emaPeriods[12] = 32; m_emaPeriods[13] = 48;
   m_emaPeriods[14] = 64; m_emaPeriods[15] = 96; m_emaPeriods[16] = 128; m_emaPeriods[17] = 160;
   m_emaPeriods[18] = 192; m_emaPeriods[19] = 224;
   
   // MACRO Periods (Blue)
   m_emaPeriods[20] = 256; m_emaPeriods[21] = 320; m_emaPeriods[22] = 384; m_emaPeriods[23] = 448;
   m_emaPeriods[24] = 512; m_emaPeriods[25] = 640; m_emaPeriods[26] = 768; m_emaPeriods[27] = 896;
   m_emaPeriods[28] = 1024; m_emaPeriods[29] = 1280;
   
   // BIAS Period (Grey)
   m_emaPeriods[30] = 2048;
   
   for(int i = 0; i < EMA_COUNT; i++) {
      m_emaHandles[i] = INVALID_HANDLE;
      m_emaBuffers[i] = 0;
      m_emaPrevBuffers[i] = 0;
   }
   
   AssignGroups();
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CEMAs::~CEMAs() {
   Deinit();
}

//+------------------------------------------------------------------+
//| Assign EMA Groups                                                |
//+------------------------------------------------------------------+
void CEMAs::AssignGroups() {
   for(int i = 0; i < EMA_COUNT; i++) {
      m_ptrEmas[i].period = m_emaPeriods[i];
      if(i < EMA_MICRO_COUNT) m_ptrEmas[i].group = EMA_GROUP_MICRO;
      else if(i < EMA_MICRO_COUNT + EMA_OPERATIONAL_COUNT) m_ptrEmas[i].group = EMA_GROUP_OPERATIONAL;
      else if(i < EMA_MICRO_COUNT + EMA_OPERATIONAL_COUNT + EMA_MACRO_COUNT) m_ptrEmas[i].group = EMA_GROUP_MACRO;
      else m_ptrEmas[i].group = EMA_GROUP_BIAS;
   }
}

//+------------------------------------------------------------------+
//| Initialize                                                       |
//+------------------------------------------------------------------+
bool CEMAs::Init(string symbol, ENUM_TIMEFRAMES tf, int atrPeriod = 14) {
   m_symbol = symbol;
   m_timeframe = tf;
   m_atrPeriod = atrPeriod;
   
   // Create EMA handles (30 EMAs + 1 SMMA)
   for(int i = 0; i < EMA_COUNT; i++) {
      ENUM_MA_METHOD method = (i == 30) ? MODE_SMMA : MODE_EMA;
      m_emaHandles[i] = iMA(m_symbol, m_timeframe, m_emaPeriods[i], 0, method, PRICE_CLOSE);
      if(m_emaHandles[i] == INVALID_HANDLE) {
         Print("[CEMAs] Failed to create handle for period ", m_emaPeriods[i]);
         Deinit();
         return false;
      }
   }
   
   // Create ATR handle
   m_atrHandle = iATR(m_symbol, m_timeframe, m_atrPeriod);
   if(m_atrHandle == INVALID_HANDLE) {
      Print("[CEMAs] Failed to create ATR handle");
      Deinit();
      return false;
   }
   
   m_initialized = true;
   return true;
}

//+------------------------------------------------------------------+
//| Deinitialize                                                     |
//+------------------------------------------------------------------+
void CEMAs::Deinit() {
   for(int i = 0; i < EMA_COUNT; i++) {
      if(m_emaHandles[i] != INVALID_HANDLE) {
         IndicatorRelease(m_emaHandles[i]);
         m_emaHandles[i] = INVALID_HANDLE;
      }
   }
   if(m_atrHandle != INVALID_HANDLE) {
      IndicatorRelease(m_atrHandle);
      m_atrHandle = INVALID_HANDLE;
   }
   m_initialized = false;
}

//+------------------------------------------------------------------+
//| Calculate EMAs and Metrics                                       |
//+------------------------------------------------------------------+
bool CEMAs::Calculate(int shift = 0) {
   if(!m_initialized) return false;
   
   double buffer[2];
   
   // Get ATR
   if(CopyBuffer(m_atrHandle, 0, shift, 1, buffer) <= 0) return false;
   m_atr = buffer[0];
   if(m_atr <= 0) m_atr = 0.0001;
   
   // Get EMA values
   for(int i = 0; i < EMA_COUNT; i++) {
      m_emaPrevBuffers[i] = m_emaBuffers[i];
      
      if(CopyBuffer(m_emaHandles[i], 0, shift, 2, buffer) < 2) return false;
      
      m_emaBuffers[i] = buffer[1];  // Current
      double prevVal = buffer[0];   // Previous
      
      m_ptrEmas[i].prevValue = m_ptrEmas[i].value;
      m_ptrEmas[i].value = m_emaBuffers[i];
      
      // Calculate slope
      double rawSlope = m_emaBuffers[i] - prevVal;
      m_ptrEmas[i].slope = NormalizeSlope(rawSlope);
      
      // Curvature
      if(m_emaPrevBuffers[i] > 0) {
         double prevSlope = (m_emaPrevBuffers[i] - prevVal);
         m_ptrEmas[i].curvature = (rawSlope - prevSlope) / m_atr;
      } else m_ptrEmas[i].curvature = 0;
   }
   
   // Calculate group metrics
   CalculateGroupMetrics(EMA_GROUP_MICRO, m_microMetrics);
   CalculateGroupMetrics(EMA_GROUP_OPERATIONAL, m_operationalMetrics);
   CalculateGroupMetrics(EMA_GROUP_MACRO, m_macroMetrics);
   CalculateGroupMetrics(EMA_GROUP_BIAS, m_biasMetrics);
   
   // Calculate fan metrics
   CalculateFanMetrics();
   
   return true;
}

//+------------------------------------------------------------------+
//| Normalize Slope by ATR                                           |
//+------------------------------------------------------------------+
double CEMAs::NormalizeSlope(double slope) {
   if(m_atr <= 0) return 0;
   return slope / m_atr;
}

//+------------------------------------------------------------------+
//| Calculate Group Metrics                                          |
//+------------------------------------------------------------------+
void CEMAs::CalculateGroupMetrics(ENUM_EMA_GROUP group, SEMAGroupMetrics &metrics) {
   int startIdx = 0, count = 0;
   
   switch(group) {
      case EMA_GROUP_MICRO:       startIdx = 0; count = EMA_MICRO_COUNT; break;
      case EMA_GROUP_OPERATIONAL: startIdx = 10; count = EMA_OPERATIONAL_COUNT; break;
      case EMA_GROUP_MACRO:       startIdx = 20; count = EMA_MACRO_COUNT; break;
      case EMA_GROUP_BIAS:        startIdx = 30; count = EMA_BIAS_COUNT; break;
   }
   
   double sumValue = 0, sumSlope = 0;
   double minValue = DBL_MAX, maxValue = -DBL_MAX;
   int slopeSign = 0;
   bool allSameSign = true;
   
   for(int i = startIdx; i < startIdx + count; i++) {
      sumValue += m_ptrEmas[i].value;
      sumSlope += m_ptrEmas[i].slope;
      
      if(m_ptrEmas[i].value < minValue) minValue = m_ptrEmas[i].value;
      if(m_ptrEmas[i].value > maxValue) maxValue = m_ptrEmas[i].value;
      
      int currentSign = (m_ptrEmas[i].slope > 0) ? 1 : ((m_ptrEmas[i].slope < 0) ? -1 : 0);
      if(i == startIdx) slopeSign = currentSign;
      else if(currentSign != slopeSign && currentSign != 0) allSameSign = false;
   }
   
   metrics.avgValue = sumValue / count;
   metrics.avgSlope = sumSlope / count;
   metrics.spread = maxValue - minValue;
   metrics.compression = (m_atr > 0) ? MathMin(1.0, metrics.spread / (m_atr * 3)) : 0;
   metrics.aligned = allSameSign;
}

//+------------------------------------------------------------------+
//| Calculate Fan Metrics                                            |
//+------------------------------------------------------------------+
void CEMAs::CalculateFanMetrics() {
   double fastestEMA = m_ptrEmas[0].value;
   double slowestEMA = m_ptrEmas[30].value;
   m_fanMetrics.totalSpread = MathAbs(fastestEMA - slowestEMA);
   
   double refSpread = m_atr * 20;
   m_fanMetrics.compression = 1.0 - MathMin(1.0, m_fanMetrics.totalSpread / refSpread);
   
   static double prevSpread = 0;
   m_fanMetrics.openingSpeed = (m_fanMetrics.totalSpread - prevSpread) / m_atr;
   prevSpread = m_fanMetrics.totalSpread;
   
   bool bullish = (m_ptrEmas[0].value > m_ptrEmas[10].value && m_ptrEmas[10].value > m_ptrEmas[20].value);
   bool bearish = (m_ptrEmas[0].value < m_ptrEmas[10].value && m_ptrEmas[10].value < m_ptrEmas[20].value);
   
   m_fanMetrics.bullishOrder = bullish;
   m_fanMetrics.bearishOrder = bearish;
   m_fanMetrics.isConverging = (m_fanMetrics.openingSpeed < -0.1);
   m_fanMetrics.isDiverging = (m_fanMetrics.openingSpeed > 0.1);
}

//+------------------------------------------------------------------+
//| Getters                                                          |
//+------------------------------------------------------------------+
double CEMAs::GetEMA(int index) const {
   if(index < 0 || index >= EMA_COUNT) return 0;
   return m_ptrEmas[index].value;
}
double CEMAs::GetEMASlope(int index) const {
   if(index < 0 || index >= EMA_COUNT) return 0;
   return m_ptrEmas[index].slope;
}
double CEMAs::GetEMACurvature(int index) const {
   if(index < 0 || index >= EMA_COUNT) return 0;
   return m_ptrEmas[index].curvature;
}
int CEMAs::GetEMAPeriod(int index) const {
   if(index < 0 || index >= EMA_COUNT) return 0;
   return m_ptrEmas[index].period;
}
ENUM_EMA_GROUP CEMAs::GetEMAGroup(int index) const {
   if(index < 0 || index >= EMA_COUNT) return EMA_GROUP_MICRO;
   return m_ptrEmas[index].group;
}
int CEMAs::GetHandle(int index) const {
   if(index < 0 || index >= EMA_COUNT) return INVALID_HANDLE;
   return m_emaHandles[index];
}

double CEMAs::GetPricePosition(double price) const {
   double cloudHigh = -DBL_MAX, cloudLow = DBL_MAX;
   for(int i = 0; i < EMA_COUNT; i++) {
      if(m_ptrEmas[i].value > cloudHigh) cloudHigh = m_ptrEmas[i].value;
      if(m_ptrEmas[i].value < cloudLow) cloudLow = m_ptrEmas[i].value;
   }
   double range = cloudHigh - cloudLow;
   if(range <= 0) return 0;
   return (price - ((cloudHigh + cloudLow) / 2.0)) / (range / 2.0);
}

bool CEMAs::IsPriceAboveCloud(double price) const {
   for(int i = 0; i < EMA_COUNT; i++) if(price <= m_ptrEmas[i].value) return false;
   return true;
}
bool CEMAs::IsPriceBelowCloud(double price) const {
   for(int i = 0; i < EMA_COUNT; i++) if(price >= m_ptrEmas[i].value) return false;
   return true;
}
bool CEMAs::IsPriceInCloud(double price) const {
   return !IsPriceAboveCloud(price) && !IsPriceBelowCloud(price);
}

//+------------------------------------------------------------------+
//|                                                 CLiquidity.mqh |
//|                    Liquidity Layer - Causal Reading             |
//|          External, Internal, Imbalance Detection                |
//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
//| LIQUIDITY TYPES                                                  |
//+------------------------------------------------------------------+
enum ENUM_LIQUIDITY_TYPE {
   LIQ_EXTERNAL,         
   LIQ_INTERNAL,         
   LIQ_IMBALANCE         
};

enum ENUM_LIQUIDITY_SIDE {
   LIQ_ABOVE,            
   LIQ_BELOW             
};

//+------------------------------------------------------------------+
//| LEVEL STRUCTURES                                                 |
//+------------------------------------------------------------------+
struct SLiquidityLevel {
   double               price;
   ENUM_LIQUIDITY_TYPE  type;
   ENUM_LIQUIDITY_SIDE  side;
   datetime             createdTime;
   bool                 mitigated;
   datetime             mitigatedTime;
   double               strength;        
   string               label;           
};

struct SImbalance {
   double               high;
   double               low;
   double               midpoint;
   datetime             time;
   bool                 mitigated;
   double               fillPercent;     
   bool                 isBullish;       
};

struct SLiquidityContext {
   SLiquidityLevel      nearestAbove;
   SLiquidityLevel      nearestBelow;
   int                  totalAbove;
   int                  totalBelow;
   double               imbalanceAbove;  
   double               imbalanceBelow;  
   bool                 isValid;
};

//+------------------------------------------------------------------+
//| SWING & PATTERN TYPES                                            |
//+------------------------------------------------------------------+
enum ENUM_SWING_TYPE { SWING_HH, SWING_HL, SWING_LH, SWING_LL, SWING_NONE };

struct SSwingPoint {
   ENUM_SWING_TYPE type;
   double          price;
   datetime        time;
   int             barIndex;
};

enum ENUM_PATTERN_TYPE { PAT_M, PAT_W, PAT_HCH, PAT_GC, PAT_TRIANGLE, PAT_FLAG, PAT_NONE };

struct SPatternEvent {
   ENUM_PATTERN_TYPE type;
   datetime          time;
   double            price;
   bool              isBullish;
   string            description;
};

enum ENUM_STRUCT_TYPE { STRUCT_BOS, STRUCT_CHOCH };

struct SStructureEvent {
   ENUM_STRUCT_TYPE  type;
   bool              isBullish;  
   double            price;      
   datetime          time;       
   int               barIndex;   
   bool              active;     
};

//+------------------------------------------------------------------+
//| CLIQUIDITY CLASS                                                 |
//+------------------------------------------------------------------+
class CLiquidity {
private:
   SLiquidityLevel      m_levels[];
   SImbalance           m_imbalances[];
   SLiquidityContext    m_context;
   
   string               m_symbol;
   ENUM_TIMEFRAMES      m_timeframe;
   
   int                  m_maxLevels;
   int                  m_lookbackBars;
   double               m_equalThreshold;  
   double               m_fvgMinSize;      
   int                  m_imbalanceCount;
   
   double               m_atr;
   double               m_tickSize;
   
   SStructureEvent      m_structures[];
   int                  m_structCount;
   SSwingPoint          m_swings[];
   int                  m_swingCount;
   SPatternEvent        m_patterns[];
   int                  m_patternCount;

   void                 DetectStructure(const double &high[], const double &low[], const double &close[], const datetime &time[], int count);
   void                 DetectHighsLows(const double &high[], const double &low[], const datetime &time[], int count);
   void                 DetectPatterns(const double &close[], const datetime &time[], int count);
   void                 DetectExternalLiquidity(const double &high[], const double &low[], const datetime &time[], int count);
   void                 DetectInternalLiquidity(const double &high[], const double &low[], const double &close[], const datetime &time[], int count);
   void                 DetectImbalances(const double &high[], const double &low[], const double &open[], const double &close[], const datetime &time[], int count);
   void                 UpdateMitigation(double currentHigh, double currentLow);
   void                 BuildContext(double currentPrice);
   bool                 IsEqualLevel(double price1, double price2);
   void                 AddLevel(double price, ENUM_LIQUIDITY_TYPE type, ENUM_LIQUIDITY_SIDE side, datetime time, double strength, string label);
   
public:
                        CLiquidity();
                       ~CLiquidity();
   
   bool                 Init(string symbol, ENUM_TIMEFRAMES tf, int maxLevels = 50, int lookback = 100, double equalPips = 5.0, double fvgMinATR = 0.5);
   void                 SetATR(double atr) { m_atr = atr; }
   bool                 Calculate(const double &high[], const double &low[], const double &open[], const double &close[], const datetime &time[], int count, double currentPrice);
   
   SLiquidityContext    GetContext() const { return m_context; }
   int                  GetLevelCount() const { return ArraySize(m_levels); }
   SLiquidityLevel      GetLevel(int index) const { if(index>=0 && index<ArraySize(m_levels)) return m_levels[index]; SLiquidityLevel e; ZeroMemory(e); return e; }
   int                  GetImbalanceCount() const { return m_imbalanceCount; }
   SImbalance           GetImbalance(int index) const { if(index>=0 && index<m_imbalanceCount) return m_imbalances[index]; SImbalance e; ZeroMemory(e); return e; }
   
   bool                 GetNearestLevel(double price, ENUM_LIQUIDITY_SIDE side, SLiquidityLevel &level);
   bool                 GetNearestImbalance(double price, ENUM_LIQUIDITY_SIDE side, SImbalance &imb);
   
   int                  GetStructureCount() const { return m_structCount; }
   SStructureEvent      GetStructureEvent(int index) const { if(index>=0 && index<m_structCount) return m_structures[index]; SStructureEvent e; ZeroMemory(e); return e; }
   int                  GetSwingCount() const { return m_swingCount; }
   SSwingPoint          GetSwingPoint(int index) const { if(index>=0 && index<m_swingCount) return m_swings[index]; SSwingPoint e; ZeroMemory(e); return e; }
   int                  GetPatternCount() const { return m_patternCount; }
   SPatternEvent        GetPatternEvent(int index) const { if(index>=0 && index<m_patternCount) return m_patterns[index]; SPatternEvent e; ZeroMemory(e); return e; }
};

CLiquidity::CLiquidity() {
   m_maxLevels = 50; m_lookbackBars = 100; m_equalThreshold = 0.0005; m_fvgMinSize = 0.5; m_atr = 0.001; m_tickSize = 0.00001;
   ZeroMemory(m_context);
}

CLiquidity::~CLiquidity() { ArrayFree(m_levels); ArrayFree(m_imbalances); }

bool CLiquidity::Init(string symbol, ENUM_TIMEFRAMES tf, int maxLevels, int lookback, double equalPips, double fvgMinATR) {
   m_symbol = symbol; m_timeframe = tf; m_maxLevels = maxLevels; m_lookbackBars = lookback; m_fvgMinSize = fvgMinATR;
   m_tickSize = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_SIZE);
   m_equalThreshold = equalPips * m_tickSize * 10;
   ArrayResize(m_levels, 0); ArrayResize(m_imbalances, m_maxLevels); m_imbalanceCount = 0;
   ArrayResize(m_structures, m_maxLevels); m_structCount = 0;
   ArrayResize(m_swings, m_maxLevels); m_swingCount = 0;
   ArrayResize(m_patterns, m_maxLevels); m_patternCount = 0;
   return true;
}

bool CLiquidity::Calculate(const double &high[], const double &low[], const double &open[], const double &close[], const datetime &time[], int count, double currentPrice) {
   if(count < 10) return false;
   int barsToProcess = MathMin(count, m_lookbackBars);
   ArrayResize(m_levels, 0); m_imbalanceCount = 0; m_structCount = 0; m_swingCount = 0; m_patternCount = 0;
   DetectHighsLows(high, low, time, barsToProcess);
   DetectStructure(high, low, close, time, barsToProcess);
   DetectPatterns(close, time, barsToProcess);
   DetectExternalLiquidity(high, low, time, barsToProcess);
   DetectInternalLiquidity(high, low, close, time, barsToProcess);
   DetectImbalances(high, low, open, close, time, barsToProcess);
   UpdateMitigation(high[0], low[0]);
   BuildContext(currentPrice);
   return true;
}

void CLiquidity::DetectHighsLows(const double &high[], const double &low[], const datetime &time[], int count) {
   double lastSH = 0, lastSL = 0, prevSH = 0, prevSL = 0;
   for(int i = count - 5; i >= 0; i--) {
      bool isSH = (high[i] > high[i+1] && high[i] > high[i+2] && high[i] > high[i-1] && high[i] > high[i-2]);
      bool isSL = (low[i] < low[i+1] && low[i] < low[i+2] && low[i] < low[i-1] && low[i] < low[i-2]);
      if(isSH) {
         prevSH = lastSH; lastSH = high[i]; SSwingPoint sp; sp.price = lastSH; sp.time = time[i]; sp.barIndex = i;
         if(prevSH == 0) sp.type = SWING_NONE; else if(lastSH > prevSH) sp.type = SWING_HH; else sp.type = SWING_LH;
         if(m_swingCount < m_maxLevels) m_swings[m_swingCount++] = sp;
      }
      if(isSL) {
         prevSL = lastSL; lastSL = low[i]; SSwingPoint sp; sp.price = lastSL; sp.time = time[i]; sp.barIndex = i;
         if(prevSL == 0) sp.type = SWING_NONE; else if(lastSL < prevSL) sp.type = SWING_LL; else sp.type = SWING_HL;
         if(m_swingCount < m_maxLevels) m_swings[m_swingCount++] = sp;
      }
   }
}

void CLiquidity::DetectStructure(const double &high[], const double &low[], const double &close[], const datetime &time[], int count) {
   int lastHighIndex = -1, lastLowIndex = -1; double lastHighPrice = 0, lastLowPrice = DBL_MAX; bool trendBullish = false;
   for(int i = count - 5; i >= 0; i--) {
      bool isSH = (high[i] > high[i+1] && high[i] > high[i+2] && high[i] > high[i-1] && high[i] > high[i-2]);
      bool isSL = (low[i] < low[i+1] && low[i] < low[i+2] && low[i] < low[i-1] && low[i] < low[i-2]);
      if(isSH) { lastHighIndex = i; lastHighPrice = high[i]; }
      if(isSL) { lastLowIndex = i; lastLowPrice = low[i]; }
      if(lastHighIndex != -1 && i < lastHighIndex) { 
         if(close[i] > lastHighPrice) {
            bool duplicate = (m_structCount > 0 && m_structures[m_structCount-1].price == lastHighPrice && m_structures[m_structCount-1].isBullish);
            if(!duplicate && m_structCount < m_maxLevels) {
               SStructureEvent e; e.time = time[i]; e.price = lastHighPrice; e.barIndex = i; e.active = true; e.isBullish = true;
               if(trendBullish) e.type = STRUCT_BOS; else { e.type = STRUCT_CHOCH; trendBullish = true; }
               m_structures[m_structCount++] = e;
            }
            lastHighIndex = -1;
         }
      }
      if(lastLowIndex != -1 && i < lastLowIndex) {
         if(close[i] < lastLowPrice) {
            bool duplicate = (m_structCount > 0 && m_structures[m_structCount-1].price == lastLowPrice && !m_structures[m_structCount-1].isBullish);
            if(!duplicate && m_structCount < m_maxLevels) {
               SStructureEvent e; e.time = time[i]; e.price = lastLowPrice; e.barIndex = i; e.active = true; e.isBullish = false;
               if(!trendBullish) e.type = STRUCT_BOS; else { e.type = STRUCT_CHOCH; trendBullish = false; }
               m_structures[m_structCount++] = e;
            }
            lastLowIndex = -1;
         }
      }
   }
}

void CLiquidity::DetectPatterns(const double &close[], const datetime &time[], int count) {
   for(int i = 0; i < m_structCount; i++) {
      if(m_structures[i].type == STRUCT_CHOCH) {
         SPatternEvent p; p.type = PAT_GC; p.time = m_structures[i].time; p.price = m_structures[i].price; p.isBullish = m_structures[i].isBullish; p.description = "GC";
         if(m_patternCount < m_maxLevels) m_patterns[m_patternCount++] = p;
      }
   }
}

void CLiquidity::DetectExternalLiquidity(const double &high[], const double &low[], const datetime &time[], int count) {
   for(int i = 2; i < count - 2; i++) {
      if(high[i] > high[i-1] && high[i] > high[i-2] && high[i] > high[i+1] && high[i] > high[i+2]) AddLevel(high[i], LIQ_EXTERNAL, LIQ_ABOVE, time[i], 0.8, "SwingHigh");
      if(low[i] < low[i-1] && low[i] < low[i-2] && low[i] < low[i+1] && low[i] < low[i+2]) AddLevel(low[i], LIQ_EXTERNAL, LIQ_BELOW, time[i], 0.8, "SwingLow");
   }
}

void CLiquidity::DetectInternalLiquidity(const double &high[], const double &low[], const double &close[], const datetime &time[], int count) {
   for(int i = 5; i < count - 5; i++) {
      if(high[i] >= high[i-1] && high[i] >= high[i+1] && high[i] < high[i-3] && high[i] < high[i+3]) AddLevel(high[i], LIQ_INTERNAL, LIQ_ABOVE, time[i], 0.4, "MicroHigh");
      if(low[i] <= low[i-1] && low[i] <= low[i+1] && low[i] > low[i-3] && low[i] > low[i+3]) AddLevel(low[i], LIQ_INTERNAL, LIQ_BELOW, time[i], 0.4, "MicroLow");
   }
}

void CLiquidity::DetectImbalances(const double &high[], const double &low[], const double &open[], const double &close[], const datetime &time[], int count) {
   for(int i = 1; i < count - 1; i++) {
      if(low[i-1] > high[i+1] && (low[i-1]-high[i+1]) > m_atr*m_fvgMinSize) {
         SImbalance imb; imb.high = low[i-1]; imb.low = high[i+1]; imb.midpoint = (imb.high+imb.low)/2; imb.time = time[i]; imb.mitigated = false; imb.fillPercent = 0; imb.isBullish = true;
         if(m_imbalanceCount < m_maxLevels) m_imbalances[m_imbalanceCount++] = imb;
      }
      if(high[i-1] < low[i+1] && (low[i+1]-high[i-1]) > m_atr*m_fvgMinSize) {
         SImbalance imb; imb.high = low[i+1]; imb.low = high[i-1]; imb.midpoint = (imb.high+imb.low)/2; imb.time = time[i]; imb.mitigated = false; imb.fillPercent = 0; imb.isBullish = false;
         if(m_imbalanceCount < m_maxLevels) m_imbalances[m_imbalanceCount++] = imb;
      }
   }
}

void CLiquidity::BuildContext(double currentPrice) {
   m_context.totalAbove = 0; m_context.totalBelow = 0; m_context.nearestAbove.price = 0; m_context.nearestBelow.price = 0; m_context.imbalanceAbove = 0; m_context.imbalanceBelow = 0;
   double nAD = DBL_MAX, nBD = DBL_MAX;
   for(int i = 0; i < ArraySize(m_levels); i++) {
      if(m_levels[i].mitigated) continue;
      if(m_levels[i].side == LIQ_ABOVE && m_levels[i].price > currentPrice) {
         m_context.totalAbove++; double d = m_levels[i].price - currentPrice; if(d < nAD) { nAD = d; m_context.nearestAbove = m_levels[i]; }
      }
      if(m_levels[i].side == LIQ_BELOW && m_levels[i].price < currentPrice) {
         m_context.totalBelow++; double d = currentPrice - m_levels[i].price; if(d < nBD) { nBD = d; m_context.nearestBelow = m_levels[i]; }
      }
   }
   m_context.isValid = true;
}

void CLiquidity::AddLevel(double price, ENUM_LIQUIDITY_TYPE type, ENUM_LIQUIDITY_SIDE side, datetime time, double strength, string label) {
   for(int i = 0; i < ArraySize(m_levels); i++) if(MathAbs(m_levels[i].price - price) < m_equalThreshold && m_levels[i].side == side) return;
   if(ArraySize(m_levels) >= m_maxLevels) ArrayRemove(m_levels, 0, 1);
   SLiquidityLevel lvl; lvl.price = price; lvl.type = type; lvl.side = side; lvl.createdTime = time; lvl.mitigated = false; lvl.strength = strength; lvl.label = label;
   int sz = ArraySize(m_levels); ArrayResize(m_levels, sz + 1); m_levels[sz] = lvl;
}

bool CLiquidity::GetNearestLevel(double price, ENUM_LIQUIDITY_SIDE side, SLiquidityLevel &level) {
   double nD = DBL_MAX; bool f = false;
   for(int i = 0; i < ArraySize(m_levels); i++) {
      if(m_levels[i].mitigated || m_levels[i].side != side) continue;
      double d = MathAbs(m_levels[i].price - price); if(d < nD) { nD = d; level = m_levels[i]; f = true; }
   }
   return f;
}

bool CLiquidity::GetNearestImbalance(double price, ENUM_LIQUIDITY_SIDE side, SImbalance &imb) {
   double nD = DBL_MAX; bool f = false;
   for(int i = 0; i < m_imbalanceCount; i++) {
      if(m_imbalances[i].mitigated) continue;
      bool isAbove = (m_imbalances[i].midpoint > price); if((side==LIQ_ABOVE && !isAbove) || (side==LIQ_BELOW && isAbove)) continue;
      double d = MathAbs(m_imbalances[i].midpoint - price); if(d < nD) { nD = d; imb = m_imbalances[i]; f = true; }
   }
   return f;
}

void CLiquidity::UpdateMitigation(double cH, double cL) {
   for(int i=0; i<ArraySize(m_levels); i++) {
      if(!m_levels[i].mitigated) {
         if(m_levels[i].side==LIQ_ABOVE && cH >= m_levels[i].price) m_levels[i].mitigated=true;
         if(m_levels[i].side==LIQ_BELOW && cL <= m_levels[i].price) m_levels[i].mitigated=true;
      }
   }
}

//+------------------------------------------------------------------+
//|                                                      CFEAT.mqh |
//|                    FEAT Layer - Cognitive Reading               |
//+------------------------------------------------------------------+


enum ENUM_FORM_TYPE { FORM_CONSTRUCTION, FORM_IMPULSE, FORM_EXHAUSTION, FORM_UNDEFINED };
enum ENUM_SPACE_TYPE { SPACE_EXPANDED, SPACE_COMPRESSED, SPACE_VOID, SPACE_NORMAL, SPACE_AT_ZONE };
enum ENUM_ACCEL_TYPE { ACCEL_VALID, ACCEL_FAKE, ACCEL_NONE };
enum ENUM_TIME_WEIGHT { TIME_NOISE, TIME_RELEVANT, TIME_STRUCTURAL };
enum ENUM_STRUCT_MODEL { MODEL_HARMONIC, MODEL_MAE, MODEL_DISPLACEMENT, MODEL_NONE };

struct SFormMetrics {
   ENUM_FORM_TYPE  type;
   ENUM_STRUCT_MODEL model;        
   double          curvatureScore;
   double          fanAngle;
   double          compressionRatio;
   double          harmonicScore;  
   bool            isFlattening;
   bool            isCurving;
   bool            hasBOS;         
   bool            hasCHoCH;       
   bool            isIntentCandle; 
};

struct SSpaceMetrics {
   ENUM_SPACE_TYPE type;
   double          fastMediumGap;
   double          mediumSlowGap;
   double          density;
   double          energy;
   int             voidZones;
   double          proximityScore; 
   bool            atZone;         
};

struct SAccelMetrics {
   ENUM_ACCEL_TYPE type;
   double          fanOpeningSpeed;
   double          fastSeparation;
   double          reactionSpeed;
   bool            hasStructure;
};

struct STimeMetrics {
   ENUM_TIME_WEIGHT weight;
   ENUM_TIMEFRAMES  currentTF;
   string           activeSession;
   double           tfMultiplier;
   double           sessionMultiplier;
   bool             isKillzone;
};

struct SFEATResult {
   SFormMetrics    form;
   SSpaceMetrics   space;
   SAccelMetrics   accel;
   STimeMetrics    time;
   double          compositeScore;
   bool            isValid;
};

class CFEAT {
private:
   CEMAs*           m_ptrEmas;
   CLiquidity*      m_ptrLiq;
   SFEATResult      m_result;
   double           m_curveTh, m_compTh, m_accelTh, m_gapTh;
   int              m_aS, m_aE, m_lS, m_lE, m_nS, m_nE;
   int              m_kzLS, m_kzLE, m_kzNS, m_kzNE;

   void             CalcForm(double open, double high, double low, double close, double volume);
   void             CalcSpace(double price);
   void             CalcAccel();
   void             CalcTime(ENUM_TIMEFRAMES tf);
   void             CalcComp();

public:
   CFEAT();
   ~CFEAT() {}

   void SetEMAs(CEMAs* e) { m_ptrEmas = e; }
   void SetLiquidity(CLiquidity* l) { m_ptrLiq = l; }
   void SetThresholds(double c, double p, double a, double g) { m_curveTh=c; m_compTh=p; m_accelTh=a; m_gapTh=g; }
   void SetSessionTimes(int as, int ae, int ls, int le, int ns, int ne) { m_aS=as; m_aE=ae; m_lS=ls; m_lE=le; m_nS=ns; m_nE=ne; }
   void SetKillzones(int kls, int kle, int kns, int kne) { m_kzLS=kls; m_kzLE=kle; m_kzNS=kns; m_kzNE=kne; }

   bool Calculate(ENUM_TIMEFRAMES tf, datetime t, double open, double high, double low, double close, double volume);
   
   SFEATResult   GetResult() const { return m_result; }
   SFormMetrics  GetForm() const { return m_result.form; }
   SSpaceMetrics GetSpace() const { return m_result.space; }
   SAccelMetrics GetAccel() const { return m_result.accel; }
   STimeMetrics  GetTime() const { return m_result.time; }
   double        GetCompositeScore() const { return m_result.compositeScore; }
};

CFEAT::CFEAT() : m_ptrEmas(NULL), m_ptrLiq(NULL), m_curveTh(0.3), m_compTh(0.7), m_accelTh(0.5), m_gapTh(1.5),
                  m_aS(0), m_aE(540), m_lS(480), m_lE(1020), m_nS(780), m_nE(1320),
                  m_kzLS(480), m_kzLE(600), m_kzNS(780), m_kzNE(900) {
   ZeroMemory(m_result);
}

bool CFEAT::Calculate(ENUM_TIMEFRAMES tf, datetime t, double open, double high, double low, double close, double volume) {
   if(m_ptrEmas == NULL) return false;
   if(!m_ptrEmas->IsReady()) return false;
   CalcForm(open, high, low, close, volume); CalcSpace(close); CalcAccel(); CalcTime(tf); CalcComp();
   m_result.isValid = true;
   return true;
}

void CFEAT::CalcForm(double open, double high, double low, double close, double volume) {
   if(m_ptrEmas == NULL) return;
   SFanMetrics fan; fan = m_ptrEmas->GetFanMetrics();
   double avgC = 0; for(int i=0; i<30; i++) avgC += m_ptrEmas->GetEMACurvature(i); avgC /= 30.0;
   m_result.form.curvatureScore = MathMax(-1.0, MathMin(1.0, avgC * 10));
   m_result.form.compressionRatio = fan.compression;
   if(m_ptrLiq != NULL) {
      int sCount = m_ptrLiq->GetStructureCount();
      if(sCount > 0) {
         SStructureEvent last; last = m_ptrLiq->GetStructureEvent(sCount-1);
         m_result.form.hasBOS = (last.type == STRUCT_BOS);
         m_result.form.hasCHoCH = (last.type == STRUCT_CHOCH);
      }
   }
   double body = MathAbs(close - open); double rC = high - low; double atr = m_ptrEmas->GetATR();
   m_result.form.isIntentCandle = (body > atr * 0.8) && (rC > 0 && body / rC > 0.7); 
   m_result.form.type = (MathAbs(m_result.form.curvatureScore)>0.5 ? FORM_IMPULSE : FORM_CONSTRUCTION);
}

void CFEAT::CalcSpace(double price) {
   if(m_ptrEmas == NULL) return;
   SEMAGroupMetrics mic; mic = m_ptrEmas->GetMicroMetrics(); 
   SEMAGroupMetrics opr; opr = m_ptrEmas->GetOperationalMetrics(); 
   double atr = m_ptrEmas->GetATR();
   m_result.space.fastMediumGap = MathAbs(mic.avgValue - opr.avgValue)/atr;
   m_result.space.density = m_ptrEmas->GetFanMetrics().compression;
}

void CFEAT::CalcAccel() {
   if(m_ptrEmas == NULL) return;
   SFanMetrics fan; fan = m_ptrEmas->GetFanMetrics();
   m_result.accel.fanOpeningSpeed = fan.openingSpeed;
}

void CFEAT::CalcTime(ENUM_TIMEFRAMES tf) {
   m_result.time.currentTF = tf;
}

void CFEAT::CalcComp() {
   m_result.compositeScore = 70.0;
}

//+------------------------------------------------------------------+
//|                                                       CFSM.mqh |
//|                    FSM Layer - State Classification             |
//|        Multifractal Layer Logic: Micro, Oper, Macro, Bias       |
//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
//| MARKET STATE ENUMERATION                                         |
//+------------------------------------------------------------------+
enum ENUM_MARKET_STATE {
   STATE_CALIBRATING,    // Filling buffers
   STATE_ACCUMULATION,   // Micro compressed, Oper flat
   STATE_EXPANSION,      // Layers separating, Micro leading
   STATE_DISTRIBUTION,   // Macro sloping, Micro chaotic
   STATE_MANIPULATION,   // Violent Micro crossing against Oper
   STATE_ABSORPTION      // High effort, low result in Micro layer
};

struct SStateThresholds {
   double   effortP80, effortP20;
   double   resultP80, resultP20;
   double   layerSeparation;  // Normalized dist between Oper and Macro
   double   biasSlope;        // Absolute Regime filter
};

struct SFSMMetrics {
   double effort;
   double result;
   double compression;
   double slope;
   double speed;
};

class CFSM {
private:
   CEMAs*            m_ptrEmas;
   CFEAT*            m_ptrFeat;
   CLiquidity*       m_liquidity;
   
   ENUM_MARKET_STATE m_state;
   double            m_confidence;
   int               m_barsInState;
   
   double            m_effortBuffer[];
   double            m_resultBuffer[];
   int               m_bufferSize;
   int               m_index;
   bool              m_full;

   // Layer Questions
   double            GetLayerMetric(ENUM_EMA_GROUP group, string type); // type: "compression", "slope"
   bool              AreLayersExpanding();
   bool              AreLayersConverging();

public:
   CFSM();
   ~CFSM();
   
   void SetComponents(CEMAs* emas, CFEAT* feat, CLiquidity* liq) { m_ptrEmas = emas; m_ptrFeat = feat; m_liquidity = liq; }
   void SetBufferSize(int size) { m_bufferSize = size; ArrayResize(m_effortBuffer, size); ArrayResize(m_resultBuffer, size); }
   
   bool Calculate(double close, double prevClose, double volume);
   
   ENUM_MARKET_STATE GetState() const { return m_state; }
   string GetStateString() const;
   double GetConfidence() const { return m_confidence; }
   SFSMMetrics GetMetrics();
};

CFSM::CFSM() : m_state(STATE_CALIBRATING), m_confidence(0), m_barsInState(0), m_bufferSize(100), m_index(0), m_full(false) {
   ArrayResize(m_effortBuffer, m_bufferSize);
   ArrayResize(m_resultBuffer, m_bufferSize);
}

CFSM::~CFSM() {}

bool CFSM::Calculate(double close, double prevClose, double volume) {
   if(m_ptrEmas == NULL) return false;
   
   double atr = m_ptrEmas->GetATR();
   double effort = volume / (atr * 100000); // Normalized effort
   double result = MathAbs(close - prevClose) / atr;
   
   m_effortBuffer[m_index] = effort;
   m_resultBuffer[m_index] = result;
   m_index = (m_index + 1) % m_bufferSize;
   if(m_index == 0) m_full = true;
   
   if(!m_full) { m_state = STATE_CALIBRATING; return true; }
   
   // --- Professional Layer Question Logic ---
   SEMAGroupMetrics micro = m_ptrEmas->GetMicroMetrics();
   SEMAGroupMetrics oper  = m_ptrEmas->GetOperationalMetrics();
   SEMAGroupMetrics macro = m_ptrEmas->GetMacroMetrics();
   SEMAGroupMetrics bias  = m_ptrEmas->GetBiasMetrics();
   
   // 1. Is Micro compressed?
   bool microCompressed = micro.compression > 0.7;
   
   // 2. Is Oper rejecting or absorbing?
   bool priceInOper = m_ptrEmas->GetPricePosition(close) < 0.5 && m_ptrEmas->GetPricePosition(close) > -0.5;
   
   // 3. Is Macro sloping?
   bool macroSloping = MathAbs(macro.avgSlope) > 0.2;
   
   // 4. Layers separating?
   double sep = MathAbs(micro.avgValue - oper.avgValue) / atr;
   bool separating = sep > 2.0;
   
   // --- State Classification ---
   ENUM_MARKET_STATE newState = m_state;
   
   // ABSORPTION: High Effort but Micro Compressed (Wyckoff)
   if(effort > 1.5 && microCompressed) newState = STATE_ABSORPTION;
   
   // MANIPULATION: Micro violent crosses against Bias
   else if(MathAbs(micro.avgSlope) > 1.0 && (micro.avgSlope * bias.avgSlope < 0)) newState = STATE_MANIPULATION;
   
   // EXPANSION: Layers separating + Macro sloping
   else if(separating && macroSloping) newState = STATE_EXPANSION;
   
   // ACCUMULATION: Micro compressed + Price inside Operational Cloud
   else if(microCompressed && priceInOper) newState = STATE_ACCUMULATION;
   
   // DISTRIBUTION: Macro flat + Micro chaotic (high spread)
   else if(MathAbs(macro.avgSlope) < 0.1 && micro.spread > atr * 5) newState = STATE_DISTRIBUTION;
   
   if(newState != m_state) {
      m_state = newState;
      m_barsInState = 0;
   } else m_barsInState++;
   
   m_confidence = 70.0; // Placeholder for ML confidence integration
   return true;
}

string CFSM::GetStateString() const {
   switch(m_state) {
      case STATE_CALIBRATING:  return "CALIBRATING";
      case STATE_ACCUMULATION: return "ACCUMULATION";
      case STATE_EXPANSION:    return "EXPANSION";
      case STATE_DISTRIBUTION: return "DISTRIBUTION";
      case STATE_MANIPULATION: return "MANIPULATION";
      case STATE_ABSORPTION:   return "ABSORPTION";
   }
   return "UNKNOWN";
}

SFSMMetrics CFSM::GetMetrics() {
   SFSMMetrics m;
   ZeroMemory(m);
   if(m_ptrEmas != NULL) {
      SEMAGroupMetrics micro = m_ptrEmas->GetMicroMetrics();
      SEMAGroupMetrics oper  = m_ptrEmas->GetOperationalMetrics();
      m.effort = m_effortBuffer[(m_index + m_bufferSize - 1) % m_bufferSize];
      m.result = m_resultBuffer[(m_index + m_bufferSize - 1) % m_bufferSize];
      m.compression = micro.compression;
      m.slope = oper.avgSlope;
      m.speed = (micro.avgValue - oper.avgValue) / m_ptrEmas->GetATR();
   }
   return m;
}

//+------------------------------------------------------------------+
//|                                                   CVisuals.mqh |
//|                    Visualization Engine - Dashboard + Levels      |
//+------------------------------------------------------------------+


class CVisuals {
private:
   string         m_prefix;
   long           m_chart;
   CEMAs*         m_ptrEmas;
   CFEAT*         m_ptrFeat;
   CLiquidity*    m_ptrLiq;
   CFSM*          m_ptrFsm;
   
   bool           m_showEMAs;
   bool           m_showDashboard;
   bool           m_showLiquidity;
   bool           m_showStructure;

   void           CreateLabel(string name, int x, int y, string text, color clr, int size);
   void           CreateHLine(string name, double price, color clr, ENUM_LINE_STYLE style, int width);

public:
   CVisuals();
   ~CVisuals() { Clear(); }

   void Init(string prefix, long chartID) { m_prefix = prefix; m_chart = chartID; }
   void SetComponents(CEMAs* e, CFEAT* f, CLiquidity* l, CFSM* sm) { m_ptrEmas = e; m_ptrFeat = f; m_ptrLiq = l; m_ptrFsm = sm; }
   void SetDrawOptions(bool emas, bool dash, bool liq, bool structr) { m_showEMAs = emas; m_showDashboard = dash; m_showLiquidity = liq; m_showStructure = structr; }

   void Draw(datetime t, double close);
   void Clear();
};

CVisuals::CVisuals() : m_ptrEmas(NULL), m_ptrFeat(NULL), m_ptrLiq(NULL), m_ptrFsm(NULL), m_showEMAs(true), m_showDashboard(true), m_showLiquidity(true), m_showStructure(true) {}

void CVisuals::Clear() { ObjectsDeleteAll(m_chart, m_prefix); }

void CVisuals::Draw(datetime t, double close) {
   ObjectsDeleteAll(m_chart, m_prefix);
   int x = 20, y = 30, h = 18;

   if(m_showDashboard && m_ptrFsm != NULL && m_ptrEmas != NULL && m_ptrFeat != NULL) {
      CreateLabel(m_prefix+"Title", x, y, "FEAT SNIPER - MASTER CORE", clrGold, 10); y += h + 5;
      ENUM_MARKET_STATE state = m_ptrFsm->GetState();
      color stateClr = (state == STATE_EXPANSION) ? clrLime : (state == STATE_MANIPULATION ? clrRed : clrDodgerBlue);
      CreateLabel(m_prefix+"State", x, y, "STATE: " + m_ptrFsm->GetStateString(), stateClr, 9); y += h;
      
      SEMAGroupMetrics micro = m_ptrEmas->GetMicroMetrics();
      SEMAGroupMetrics oper  = m_ptrEmas->GetOperationalMetrics();
      SEMAGroupMetrics macro = m_ptrEmas->GetMacroMetrics();
      
      CreateLabel(m_prefix+"L1", x, y, StringFormat("Layer 1 (Micro): %s | %.1f%% Comp", (micro.avgSlope > 0 ? "Up" : "Down"), micro.compression*100), clrOrangeRed, 8); y += h;
      CreateLabel(m_prefix+"L2", x, y, StringFormat("Layer 2 (Oper): %s | Path Stability", (oper.avgSlope > 0 ? "Bullish" : "Bearish")), clrLime, 8); y += h;
      CreateLabel(m_prefix+"L3", x, y, StringFormat("Layer 3 (Macro): %s | Memory", (macro.avgSlope > 0 ? "Sloping" : "Flat")), clrDeepSkyBlue, 8); y += h;
      
      y += 5;
      CreateLabel(m_prefix+"Q1", x, y, "Question: Are Layers Expanding? " + (m_ptrEmas->GetFanMetrics().isDiverging ? "YES" : "No"), clrWhite, 8); y += h;
      CreateLabel(m_prefix+"Q2", x, y, "Question: FEAT Model: " + (m_ptrFeat->GetForm().model == MODEL_MAE ? "MAE (Momentum)" : (m_ptrFeat->GetForm().model == MODEL_HARMONIC ? "HARMONIC (FIB 50%)" : "NONE")), clrGold, 8); y += h;
      CreateLabel(m_prefix+"Q3", x, y, "Question: Is Bias Holding? " + (m_ptrEmas->GetBiasMetrics().avgSlope > 0 ? "BUY SIDE" : "SELL SIDE"), clrSilver, 8); y += h;
   }

   if(m_ptrLiq != NULL) {
      SLiquidityContext lctx = m_ptrLiq->GetContext();
      if(m_showDashboard) {
         CreateLabel(m_prefix+"LiqA", x, y, "Nearest Buy Liquidity: " + DoubleToString(lctx.nearestAbove.price, _Digits), clrWhite, 8); y += h;
         CreateLabel(m_prefix+"LiqB", x, y, "Nearest Sell Liquidity: " + DoubleToString(lctx.nearestBelow.price, _Digits), clrWhite, 8); y += h;
      }
      
      if(m_showStructure) {
         int sCount = m_ptrLiq->GetStructureCount();
         for(int i = 0; i < sCount; i++) {
            SStructureEvent e = m_ptrLiq->GetStructureEvent(i);
            if(e.active) {
               string name = m_prefix + "STRUCT_" + IntegerToString(i);
               color c = e.isBullish ? clrLime : clrRed; ENUM_LINE_STYLE s = (e.type == STRUCT_BOS) ? STYLE_SOLID : STYLE_DASH;
               ObjectCreate(m_chart, name, OBJ_TREND, 0, e.time, e.price, TimeCurrent(), e.price);
               ObjectSetInteger(m_chart, name, OBJPROP_COLOR, c); ObjectSetInteger(m_chart, name, OBJPROP_STYLE, s);
               ObjectSetInteger(m_chart, name, OBJPROP_RAY_RIGHT, true);
            }
         }
         int swingCount = m_ptrLiq->GetSwingCount();
         for(int i=0; i<swingCount; i++) {
            SSwingPoint sp = m_ptrLiq->GetSwingPoint(i); string name = m_prefix + "SWING_" + IntegerToString(i);
            string txt = (sp.type==SWING_HH?"HH":(sp.type==SWING_HL?"HL":(sp.type==SWING_LH?"LH":(sp.type==SWING_LL?"LL":""))));
            if(txt != "") {
               ObjectCreate(m_chart, name, OBJ_TEXT, 0, sp.time, sp.price + (sp.type==SWING_HH||sp.type==SWING_LH ? _Point*50 : -_Point*150));
               ObjectSetString(m_chart, name, OBJPROP_TEXT, txt); ObjectSetInteger(m_chart, name, OBJPROP_COLOR, clrWhite);
            }
         }
      }
      
      if(m_showLiquidity) {
         for(int i=0; i<m_ptrLiq->GetLevelCount(); i++) {
            SLiquidityLevel lvl = m_ptrLiq->GetLevel(i); if(lvl.mitigated) continue;
            string name = m_prefix + "LVL_" + IntegerToString(i);
            CreateHLine(name, lvl.price, (lvl.side==LIQ_ABOVE ? clrOrangeRed : clrMediumSeaGreen), STYLE_DOT, 1);
         }
      }
   }
   
   if(m_ptrFeat != NULL && (m_ptrFeat->GetForm()).isIntentCandle) {
      CreateLabel(m_prefix+"Intent", x, y, "RULES OF CONTINUITY: INTENT DETECTED", clrCyan, 8); y += h;
   }
   
   ChartRedraw(m_chart);
}

void CVisuals::CreateLabel(string name, int x, int y, string text, color clr, int size) {
   ObjectCreate(m_chart, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(m_chart, name, OBJPROP_XDISTANCE, x); ObjectSetInteger(m_chart, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(m_chart, name, OBJPROP_TEXT, text); ObjectSetInteger(m_chart, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(m_chart, name, OBJPROP_FONTSIZE, size); ObjectSetString(m_chart, name, OBJPROP_FONT, "Segoe UI Semibold");
}

void CVisuals::CreateHLine(string name, double price, color clr, ENUM_LINE_STYLE style, int width) {
   ObjectCreate(m_chart, name, OBJ_HLINE, 0, 0, price);
   ObjectSetInteger(m_chart, name, OBJPROP_COLOR, clr); ObjectSetInteger(m_chart, name, OBJPROP_STYLE, style);
   ObjectSetInteger(m_chart, name, OBJPROP_WIDTH, width);
}

//+------------------------------------------------------------------+
//|                                                   CInterop.mqh |
//|                    MT5 <-> Python Interoperability              |
//|            Upgraded for Multifractal Layer Export               |
//+------------------------------------------------------------------+


struct SBarDataExport {
   datetime time;
   double open, high, low, close;
   double effort, result;
   // Layer Metrics
   double microComp, microSlope;
   double operComp, operSlope;
   double macroComp, macroSlope;
   double biasSlope;
   double layerSep12; // micro - oper
   double layerSep23; // oper - macro
   string state;
};

class CInterop {
private:
   bool m_enabled;
   string m_path;

public:
   CInterop() : m_enabled(true), m_path("UnifiedModel\\") {}
   void SetEnabled(bool e) { m_enabled = e; }
   void SetDataPath(string p) { m_path = p; }

   bool AppendBarData(string filename, datetime time, double open, double high, double low, double close,
                      double effort, double result, double compression, double slope, double speed, 
                      double confidence, string state);
};

bool CInterop::AppendBarData(string filename, datetime time, double o, double h, double l, double c,
                            double eff, double res, double comp, double slp, double spd, 
                            double conf, string state) {
   if(!m_enabled) return false;
   int handle = FileOpen(filename, FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON);
   if(handle == INVALID_HANDLE) {
      handle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON);
      if(handle != INVALID_HANDLE)
         FileWrite(handle, "time,open,high,low,close,effort,result,microComp,operSlope,layerSep,state");
   }
   if(handle == INVALID_HANDLE) return false;
   FileSeek(handle, 0, SEEK_END);
   FileWrite(handle, TimeToString(time) + "," + DoubleToString(o, 5) + "," + DoubleToString(h, 5) + "," +
                     DoubleToString(l, 5) + "," + DoubleToString(c, 5) + "," + DoubleToString(eff, 5) + "," +
                     DoubleToString(res, 5) + "," + DoubleToString(comp, 5) + "," + DoubleToString(slp, 5) + "," +
                     DoubleToString(spd, 5) + "," + state);
   FileClose(handle);
   return true;
}

//+------------------------------------------------------------------+
//|                                          UnifiedModel_Main.mq5 |
//|                    Unified Institutional Model                  |
//|              31 Multifractal Layers + FEAT + Liquidity          |
//|                                                                  |
//| NOTE: This indicator focuses on EMA Layers, FEAT, and Liquidity.|
//|       Volume Profile (PVP) is handled by 'InstitutionalPVP.mq5' |
//|       to maintain performance and visual clarity.               |
//+------------------------------------------------------------------+
#property copyright "Institutional Trading Systems"
#property link      "https://github.com/SrAndres629/feat_sniper_mcp"
#property version   "2.01"
#property indicator_chart_window
#property indicator_buffers 31
#property indicator_plots   31

// --- EMA VISUAL HIERARCHY ---
// Micro: Thin, Reactive (Red/Orange)
#property indicator_label1 "M1"
#property indicator_type1 DRAW_LINE
#property indicator_color1 C'80,0,0' // Dark Red
#property indicator_width1 1
#property indicator_label2 "M2"
#property indicator_type2 DRAW_LINE
#property indicator_color2 C'100,0,0'
#property indicator_width2 1
#property indicator_label3 "M3"
#property indicator_type3 DRAW_LINE
#property indicator_color3 C'120,0,0'
#property indicator_width3 1
#property indicator_label4 "M4"
#property indicator_type4 DRAW_LINE
#property indicator_color4 C'140,0,0'
#property indicator_width4 1
#property indicator_label5 "M5 (Core)"
#property indicator_type5 DRAW_LINE
#property indicator_color5 clrRed   // Bright Red (Core)
#property indicator_width5 2    // Slightly thicker
#property indicator_label6 "M6"
#property indicator_type6 DRAW_LINE
#property indicator_color6 C'255,69,0' // Orange Red
#property indicator_width6 1
#property indicator_label7 "M7"
#property indicator_type7 DRAW_LINE
#property indicator_color7 C'255,69,0'
#property indicator_width7 1
#property indicator_label8 "M8"
#property indicator_type8 DRAW_LINE
#property indicator_color8 clrOrange
#property indicator_width8 1
#property indicator_label9 "M9"
#property indicator_type9 DRAW_LINE
#property indicator_color9 clrOrange
#property indicator_width9 1
#property indicator_label10 "M10"
#property indicator_type10 DRAW_LINE
#property indicator_color10 clrOrange
#property indicator_width10 1

// Operational: Structure (Green)
#property indicator_label11 "O1"
#property indicator_type11 DRAW_LINE
#property indicator_color11 C'0,100,0' // Dark Green
#property indicator_width11 1
#property indicator_label12 "O2"
#property indicator_type12 DRAW_LINE
#property indicator_color12 C'0,128,0'
#property indicator_width12 1
#property indicator_label13 "O3"
#property indicator_type13 DRAW_LINE
#property indicator_color13 C'34,139,34'
#property indicator_width13 1
#property indicator_label14 "O4"
#property indicator_type14 DRAW_LINE
#property indicator_color14 C'34,139,34'
#property indicator_width14 1
#property indicator_label15 "O5 (Core)"
#property indicator_type15 DRAW_LINE
#property indicator_color15 clrLime // Bright Lime (Core)
#property indicator_width15 2
#property indicator_label16 "O6"
#property indicator_type16 DRAW_LINE
#property indicator_color16 C'50,205,50' // Lime Green
#property indicator_width16 1
#property indicator_label17 "O7"
#property indicator_type17 DRAW_LINE
#property indicator_color17 C'154,205,50'
#property indicator_width17 1
#property indicator_label18 "O8"
#property indicator_type18 DRAW_LINE
#property indicator_color18 C'154,205,50'
#property indicator_width18 1
#property indicator_label19 "O9"
#property indicator_type19 DRAW_LINE
#property indicator_color19 clrYellow
#property indicator_width19 1
#property indicator_label20 "O10"
#property indicator_type20 DRAW_LINE
#property indicator_color20 clrYellow
#property indicator_width20 1

// Macro: Background (Blue)
#property indicator_label21 "Ma1"
#property indicator_type21 DRAW_LINE
#property indicator_color21 C'0,0,139' // Dark Blue
#property indicator_width21 1
#property indicator_label22 "Ma2"
#property indicator_type22 DRAW_LINE
#property indicator_color22 C'0,0,205'
#property indicator_width22 1
#property indicator_label23 "Ma3"
#property indicator_type23 DRAW_LINE
#property indicator_color23 C'0,0,255'
#property indicator_width23 1
#property indicator_label24 "Ma4"
#property indicator_type24 DRAW_LINE
#property indicator_color24 C'65,105,225'
#property indicator_width24 1
#property indicator_label25 "Ma5 (Core)"
#property indicator_type25 DRAW_LINE
#property indicator_color25 clrDodgerBlue
#property indicator_width25 2
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
#property indicator_width31 3 // Prominent thickness
#property indicator_style31 STYLE_SOLID


input group "--- Configuration ---"
input int ATR_Period = 14;
input int Lookback = 100;
input bool ShowDashboard = true; // Toggle Information Panel
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
   g_feat.SetLiquidity(&g_liq); // Connect V2 Logic
   g_liq.Init(_Symbol, _Period, 50, Lookback, 5.0, 0.5);
   g_fsm.SetComponents(&g_emas, &g_feat, &g_liq);
   g_fsm.SetBufferSize(Lookback);
   g_vis.Init("UM_", ChartID());
   g_vis.SetComponents(&g_emas, &g_feat, &g_liq, &g_fsm);
   g_vis.SetDrawOptions(true, ShowDashboard, true, true); // Pass toggle to visuals
   
   return INIT_SUCCEEDED;
}

void OnDeinit(const int r) { g_emas.Deinit(); g_vis.Clear(); }

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
      last = time[total-1];
      g_emas.Calculate(0);
      
      // Calculate FEAT V2 Layers
      g_liq.Calculate(high, low, open, close, time, total, close[total-1]);
      g_feat.Calculate((ENUM_TIMEFRAMES)_Period, time[total-1], open[total-1], high[total-1], low[total-1], close[total-1], (double)tick[total-1]);
      
      g_fsm.Calculate(close[total-1], close[total-2], (double)tick[total-1]);
      if(ShowDashboard) g_vis.Draw(time[total-1], close[total-1]);
      else g_vis.Clear(); // Ensure it's cleared if disabled
   }
   
   return total;
}
