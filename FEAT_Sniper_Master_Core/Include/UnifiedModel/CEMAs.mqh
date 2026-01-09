//+------------------------------------------------------------------+
//|                                                      CEMAs.mqh |
//|                    Multifractal EMA Engine - 31 Layers          |
//|         Micro (Red), Operational (Green), Macro (Blue), Bias (Grey) |
//|         Updated for User Physics: Wind(8), River(21), Wall(50), Magnet(200) |
//+------------------------------------------------------------------+
#ifndef CEMAS_MQH
#define CEMAS_MQH

//+------------------------------------------------------------------+
//| EMA GROUP DEFINITIONS                                            |
//+------------------------------------------------------------------+
#define EMA_COUNT 31
#define EMA_MICRO_COUNT 10
#define EMA_OPERATIONAL_COUNT 10
#define EMA_MACRO_COUNT 10
#define EMA_BIAS_COUNT 1

// Indices of Key Physics Objects
#define EMA_IDX_WIND 5    // Index of EMA 8
#define EMA_IDX_RIVER 11  // Index of EMA 21
#define EMA_IDX_WALL 13   // Index of EMA 50
#define EMA_IDX_MAGNET 19 // Index of EMA 200 (Moved to end of Operational or start of Macro)

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
   
   // New Physics Getters
   double            GetWind() const { return m_ptrEmas[EMA_IDX_WIND].value; }   // EMA 8
   double            GetRiver() const { return m_ptrEmas[EMA_IDX_RIVER].value; } // EMA 21
   double            GetWall() const { return m_ptrEmas[EMA_IDX_WALL].value; }   // EMA 50
   double            GetMagnet() const { return m_ptrEmas[EMA_IDX_MAGNET].value; } // EMA 200
   
   double            GetWindSlope() const { return m_ptrEmas[EMA_IDX_WIND].slope; }
   double            GetRiverSlope() const { return m_ptrEmas[EMA_IDX_RIVER].slope; }
   double            GetWallSlope() const { return m_ptrEmas[EMA_IDX_WALL].slope; }
   
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
   double            GetBiasSlope() const { return m_ptrEmas[30].slope; }
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEMAs::CEMAs() {
   m_initialized = false;
   m_atr = 0;
   m_atrHandle = INVALID_HANDLE;
   
   // MICRO Periods (Red) - 10 Layers
   m_emaPeriods[0] = 1; m_emaPeriods[1] = 2; m_emaPeriods[2] = 3; m_emaPeriods[3] = 4;
   m_emaPeriods[4] = 6; 
   m_emaPeriods[5] = 8; // KEY: WIND (Index 5)
   m_emaPeriods[6] = 10; m_emaPeriods[7] = 12;
   m_emaPeriods[8] = 13; m_emaPeriods[9] = 14;
   
   // OPERATIONAL Periods (Green) - 10 Layers (+ Magnet at end)
   m_emaPeriods[10] = 16; 
   m_emaPeriods[11] = 21; // KEY: RIVER (Index 11)
   m_emaPeriods[12] = 32; 
   m_emaPeriods[13] = 50; // KEY: WALL (Index 13)
   m_emaPeriods[14] = 64; m_emaPeriods[15] = 96; m_emaPeriods[16] = 100; m_emaPeriods[17] = 128;
   m_emaPeriods[18] = 150; 
   m_emaPeriods[19] = 200; // KEY: MAGNET (Index 19)
   
   // MACRO Periods (Blue) - 10 Layers (Deep Institutional)
   m_emaPeriods[20] = 256; m_emaPeriods[21] = 300; m_emaPeriods[22] = 365; m_emaPeriods[23] = 400;
   m_emaPeriods[24] = 500; m_emaPeriods[25] = 600; m_emaPeriods[26] = 800; m_emaPeriods[27] = 900;
   m_emaPeriods[28] = 1000; m_emaPeriods[29] = 1200;
   
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

#endif // CEMAS_MQH
