//+------------------------------------------------------------------+
//|                                                      CEMAs.mqh |
//|                         30 EMAs Engine - Memory Map             |
//|                    Micro/Meso/Macro Temporal Scales             |
//+------------------------------------------------------------------+
#ifndef CEMAS_MQH
#define CEMAS_MQH

//+------------------------------------------------------------------+
//| EMA GROUP DEFINITIONS                                            |
//| Fast (Micro): 3,5,8,13 — Algorithms, scalpers, noise            |
//| Medium (Meso): 21,34,55,89 — Smart Money, fair value            |
//| Slow (Macro): 144,233,377,610 — Heavy capital, structure        |
//+------------------------------------------------------------------+
#define EMA_COUNT 12
#define EMA_FAST_COUNT 4
#define EMA_MEDIUM_COUNT 4
#define EMA_SLOW_COUNT 4

enum ENUM_EMA_GROUP {
   EMA_GROUP_FAST,    // Microstructure (3,5,8,13)
   EMA_GROUP_MEDIUM,  // Institutional (21,34,55,89)
   EMA_GROUP_SLOW     // Macro (144,233,377,610)
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
   bool     bullishOrder;     // Fast > Medium > Slow
   bool     bearishOrder;     // Fast < Medium < Slow
   bool     isConverging;     // EMAs coming together
   bool     isDiverging;      // EMAs spreading apart
};

//+------------------------------------------------------------------+
//| CEMAs CLASS                                                      |
//+------------------------------------------------------------------+
class CEMAs {
private:
   SEMAData          m_emas[EMA_COUNT];
   SEMAGroupMetrics  m_fastMetrics;
   SEMAGroupMetrics  m_mediumMetrics;
   SEMAGroupMetrics  m_slowMetrics;
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
   double            GetEMACurvature(int index) const;
   int               GetEMAPeriod(int index) const;
   ENUM_EMA_GROUP    GetEMAGroup(int index) const;
   int               GetHandle(int index) const;
   
   // Getters - Group Metrics
   SEMAGroupMetrics  GetFastMetrics() const { return m_fastMetrics; }
   SEMAGroupMetrics  GetMediumMetrics() const { return m_mediumMetrics; }
   SEMAGroupMetrics  GetSlowMetrics() const { return m_slowMetrics; }
   
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
   
   // Define EMA periods (Fibonacci-based)
   m_emaPeriods[0] = 3;    m_emaPeriods[1] = 5;    m_emaPeriods[2] = 8;    m_emaPeriods[3] = 13;
   m_emaPeriods[4] = 21;   m_emaPeriods[5] = 34;   m_emaPeriods[6] = 55;   m_emaPeriods[7] = 89;
   m_emaPeriods[8] = 144;  m_emaPeriods[9] = 233;  m_emaPeriods[10] = 377; m_emaPeriods[11] = 610;
   
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
      m_emas[i].period = m_emaPeriods[i];
      if(i < EMA_FAST_COUNT) m_emas[i].group = EMA_GROUP_FAST;
      else if(i < EMA_FAST_COUNT + EMA_MEDIUM_COUNT) m_emas[i].group = EMA_GROUP_MEDIUM;
      else m_emas[i].group = EMA_GROUP_SLOW;
   }
}

//+------------------------------------------------------------------+
//| Initialize                                                       |
//+------------------------------------------------------------------+
bool CEMAs::Init(string symbol, ENUM_TIMEFRAMES tf, int atrPeriod = 14) {
   m_symbol = symbol;
   m_timeframe = tf;
   m_atrPeriod = atrPeriod;
   
   // Create EMA handles
   for(int i = 0; i < EMA_COUNT; i++) {
      m_emaHandles[i] = iMA(m_symbol, m_timeframe, m_emaPeriods[i], 0, MODE_EMA, PRICE_CLOSE);
      if(m_emaHandles[i] == INVALID_HANDLE) {
         Print("[CEMAs] Failed to create EMA handle for period ", m_emaPeriods[i]);
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
   if(m_atr <= 0) m_atr = 0.0001;  // Prevent division by zero
   
   // Get EMA values
   for(int i = 0; i < EMA_COUNT; i++) {
      m_emaPrevBuffers[i] = m_emaBuffers[i];
      
      if(CopyBuffer(m_emaHandles[i], 0, shift, 2, buffer) < 2) return false;
      
      m_emaBuffers[i] = buffer[1];  // Current
      double prevEma = buffer[0];   // Previous
      
      m_emas[i].prevValue = m_emas[i].value;
      m_emas[i].value = m_emaBuffers[i];
      
      // Calculate slope (normalized by ATR)
      double rawSlope = m_emaBuffers[i] - prevEma;
      m_emas[i].slope = NormalizeSlope(rawSlope);
      
      // Calculate curvature (change in slope)
      if(m_emaPrevBuffers[i] > 0) {
         double prevSlope = (m_emaPrevBuffers[i] - prevEma);
         m_emas[i].curvature = (rawSlope - prevSlope) / m_atr;
      } else {
         m_emas[i].curvature = 0;
      }
   }
   
   // Calculate group metrics
   CalculateGroupMetrics(EMA_GROUP_FAST, m_fastMetrics);
   CalculateGroupMetrics(EMA_GROUP_MEDIUM, m_mediumMetrics);
   CalculateGroupMetrics(EMA_GROUP_SLOW, m_slowMetrics);
   
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
   int startIdx = 0, endIdx = 0;
   
   switch(group) {
      case EMA_GROUP_FAST:   startIdx = 0; endIdx = EMA_FAST_COUNT; break;
      case EMA_GROUP_MEDIUM: startIdx = EMA_FAST_COUNT; endIdx = startIdx + EMA_MEDIUM_COUNT; break;
      case EMA_GROUP_SLOW:   startIdx = EMA_FAST_COUNT + EMA_MEDIUM_COUNT; endIdx = startIdx + EMA_SLOW_COUNT; break;
   }
   
   double sumValue = 0, sumSlope = 0;
   double minValue = DBL_MAX, maxValue = -DBL_MAX;
   int slopeSign = 0;
   bool allSameSign = true;
   
   for(int i = startIdx; i < endIdx; i++) {
      sumValue += m_emas[i].value;
      sumSlope += m_emas[i].slope;
      
      if(m_emas[i].value < minValue) minValue = m_emas[i].value;
      if(m_emas[i].value > maxValue) maxValue = m_emas[i].value;
      
      int currentSign = (m_emas[i].slope > 0) ? 1 : ((m_emas[i].slope < 0) ? -1 : 0);
      if(i == startIdx) slopeSign = currentSign;
      else if(currentSign != slopeSign && currentSign != 0) allSameSign = false;
   }
   
   int count = endIdx - startIdx;
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
   // Total spread: fastest to slowest
   double fastestEMA = m_emas[0].value;
   double slowestEMA = m_emas[EMA_COUNT - 1].value;
   m_fanMetrics.totalSpread = MathAbs(fastestEMA - slowestEMA);
   
   // Compression (0 = max spread, 1 = compressed)
   double refSpread = m_atr * 10;  // Reference spread
   m_fanMetrics.compression = 1.0 - MathMin(1.0, m_fanMetrics.totalSpread / refSpread);
   
   // Opening speed (rate of change of spread)
   static double prevSpread = 0;
   m_fanMetrics.openingSpeed = (m_fanMetrics.totalSpread - prevSpread) / m_atr;
   prevSpread = m_fanMetrics.totalSpread;
   
   // Order detection
   bool bullish = true, bearish = true;
   for(int i = 1; i < EMA_COUNT; i++) {
      if(m_emas[i-1].value <= m_emas[i].value) bullish = false;
      if(m_emas[i-1].value >= m_emas[i].value) bearish = false;
   }
   m_fanMetrics.bullishOrder = bullish;
   m_fanMetrics.bearishOrder = bearish;
   
   // Convergence/Divergence
   m_fanMetrics.isConverging = (m_fanMetrics.openingSpeed < -0.1);
   m_fanMetrics.isDiverging = (m_fanMetrics.openingSpeed > 0.1);
}

//+------------------------------------------------------------------+
//| Get EMA Value by Index                                           |
//+------------------------------------------------------------------+
double CEMAs::GetEMA(int index) const {
   if(index < 0 || index >= EMA_COUNT) return 0;
   return m_emas[index].value;
}

//+------------------------------------------------------------------+
//| Get EMA Slope by Index                                           |
//+------------------------------------------------------------------+
double CEMAs::GetEMASlope(int index) const {
   if(index < 0 || index >= EMA_COUNT) return 0;
   return m_emas[index].slope;
}

//+------------------------------------------------------------------+
//| Get EMA Curvature by Index                                       |
//+------------------------------------------------------------------+
double CEMAs::GetEMACurvature(int index) const {
   if(index < 0 || index >= EMA_COUNT) return 0;
   return m_emas[index].curvature;
}

//+------------------------------------------------------------------+
//| Get EMA Period by Index                                          |
//+------------------------------------------------------------------+
int CEMAs::GetEMAPeriod(int index) const {
   if(index < 0 || index >= EMA_COUNT) return 0;
   return m_emas[index].period;
}

//+------------------------------------------------------------------+
//| Get EMA Group by Index                                           |
//+------------------------------------------------------------------+
ENUM_EMA_GROUP CEMAs::GetEMAGroup(int index) const {
   if(index < 0 || index >= EMA_COUNT) return EMA_GROUP_FAST;
   return m_emas[index].group;
}

//+------------------------------------------------------------------+
//| Get EMA Handle by Index                                          |
//+------------------------------------------------------------------+
int CEMAs::GetHandle(int index) const {
   if(index < 0 || index >= EMA_COUNT) return INVALID_HANDLE;
   return m_emaHandles[index];
}

//+------------------------------------------------------------------+
//| Get Price Position Relative to EMA Cloud (-1 to 1)               |
//+------------------------------------------------------------------+
double CEMAs::GetPricePosition(double price) const {
   double cloudHigh = m_emas[0].value;
   double cloudLow = m_emas[0].value;
   
   for(int i = 1; i < EMA_COUNT; i++) {
      if(m_emas[i].value > cloudHigh) cloudHigh = m_emas[i].value;
      if(m_emas[i].value < cloudLow) cloudLow = m_emas[i].value;
   }
   
   double cloudMid = (cloudHigh + cloudLow) / 2;
   double cloudRange = cloudHigh - cloudLow;
   
   if(cloudRange <= 0) return 0;
   
   double position = (price - cloudMid) / (cloudRange / 2);
   return MathMax(-2.0, MathMin(2.0, position));
}

//+------------------------------------------------------------------+
//| Is Price Above EMA Cloud                                         |
//+------------------------------------------------------------------+
bool CEMAs::IsPriceAboveCloud(double price) const {
   for(int i = 0; i < EMA_COUNT; i++) {
      if(price <= m_emas[i].value) return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Is Price Below EMA Cloud                                         |
//+------------------------------------------------------------------+
bool CEMAs::IsPriceBelowCloud(double price) const {
   for(int i = 0; i < EMA_COUNT; i++) {
      if(price >= m_emas[i].value) return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Is Price Inside EMA Cloud                                        |
//+------------------------------------------------------------------+
bool CEMAs::IsPriceInCloud(double price) const {
   return !IsPriceAboveCloud(price) && !IsPriceBelowCloud(price);
}

#endif // CEMAS_MQH
