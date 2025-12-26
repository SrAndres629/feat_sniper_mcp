//+------------------------------------------------------------------+
//|                                                      CFEAT.mqh |
//|                    FEAT Layer - Cognitive Reading               |
//|              Form, Space, Acceleration, Time                    |
//+------------------------------------------------------------------+
#ifndef CFEAT_MQH
#define CFEAT_MQH

#include "CEMAs.mqh"

//+------------------------------------------------------------------+
//| FORM TYPES                                                       |
//+------------------------------------------------------------------+
enum ENUM_FORM_TYPE {
   FORM_CONSTRUCTION,    // Smooth curvature - building position
   FORM_IMPULSE,         // Aggressive curvature - directional move
   FORM_EXHAUSTION,      // Flattening - energy depleting
   FORM_UNDEFINED
};

//+------------------------------------------------------------------+
//| SPACE TYPES                                                      |
//+------------------------------------------------------------------+
enum ENUM_SPACE_TYPE {
   SPACE_EXPANDED,       // Wide spread - energy released
   SPACE_COMPRESSED,     // Tight spread - energy building
   SPACE_VOID,           // Gaps present - fast displacement zones
   SPACE_NORMAL
};

//+------------------------------------------------------------------+
//| ACCELERATION TYPES                                               |
//+------------------------------------------------------------------+
enum ENUM_ACCEL_TYPE {
   ACCEL_VALID,          // Acceleration with structure support
   ACCEL_FAKE,           // Acceleration without support
   ACCEL_NONE            // No significant acceleration
};

//+------------------------------------------------------------------+
//| TIME WEIGHT TYPES                                                |
//+------------------------------------------------------------------+
enum ENUM_TIME_WEIGHT {
   TIME_NOISE,           // Low TF, off-session
   TIME_RELEVANT,        // Medium TF/session
   TIME_STRUCTURAL       // High TF, key session
};

//+------------------------------------------------------------------+
//| FORM METRICS                                                     |
//+------------------------------------------------------------------+
struct SFormMetrics {
   ENUM_FORM_TYPE  type;
   double          curvatureScore;    // -1 (bearish curve) to +1 (bullish curve)
   double          fanAngle;          // Normalized fan opening angle
   double          compressionRatio;  // 0 (max spread) to 1 (compressed)
   bool            isFlattening;
   bool            isCurving;
};

//+------------------------------------------------------------------+
//| SPACE METRICS                                                    |
//+------------------------------------------------------------------+
struct SSpaceMetrics {
   ENUM_SPACE_TYPE type;
   double          fastMediumGap;     // Gap between fast and medium EMAs
   double          mediumSlowGap;     // Gap between medium and slow EMAs
   double          density;           // How tightly packed (0=loose, 1=dense)
   double          energy;            // Implied energy (inverse of compression)
   int             voidZones;         // Count of significant gaps
};

//+------------------------------------------------------------------+
//| ACCELERATION METRICS                                             |
//+------------------------------------------------------------------+
struct SAccelMetrics {
   ENUM_ACCEL_TYPE type;
   double          fanOpeningSpeed;   // Rate of spread expansion
   double          fastSeparation;    // How fast EMAs are separating from meso
   double          reactionSpeed;     // Velocity of recent move
   bool            hasStructure;      // Supported by medium/slow trend
};

//+------------------------------------------------------------------+
//| TIME METRICS                                                     |
//+------------------------------------------------------------------+
struct STimeMetrics {
   ENUM_TIME_WEIGHT weight;
   ENUM_TIMEFRAMES  currentTF;
   string           activeSession;    // Asia, London, NY, Off
   double           tfMultiplier;     // Weight for current TF
   double           sessionMultiplier;// Weight for current session
   bool             isKillzone;
};

//+------------------------------------------------------------------+
//| UNIFIED FEAT RESULT                                              |
//+------------------------------------------------------------------+
struct SFEATResult {
   SFormMetrics    form;
   SSpaceMetrics   space;
   SAccelMetrics   accel;
   STimeMetrics    time;
   double          compositeScore;    // Unified FEAT score 0-100
   bool            isValid;
};

//+------------------------------------------------------------------+
//| CFEAT CLASS                                                      |
//+------------------------------------------------------------------+
class CFEAT {
private:
   CEMAs*           m_emas;
   SFEATResult      m_result;
   
   // Thresholds (can be overridden by Python calibration)
   double           m_curvatureThreshold;
   double           m_compressionThreshold;
   double           m_accelThreshold;
   double           m_gapThreshold;
   
   // Session times (server time)
   int              m_asiaStart, m_asiaEnd;
   int              m_londonStart, m_londonEnd;
   int              m_nyStart, m_nyEnd;
   int              m_kzLondonStart, m_kzLondonEnd;
   int              m_kzNYStart, m_kzNYEnd;
   
   // Private methods
   void             CalculateForm();
   void             CalculateSpace();
   void             CalculateAccel();
   void             CalculateTime(ENUM_TIMEFRAMES tf);
   void             CalculateComposite();
   string           DetectSession(datetime time);
   bool             IsInKillzone(datetime time);
   
public:
                    CFEAT();
                   ~CFEAT();
   
   // Configuration
   void             SetEMAs(CEMAs* emas) { m_emas = emas; }
   void             SetThresholds(double curve, double compress, double accel, double gap);
   void             SetSessionTimes(int asiaS, int asiaE, int londonS, int londonE, int nyS, int nyE);
   void             SetKillzones(int kzLS, int kzLE, int kzNS, int kzNE);
   
   // Calculation
   bool             Calculate(ENUM_TIMEFRAMES tf, datetime currentTime);
   
   // Getters
   SFEATResult      GetResult() const { return m_result; }
   SFormMetrics     GetForm() const { return m_result.form; }
   SSpaceMetrics    GetSpace() const { return m_result.space; }
   SAccelMetrics    GetAccel() const { return m_result.accel; }
   STimeMetrics     GetTime() const { return m_result.time; }
   double           GetCompositeScore() const { return m_result.compositeScore; }
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CFEAT::CFEAT() {
   m_emas = NULL;
   ZeroMemory(m_result);
   
   // Default thresholds
   m_curvatureThreshold = 0.3;
   m_compressionThreshold = 0.7;
   m_accelThreshold = 0.5;
   m_gapThreshold = 1.5;
   
   // Default session times (server time, minutes from midnight)
   m_asiaStart = 0;      m_asiaEnd = 540;      // 00:00-09:00
   m_londonStart = 480;  m_londonEnd = 1020;   // 08:00-17:00
   m_nyStart = 780;      m_nyEnd = 1320;       // 13:00-22:00
   
   m_kzLondonStart = 480; m_kzLondonEnd = 600;  // 08:00-10:00
   m_kzNYStart = 780;     m_kzNYEnd = 900;      // 13:00-15:00
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CFEAT::~CFEAT() {
   m_emas = NULL;
}

//+------------------------------------------------------------------+
//| Set Thresholds                                                   |
//+------------------------------------------------------------------+
void CFEAT::SetThresholds(double curve, double compress, double accel, double gap) {
   m_curvatureThreshold = curve;
   m_compressionThreshold = compress;
   m_accelThreshold = accel;
   m_gapThreshold = gap;
}

//+------------------------------------------------------------------+
//| Set Session Times                                                |
//+------------------------------------------------------------------+
void CFEAT::SetSessionTimes(int asiaS, int asiaE, int londonS, int londonE, int nyS, int nyE) {
   m_asiaStart = asiaS;   m_asiaEnd = asiaE;
   m_londonStart = londonS; m_londonEnd = londonE;
   m_nyStart = nyS;       m_nyEnd = nyE;
}

//+------------------------------------------------------------------+
//| Set Killzones                                                    |
//+------------------------------------------------------------------+
void CFEAT::SetKillzones(int kzLS, int kzLE, int kzNS, int kzNE) {
   m_kzLondonStart = kzLS; m_kzLondonEnd = kzLE;
   m_kzNYStart = kzNS;     m_kzNYEnd = kzNE;
}

//+------------------------------------------------------------------+
//| Calculate All FEAT Components                                    |
//+------------------------------------------------------------------+
bool CFEAT::Calculate(ENUM_TIMEFRAMES tf, datetime currentTime) {
   if(m_emas == NULL || !m_emas.IsReady()) {
      m_result.isValid = false;
      return false;
   }
   
   CalculateForm();
   CalculateSpace();
   CalculateAccel();
   CalculateTime(tf);
   CalculateComposite();
   
   m_result.isValid = true;
   return true;
}

//+------------------------------------------------------------------+
//| Calculate Form (F)                                               |
//| Curvature, fan angle, compression                                |
//+------------------------------------------------------------------+
void CFEAT::CalculateForm() {
   SFanMetrics fan = m_emas.GetFanMetrics();
   SEMAGroupMetrics fast = m_emas.GetFastMetrics();
   SEMAGroupMetrics medium = m_emas.GetMediumMetrics();
   SEMAGroupMetrics slow = m_emas.GetSlowMetrics();
   
   // Curvature score: average curvature across groups
   double avgCurvature = 0;
   for(int i = 0; i < EMA_COUNT; i++) {
      avgCurvature += m_emas.GetEMACurvature(i);
   }
   avgCurvature /= EMA_COUNT;
   m_result.form.curvatureScore = MathMax(-1.0, MathMin(1.0, avgCurvature * 10));
   
   // Fan angle (based on spread and order)
   if(fan.bullishOrder) {
      m_result.form.fanAngle = 1.0 - fan.compression;
   } else if(fan.bearishOrder) {
      m_result.form.fanAngle = -(1.0 - fan.compression);
   } else {
      m_result.form.fanAngle = 0;
   }
   
   // Compression ratio
   m_result.form.compressionRatio = fan.compression;
   
   // Form type determination
   m_result.form.isFlattening = (MathAbs(avgCurvature) < 0.05 && fan.isConverging);
   m_result.form.isCurving = (MathAbs(avgCurvature) > m_curvatureThreshold);
   
   if(m_result.form.isFlattening) {
      m_result.form.type = FORM_EXHAUSTION;
   } else if(MathAbs(m_result.form.curvatureScore) > 0.5) {
      m_result.form.type = FORM_IMPULSE;
   } else if(m_result.form.isCurving) {
      m_result.form.type = FORM_CONSTRUCTION;
   } else {
      m_result.form.type = FORM_UNDEFINED;
   }
}

//+------------------------------------------------------------------+
//| Calculate Space (E)                                              |
//| Distances, density, gaps                                         |
//+------------------------------------------------------------------+
void CFEAT::CalculateSpace() {
   SEMAGroupMetrics fast = m_emas.GetFastMetrics();
   SEMAGroupMetrics medium = m_emas.GetMediumMetrics();
   SEMAGroupMetrics slow = m_emas.GetSlowMetrics();
   double atr = m_emas.GetATR();
   
   // Gap between groups
   m_result.space.fastMediumGap = MathAbs(fast.avgValue - medium.avgValue) / atr;
   m_result.space.mediumSlowGap = MathAbs(medium.avgValue - slow.avgValue) / atr;
   
   // Density (inverse of total spread)
   SFanMetrics fan = m_emas.GetFanMetrics();
   m_result.space.density = fan.compression;
   
   // Energy (inverse of compression - more spread = more energy released)
   m_result.space.energy = 1.0 - fan.compression;
   
   // Count void zones (significant gaps)
   m_result.space.voidZones = 0;
   if(m_result.space.fastMediumGap > m_gapThreshold) m_result.space.voidZones++;
   if(m_result.space.mediumSlowGap > m_gapThreshold) m_result.space.voidZones++;
   
   // Space type
   if(m_result.space.voidZones > 0) {
      m_result.space.type = SPACE_VOID;
   } else if(m_result.space.energy > 0.6) {
      m_result.space.type = SPACE_EXPANDED;
   } else if(m_result.space.density > m_compressionThreshold) {
      m_result.space.type = SPACE_COMPRESSED;
   } else {
      m_result.space.type = SPACE_NORMAL;
   }
}

//+------------------------------------------------------------------+
//| Calculate Acceleration (A)                                       |
//| Fan opening speed, separation, reaction                          |
//+------------------------------------------------------------------+
void CFEAT::CalculateAccel() {
   SFanMetrics fan = m_emas.GetFanMetrics();
   SEMAGroupMetrics fast = m_emas.GetFastMetrics();
   SEMAGroupMetrics medium = m_emas.GetMediumMetrics();
   SEMAGroupMetrics slow = m_emas.GetSlowMetrics();
   
   m_result.accel.fanOpeningSpeed = fan.openingSpeed;
   m_result.accel.fastSeparation = MathAbs(fast.avgSlope - medium.avgSlope);
   m_result.accel.reactionSpeed = MathAbs(fast.avgSlope);
   
   // Has structure: medium and slow aligned with fast direction
   bool fastBullish = (fast.avgSlope > 0);
   bool mediumSupports = (fastBullish && medium.avgSlope > 0) || (!fastBullish && medium.avgSlope < 0);
   bool slowSupports = (fastBullish && slow.avgSlope >= 0) || (!fastBullish && slow.avgSlope <= 0);
   m_result.accel.hasStructure = mediumSupports && slowSupports;
   
   // Acceleration type
   bool hasAccel = (MathAbs(m_result.accel.fanOpeningSpeed) > m_accelThreshold || 
                    m_result.accel.fastSeparation > m_accelThreshold);
   
   if(hasAccel && m_result.accel.hasStructure) {
      m_result.accel.type = ACCEL_VALID;
   } else if(hasAccel && !m_result.accel.hasStructure) {
      m_result.accel.type = ACCEL_FAKE;
   } else {
      m_result.accel.type = ACCEL_NONE;
   }
}

//+------------------------------------------------------------------+
//| Calculate Time (T)                                               |
//| TF weight, session, killzone                                     |
//+------------------------------------------------------------------+
void CFEAT::CalculateTime(ENUM_TIMEFRAMES tf) {
   m_result.time.currentTF = tf;
   
   // TF multiplier (higher TF = more weight)
   switch(tf) {
      case PERIOD_M1:  case PERIOD_M5:  m_result.time.tfMultiplier = 0.3; break;
      case PERIOD_M15: case PERIOD_M30: m_result.time.tfMultiplier = 0.5; break;
      case PERIOD_H1:  case PERIOD_H2:  m_result.time.tfMultiplier = 0.7; break;
      case PERIOD_H4:  case PERIOD_D1:  m_result.time.tfMultiplier = 1.0; break;
      default: m_result.time.tfMultiplier = 0.5;
   }
   
   // Session detection
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int currentMinutes = dt.hour * 60 + dt.min;
   
   if(currentMinutes >= m_londonStart && currentMinutes < m_londonEnd) {
      m_result.time.activeSession = "London";
      m_result.time.sessionMultiplier = 1.0;
   } else if(currentMinutes >= m_nyStart && currentMinutes < m_nyEnd) {
      m_result.time.activeSession = "NY";
      m_result.time.sessionMultiplier = 1.0;
   } else if(currentMinutes >= m_asiaStart && currentMinutes < m_asiaEnd) {
      m_result.time.activeSession = "Asia";
      m_result.time.sessionMultiplier = 0.6;
   } else {
      m_result.time.activeSession = "Off";
      m_result.time.sessionMultiplier = 0.3;
   }
   
   // Killzone check
   m_result.time.isKillzone = (currentMinutes >= m_kzLondonStart && currentMinutes < m_kzLondonEnd) ||
                              (currentMinutes >= m_kzNYStart && currentMinutes < m_kzNYEnd);
   
   if(m_result.time.isKillzone) m_result.time.sessionMultiplier = 1.2;
   
   // Time weight
   if(m_result.time.tfMultiplier < 0.5 && m_result.time.sessionMultiplier < 0.5) {
      m_result.time.weight = TIME_NOISE;
   } else if(m_result.time.tfMultiplier >= 0.7 || m_result.time.isKillzone) {
      m_result.time.weight = TIME_STRUCTURAL;
   } else {
      m_result.time.weight = TIME_RELEVANT;
   }
}

//+------------------------------------------------------------------+
//| Calculate Composite FEAT Score                                   |
//+------------------------------------------------------------------+
void CFEAT::CalculateComposite() {
   // Form contribution (0-25)
   double formScore = 0;
   if(m_result.form.type == FORM_IMPULSE) formScore = 25;
   else if(m_result.form.type == FORM_CONSTRUCTION) formScore = 18;
   else if(m_result.form.type == FORM_EXHAUSTION) formScore = 5;
   else formScore = 12;
   
   // Space contribution (0-25)
   double spaceScore = m_result.space.energy * 25;
   
   // Acceleration contribution (0-25)
   double accelScore = 0;
   if(m_result.accel.type == ACCEL_VALID) accelScore = 25;
   else if(m_result.accel.type == ACCEL_FAKE) accelScore = 5;
   else accelScore = 12;
   
   // Time contribution (0-25)
   double timeScore = m_result.time.tfMultiplier * m_result.time.sessionMultiplier * 25;
   
   m_result.compositeScore = formScore + spaceScore + accelScore + timeScore;
}

#endif // CFEAT_MQH
