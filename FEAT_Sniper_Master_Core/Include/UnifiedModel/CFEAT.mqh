//+------------------------------------------------------------------+
//|                                                      CFEAT.mqh |
//|                    FEAT Layer - Cognitive Reading               |
//+------------------------------------------------------------------+
#ifndef CFEAT_MQH
#define CFEAT_MQH

#include "CEMAs.mqh"
#include "CLiquidity.mqh"

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
   bool            hasHCH;         // Head & Shoulders pattern
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
   string          activeZoneType; 
};

struct SAccelMetrics {
   ENUM_ACCEL_TYPE type;
   double          fanOpeningSpeed;
   double          fastSeparation;
   double          velocity;
   double          volumeBodyRatio;
   double          atrProjected;
   double          momentum;
   double          deltaFlow;       // Institutional order flow approximation
   double          rsi;             // RSI(14)
   double          macdHist;        // MACD histogram
   double          ao;              // Awesome Oscillator
   double          ac;              // Acceleration Oscillator
   bool            isInstitutional;
   bool            isExhausted;
};

struct STimeMetrics {
   ENUM_TIME_WEIGHT weight;
   ENUM_TIMEFRAMES  currentTF;
   string           activeSession;
   double           tfMultiplier;
   double           sessionMultiplier;
   bool             isKillzone;
   bool             isLondonKZ;
   bool             isNYKZ;
   bool             isAgainstH4;     // Trading against H4 strong close
   int              h4Direction;     // 1=bullish, -1=bearish, 0=neutral
   int              currentHour;
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
   double           m_prevVelocity;
   double           m_prevMomentum;     // For AC calculation
   
   // Indicator handles for momentum
   int              m_rsiHandle;
   int              m_macdHandle;
   int              m_aoHandle;
   string           m_symbol;
   ENUM_TIMEFRAMES  m_timeframe;
   bool             m_initialized;

   void             CalcForm(double open, double high, double low, double close, double volume);
   void             CalcSpace(double price);
   void             CalcAccel(double open, double high, double low, double close, double volume);
   void             CalcTime(ENUM_TIMEFRAMES tf, datetime t);
   void             CalcH4Coordination(datetime t);
   void             CalcComp();

public:
   CFEAT();
   ~CFEAT() { Deinit(); }

   bool Init(string symbol, ENUM_TIMEFRAMES tf);
   void Deinit();
   void SetEMAs(CEMAs* e) { m_ptrEmas = e; }
   void SetLiquidity(CLiquidity* l) { m_ptrLiq = l; }

   bool Calculate(ENUM_TIMEFRAMES tf, datetime t, double open, double high, double low, double close, double volume);
   
   SFEATResult   GetResult() const { return m_result; }
   SFormMetrics  GetForm() const { return m_result.form; }
   SSpaceMetrics GetSpace() const { return m_result.space; }
   SAccelMetrics GetAccel() const { return m_result.accel; }
   STimeMetrics  GetTime() const { return m_result.time; }
   double        GetCompositeScore() const { return m_result.compositeScore; }
};

CFEAT::CFEAT() : m_ptrEmas(NULL), m_ptrLiq(NULL), m_curveTh(0.3), m_compTh(0.7), m_accelTh(0.5), m_gapTh(1.5), m_prevVelocity(0), m_prevMomentum(0), m_rsiHandle(INVALID_HANDLE), m_macdHandle(INVALID_HANDLE), m_aoHandle(INVALID_HANDLE), m_initialized(false) {
   ZeroMemory(m_result);
}

bool CFEAT::Init(string symbol, ENUM_TIMEFRAMES tf) {
   m_symbol = symbol;
   m_timeframe = tf;
   
   m_rsiHandle = iRSI(symbol, tf, 14, PRICE_CLOSE);
   m_macdHandle = iMACD(symbol, tf, 12, 26, 9, PRICE_CLOSE);
   m_aoHandle = iAO(symbol, tf);
   
   if(m_rsiHandle == INVALID_HANDLE || m_macdHandle == INVALID_HANDLE || m_aoHandle == INVALID_HANDLE) {
      Print("[CFEAT] Error creating indicator handles");
      Deinit();
      return false;
   }
   m_initialized = true;
   return true;
}

void CFEAT::Deinit() {
   if(m_rsiHandle != INVALID_HANDLE) { IndicatorRelease(m_rsiHandle); m_rsiHandle = INVALID_HANDLE; }
   if(m_macdHandle != INVALID_HANDLE) { IndicatorRelease(m_macdHandle); m_macdHandle = INVALID_HANDLE; }
   if(m_aoHandle != INVALID_HANDLE) { IndicatorRelease(m_aoHandle); m_aoHandle = INVALID_HANDLE; }
   m_initialized = false;
}

bool CFEAT::Calculate(ENUM_TIMEFRAMES tf, datetime t, double open, double high, double low, double close, double volume) {
   if(m_ptrEmas == NULL) return false;
   if(!m_ptrEmas.IsReady()) return false;
   CalcForm(open, high, low, close, volume); 
   CalcSpace(close); 
   CalcAccel(open, high, low, close, volume); 
   CalcTime(tf, t);
   CalcH4Coordination(t);  // NEW: Check H4 bias alignment
   CalcComp();
   m_result.isValid = true;
   return true;
}

void CFEAT::CalcForm(double open, double high, double low, double close, double volume) {
   if(m_ptrEmas == NULL) return;
   SFanMetrics fan = m_ptrEmas.GetFanMetrics();
   double avgC = 0; for(int i=0; i<30; i++) avgC += m_ptrEmas.GetEMACurvature(i); avgC /= 30.0;
   m_result.form.curvatureScore = MathMax(-1.0, MathMin(1.0, avgC * 10));
   m_result.form.compressionRatio = fan.compression;
   m_result.form.hasBOS = false; m_result.form.hasCHoCH = false; m_result.form.hasHCH = false;
   
   if(m_ptrLiq != NULL) {
      int sCount = m_ptrLiq.GetStructureCount();
      if(sCount > 0) {
         SStructureEvent last = m_ptrLiq.GetStructureEvent(sCount-1);
         m_result.form.hasBOS = (last.type == STRUCT_BOS);
         m_result.form.hasCHoCH = (last.type == STRUCT_CHOCH);
      }
      
      // HCH Detection: Look for HL-HH-LH-LL (bearish) or LH-LL-HL-HH (bullish) sequence
      int swCount = m_ptrLiq.GetSwingCount();
      if(swCount >= 4) {
         SSwingPoint s0 = m_ptrLiq.GetSwingPoint(0);
         SSwingPoint s1 = m_ptrLiq.GetSwingPoint(1);
         SSwingPoint s2 = m_ptrLiq.GetSwingPoint(2);
         SSwingPoint s3 = m_ptrLiq.GetSwingPoint(3);
         
         // Bearish HCH: HL -> HH (head) -> LH (lower high, shoulder) -> break
         if(s3.type == SWING_HL && s2.type == SWING_HH && s1.type == SWING_LH) {
            if(s2.price > s3.price && s2.price > s1.price && s1.price < s3.price * 1.01) {
               m_result.form.hasHCH = true;
            }
         }
         // Bullish Inverse HCH: LH -> LL (head) -> HL (higher low, shoulder) -> break
         if(s3.type == SWING_LH && s2.type == SWING_LL && s1.type == SWING_HL) {
            if(s2.price < s3.price && s2.price < s1.price && s1.price > s3.price * 0.99) {
               m_result.form.hasHCH = true;
            }
         }
      }
   }
   
   double body = MathAbs(close - open); double rC = high - low; double atr = m_ptrEmas.GetATR();
   m_result.form.isIntentCandle = (body > atr * 0.8) && (rC > 0 && body / rC > 0.7); 
   m_result.form.type = (MathAbs(m_result.form.curvatureScore) > 0.5 ? FORM_IMPULSE : FORM_CONSTRUCTION);
}

void CFEAT::CalcSpace(double price) {
   if(m_ptrEmas == NULL) return;
   SEMAGroupMetrics mic = m_ptrEmas.GetMicroMetrics(); 
   SEMAGroupMetrics opr = m_ptrEmas.GetOperationalMetrics(); 
   double atr = m_ptrEmas.GetATR();
   m_result.space.fastMediumGap = MathAbs(mic.avgValue - opr.avgValue)/atr;
   m_result.space.density = m_ptrEmas.GetFanMetrics().compression;
   m_result.space.proximityScore = 0; m_result.space.atZone = false;
   
   if(m_ptrLiq != NULL) {
      int zCount = m_ptrLiq.GetZoneCount();
      for(int i = 0; i < zCount; i++) {
         SInstitutionalZone z = m_ptrLiq.GetZone(i);
         if(z.mitigated) continue;
         if(price <= z.top && price >= z.bottom) {
            m_result.space.atZone = true;
            m_result.space.proximityScore = 1.0;
            m_result.space.activeZoneType = z.label;
            break;
         }
         double dist = MathMin(MathAbs(price - z.top), MathAbs(price - z.bottom));
         if(dist < atr * 2) {
            double score = 1.0 - (dist / (atr * 2));
            if(score > m_result.space.proximityScore) {
               m_result.space.proximityScore = score;
               m_result.space.activeZoneType = z.label;
            }
         }
      }
   }
}

void CFEAT::CalcAccel(double open, double high, double low, double close, double volume) {
   if(m_ptrEmas == NULL) return;
   SFanMetrics fan = m_ptrEmas.GetFanMetrics();
   double atr = m_ptrEmas.GetATR();
   
   double body = MathAbs(close - open);
   double currVelocity = (atr > 0) ? (body / atr) : 0;
   m_result.accel.velocity = currVelocity;
   m_result.accel.volumeBodyRatio = (volume > 0) ? (body / volume * 1000) : 0;
   m_result.accel.fanOpeningSpeed = fan.openingSpeed;
   m_result.accel.momentum = fan.openingSpeed * m_result.accel.velocity;
   m_result.accel.isInstitutional = (body > atr * 1.2) && (volume > atr * 500);
   m_result.accel.atrProjected = atr * 1.5;
   
   // Delta Flow: Institutional order flow approximation
   // Positive = bullish pressure, Negative = bearish pressure
   double direction = (close > open) ? 1.0 : -1.0;
   m_result.accel.deltaFlow = direction * body * (volume / 1000.0) / (atr > 0 ? atr : 0.0001);
   
   // Get RSI value
   m_result.accel.rsi = 50.0;  // Default
   if(m_initialized && m_rsiHandle != INVALID_HANDLE) {
      double rsiBuffer[1];
      if(CopyBuffer(m_rsiHandle, 0, 0, 1, rsiBuffer) > 0) {
         m_result.accel.rsi = rsiBuffer[0];
      }
   }
   
   // Get MACD Histogram
   m_result.accel.macdHist = 0.0;
   if(m_initialized && m_macdHandle != INVALID_HANDLE) {
      double macdBuffer[1];
      if(CopyBuffer(m_macdHandle, 2, 0, 1, macdBuffer) > 0) {  // Buffer 2 = histogram
         m_result.accel.macdHist = macdBuffer[0];
      }
   }
   
   // Get AO (Awesome Oscillator)
   m_result.accel.ao = 0.0;
   if(m_initialized && m_aoHandle != INVALID_HANDLE) {
      double aoBuffer[1];
      if(CopyBuffer(m_aoHandle, 0, 0, 1, aoBuffer) > 0) {
         m_result.accel.ao = aoBuffer[0];
      }
   }
   
   // AC (Acceleration Oscillator): Difference between current and previous momentum
   m_result.accel.ac = m_result.accel.momentum - m_prevMomentum;
   m_prevMomentum = m_result.accel.momentum;
   
   // Exhaustion Detection: velocity decreasing significantly
   m_result.accel.isExhausted = (m_prevVelocity > 0.5 && currVelocity < m_prevVelocity * 0.6);
   m_prevVelocity = currVelocity;
   
   m_result.accel.type = (m_result.accel.velocity > 1.0) ? ACCEL_VALID : ACCEL_NONE;
}

void CFEAT::CalcTime(ENUM_TIMEFRAMES tf, datetime t) {
   m_result.time.currentTF = tf;
   
   MqlDateTime dt;
   TimeToStruct(t, dt);
   int hour = dt.hour;
   m_result.time.currentHour = hour;
   
   // Kill Zone Detection (Broker Time - Adjust offset as needed)
   // London Open: 02:00-05:00 (UTC-4 = 06:00-09:00 UTC)
   // NY Open: 07:00-10:00 (UTC-4 = 11:00-14:00 UTC)
   m_result.time.isLondonKZ = (hour >= 2 && hour < 5);
   m_result.time.isNYKZ = (hour >= 7 && hour < 10);
   m_result.time.isKillzone = m_result.time.isLondonKZ || m_result.time.isNYKZ;
   
   // Session Detection
   if(hour >= 0 && hour < 9) m_result.time.activeSession = "ASIA";
   else if(hour >= 8 && hour < 17) m_result.time.activeSession = "LONDON";
   else if(hour >= 13 && hour < 22) m_result.time.activeSession = "NY";
   else m_result.time.activeSession = "OFF";
   
   // Time Weight
   if(m_result.time.isKillzone) {
      m_result.time.weight = TIME_STRUCTURAL;
      m_result.time.sessionMultiplier = 1.5;
   } else if(m_result.time.activeSession != "OFF") {
      m_result.time.weight = TIME_RELEVANT;
      m_result.time.sessionMultiplier = 1.0;
   } else {
      m_result.time.weight = TIME_NOISE;
      m_result.time.sessionMultiplier = 0.5;
   }
   
   // TF Multiplier (Higher TF = More Weight)
   switch(tf) {
      case PERIOD_M1: case PERIOD_M5: m_result.time.tfMultiplier = 0.5; break;
      case PERIOD_M15: case PERIOD_M30: m_result.time.tfMultiplier = 0.75; break;
      case PERIOD_H1: m_result.time.tfMultiplier = 1.0; break;
      case PERIOD_H4: m_result.time.tfMultiplier = 1.5; break;
      case PERIOD_D1: m_result.time.tfMultiplier = 2.0; break;
      default: m_result.time.tfMultiplier = 1.0;
   }
   
   // Initialize H4 flags (will be populated by CalcH4Coordination)
   m_result.time.isAgainstH4 = false;
   m_result.time.h4Direction = 0;
}

//+------------------------------------------------------------------+
//| H4 Coordination - Never trade against strong H4 close           |
//+------------------------------------------------------------------+
void CFEAT::CalcH4Coordination(datetime t) {
   // Get H4 candle data
   MqlRates h4Rates[];
   ArraySetAsSeries(h4Rates, true);
   
   if(CopyRates(m_symbol, PERIOD_H4, 0, 2, h4Rates) < 2) {
      m_result.time.h4Direction = 0;
      m_result.time.isAgainstH4 = false;
      return;
   }
   
   // Determine H4 direction based on last closed candle
   double h4Body = h4Rates[1].close - h4Rates[1].open;
   double h4Range = h4Rates[1].high - h4Rates[1].low;
   double h4BodyRatio = (h4Range > 0) ? MathAbs(h4Body) / h4Range : 0;
   
   // Strong close = body > 60% of range
   if(h4BodyRatio > 0.6) {
      m_result.time.h4Direction = (h4Body > 0) ? 1 : -1;
   } else {
      m_result.time.h4Direction = 0;  // Neutral/indecisive
   }
   
   // Check if current micro intention is against H4 bias
   if(m_result.time.h4Direction != 0 && m_result.form.isIntentCandle) {
      // Check micro EMA slope direction
      double microSlope = 0;
      if(m_ptrEmas != NULL) {
         SEMAGroupMetrics micro = m_ptrEmas.GetMicroMetrics();
         microSlope = micro.avgSlope;
      }
      
      // Against H4 if micro slope opposes H4 direction
      bool microBullish = (microSlope > 0.1);
      bool microBearish = (microSlope < -0.1);
      
      m_result.time.isAgainstH4 = (m_result.time.h4Direction > 0 && microBearish) ||
                                   (m_result.time.h4Direction < 0 && microBullish);
   }
}

void CFEAT::CalcComp() {
   m_result.compositeScore = 20.0; // Base
   m_result.compositeScore += m_result.space.proximityScore * 25.0;
   m_result.compositeScore += MathMin(20.0, m_result.accel.velocity * 10.0);
   if(m_result.form.hasBOS) m_result.compositeScore += 10.0;
   if(m_result.form.hasCHoCH) m_result.compositeScore += 8.0;   // CHoCH bonus
   if(m_result.form.hasHCH) m_result.compositeScore += 12.0;    // HCH pattern bonus
   if(m_result.form.isIntentCandle) m_result.compositeScore += 10.0;
   if(m_result.time.isKillzone) m_result.compositeScore += 15.0;
   
   // RSI confluence bonus (oversold/overbought confirmation)
   if(m_result.accel.rsi < 30 || m_result.accel.rsi > 70) {
      m_result.compositeScore += 5.0;
   }
   
   // PENALTIES
   if(m_result.accel.isExhausted) m_result.compositeScore -= 20.0;
   
   // H4 Coordination: Reduce score if trading against H4
   // Priority: Time > Accel > Space > Form
   if(m_result.time.isAgainstH4) {
      m_result.compositeScore *= 0.5;  // 50% reduction
   }
   
   // Non-killzone penalty
   if(!m_result.time.isKillzone) {
      m_result.compositeScore *= 0.7;  // 30% reduction
   }
   
   m_result.compositeScore = MathMax(0, MathMin(100.0, m_result.compositeScore));
}

#endif
