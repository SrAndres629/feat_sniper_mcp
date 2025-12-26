//+------------------------------------------------------------------+
//|                                                       CFSM.mqh |
//|                    FSM Layer - State Classification             |
//|        ACCUMULATION, EXPANSION, DISTRIBUTION, RESET             |
//+------------------------------------------------------------------+
#ifndef CFSM_MQH
#define CFSM_MQH

#include "CEMAs.mqh"
#include "CFEAT.mqh"
#include "CLiquidity.mqh"

//+------------------------------------------------------------------+
//| MARKET STATE ENUMERATION                                         |
//+------------------------------------------------------------------+
enum ENUM_MARKET_STATE {
   STATE_ACCUMULATION,   // Building position, compressed structure
   STATE_EXPANSION,      // Directional move with structure support
   STATE_DISTRIBUTION,   // Profit taking, momentum decay
   STATE_RESET           // Quick mean reversion
};

//+------------------------------------------------------------------+
//| STATE TRANSITION RESULT                                          |
//+------------------------------------------------------------------+
struct SStateTransition {
   ENUM_MARKET_STATE  prevState;
   ENUM_MARKET_STATE  newState;
   datetime           transitionTime;
   double             confidence;
   string             reason;
};

//+------------------------------------------------------------------+
//| FSM THRESHOLDS (can be loaded from Python calibration)           |
//+------------------------------------------------------------------+
struct SFSMThresholds {
   // Effort/Result percentiles
   double   effortP80;
   double   effortP50;
   double   effortP20;
   double   resultP80;
   double   resultP50;
   double   resultP20;
   
   // State-specific thresholds
   double   accumulationCompression;  // Min compression for accumulation
   double   expansionSlope;           // Min slope for expansion
   double   distributionMomentum;     // Momentum decay threshold
   double   resetSpeed;               // Speed threshold for reset
   
   // Hysteresis
   double   hysteresisMargin;         // % margin to prevent flip-flop
   int      minBarsInState;           // Minimum bars before allowing transition
};

//+------------------------------------------------------------------+
//| FSM METRICS                                                      |
//+------------------------------------------------------------------+
struct SFSMMetrics {
   double   effort;         // Volume effort (normalized)
   double   result;         // Price result (normalized)
   double   compression;    // EMA compression
   double   slope;          // Macro EMA slope
   double   momentum;       // Rate of change of result
   double   speed;          // Current price velocity
   bool     isValid;
};

//+------------------------------------------------------------------+
//| FSM STATE CONTEXT                                                |
//+------------------------------------------------------------------+
struct SFSMContext {
   ENUM_MARKET_STATE  state;
   double             confidence;      // 0-100
   int                barsInState;
   datetime           lastTransition;
   SStateTransition   lastTransitionInfo;
   SFSMMetrics        metrics;
   bool               isStable;        // No recent flip-flops
};

//+------------------------------------------------------------------+
//| CFSM CLASS                                                       |
//+------------------------------------------------------------------+
class CFSM {
private:
   CEMAs*            m_emas;
   CFEAT*            m_feat;
   CLiquidity*       m_liquidity;
   
   SFSMContext       m_context;
   SFSMThresholds    m_thresholds;
   SStateTransition  m_history[];
   
   int               m_historySize;
   double            m_confidenceEMA;
   int               m_flipFlopCount;
   datetime          m_lastFlipFlopCheck;
   
   // Rolling buffers for percentile calculation
   double            m_effortBuffer[];
   double            m_resultBuffer[];
   int               m_bufferSize;
   int               m_bufferIndex;
   bool              m_bufferFull;
   
   // Private methods
   void              CalculateMetrics(double currentPrice, double prevPrice, double volume);
   ENUM_MARKET_STATE DetermineState();
   double            CalculateConfidence(ENUM_MARKET_STATE proposedState);
   void              ApplyHysteresis(ENUM_MARKET_STATE &proposedState);
   void              RecordTransition(ENUM_MARKET_STATE newState, string reason);
   void              UpdatePercentiles();
   double            GetPercentile(const double &buffer[], double value);
   
public:
                     CFSM();
                    ~CFSM();
   
   // Configuration
   void              SetComponents(CEMAs* emas, CFEAT* feat, CLiquidity* liq);
   void              SetThresholds(const SFSMThresholds &thresholds);
   bool              LoadThresholdsFromFile(string filename);
   void              SetBufferSize(int size);
   
   // Calculation
   bool              Calculate(double currentPrice, double prevPrice, double volume);
   
   // Getters
   SFSMContext       GetContext() const { return m_context; }
   ENUM_MARKET_STATE GetState() const { return m_context.state; }
   double            GetConfidence() const { return m_context.confidence; }
   SFSMMetrics       GetMetrics() const { return m_context.metrics; }
   string            GetStateString() const;
   
   // History
   int               GetHistoryCount() const { return ArraySize(m_history); }
   SStateTransition  GetHistoryItem(int index) const;
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CFSM::CFSM() {
   m_emas = NULL;
   m_feat = NULL;
   m_liquidity = NULL;
   
   ZeroMemory(m_context);
   m_context.state = STATE_ACCUMULATION;
   m_context.confidence = 50;
   
   // Default thresholds (should be calibrated via Python)
   m_thresholds.effortP80 = 0.8;
   m_thresholds.effortP50 = 0.5;
   m_thresholds.effortP20 = 0.2;
   m_thresholds.resultP80 = 0.8;
   m_thresholds.resultP50 = 0.5;
   m_thresholds.resultP20 = 0.2;
   
   m_thresholds.accumulationCompression = 0.7;
   m_thresholds.expansionSlope = 0.3;
   m_thresholds.distributionMomentum = -0.2;
   m_thresholds.resetSpeed = 2.0;
   
   m_thresholds.hysteresisMargin = 0.1;
   m_thresholds.minBarsInState = 3;
   
   m_historySize = 100;
   m_confidenceEMA = 50;
   m_flipFlopCount = 0;
   m_lastFlipFlopCheck = 0;
   
   m_bufferSize = 100;
   m_bufferIndex = 0;
   m_bufferFull = false;
   ArrayResize(m_effortBuffer, m_bufferSize);
   ArrayResize(m_resultBuffer, m_bufferSize);
   ArrayInitialize(m_effortBuffer, 0);
   ArrayInitialize(m_resultBuffer, 0);
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CFSM::~CFSM() {
   ArrayFree(m_history);
   ArrayFree(m_effortBuffer);
   ArrayFree(m_resultBuffer);
}

//+------------------------------------------------------------------+
//| Set Components                                                   |
//+------------------------------------------------------------------+
void CFSM::SetComponents(CEMAs* emas, CFEAT* feat, CLiquidity* liq) {
   m_emas = emas;
   m_feat = feat;
   m_liquidity = liq;
}

//+------------------------------------------------------------------+
//| Set Thresholds                                                   |
//+------------------------------------------------------------------+
void CFSM::SetThresholds(const SFSMThresholds &thresholds) {
   m_thresholds = thresholds;
}

//+------------------------------------------------------------------+
//| Load Thresholds from File (Python calibration output)            |
//+------------------------------------------------------------------+
bool CFSM::LoadThresholdsFromFile(string filename) {
   int handle = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE) {
      Print("[CFSM] Cannot open thresholds file: ", filename);
      return false;
   }
   
   // Simple line-by-line parsing for key=value format
   while(!FileIsEnding(handle)) {
      string line = FileReadString(handle);
      string parts[];
      int count = StringSplit(line, '=', parts);
      if(count != 2) continue;
      
      string key = parts[0];
      double value = StringToDouble(parts[1]);
      
      if(key == "effortP80") m_thresholds.effortP80 = value;
      else if(key == "effortP50") m_thresholds.effortP50 = value;
      else if(key == "effortP20") m_thresholds.effortP20 = value;
      else if(key == "resultP80") m_thresholds.resultP80 = value;
      else if(key == "resultP50") m_thresholds.resultP50 = value;
      else if(key == "resultP20") m_thresholds.resultP20 = value;
      else if(key == "accumulationCompression") m_thresholds.accumulationCompression = value;
      else if(key == "expansionSlope") m_thresholds.expansionSlope = value;
      else if(key == "distributionMomentum") m_thresholds.distributionMomentum = value;
      else if(key == "resetSpeed") m_thresholds.resetSpeed = value;
      else if(key == "hysteresisMargin") m_thresholds.hysteresisMargin = value;
      else if(key == "minBarsInState") m_thresholds.minBarsInState = (int)value;
   }
   
   FileClose(handle);
   Print("[CFSM] Loaded thresholds from: ", filename);
   return true;
}

//+------------------------------------------------------------------+
//| Set Buffer Size                                                  |
//+------------------------------------------------------------------+
void CFSM::SetBufferSize(int size) {
   m_bufferSize = MathMax(20, size);
   ArrayResize(m_effortBuffer, m_bufferSize);
   ArrayResize(m_resultBuffer, m_bufferSize);
   ArrayInitialize(m_effortBuffer, 0);
   ArrayInitialize(m_resultBuffer, 0);
   m_bufferIndex = 0;
   m_bufferFull = false;
}

//+------------------------------------------------------------------+
//| Main Calculation                                                 |
//+------------------------------------------------------------------+
bool CFSM::Calculate(double currentPrice, double prevPrice, double volume) {
   if(m_emas == NULL || m_feat == NULL) return false;
   
   // Calculate metrics
   CalculateMetrics(currentPrice, prevPrice, volume);
   
   // Update rolling buffers
   m_effortBuffer[m_bufferIndex] = m_context.metrics.effort;
   m_resultBuffer[m_bufferIndex] = m_context.metrics.result;
   m_bufferIndex = (m_bufferIndex + 1) % m_bufferSize;
   if(m_bufferIndex == 0) m_bufferFull = true;
   
   // Update dynamic percentiles
   UpdatePercentiles();
   
   // Determine proposed state
   ENUM_MARKET_STATE proposedState = DetermineState();
   
   // Apply hysteresis
   ApplyHysteresis(proposedState);
   
   // Calculate confidence
   double confidence = CalculateConfidence(proposedState);
   
   // EMA smooth confidence
   double alpha = 2.0 / 6.0;  // 5-period EMA
   m_confidenceEMA = alpha * confidence + (1 - alpha) * m_confidenceEMA;
   m_context.confidence = m_confidenceEMA;
   
   // Check for state transition
   if(proposedState != m_context.state) {
      if(m_context.barsInState >= m_thresholds.minBarsInState) {
         RecordTransition(proposedState, "State criteria met");
         m_context.state = proposedState;
         m_context.barsInState = 0;
      }
   } else {
      m_context.barsInState++;
   }
   
   // Check for flip-flop (instability)
   datetime now = TimeCurrent();
   if(now - m_lastFlipFlopCheck > 3600) {  // Check every hour
      m_flipFlopCount = 0;
      m_lastFlipFlopCheck = now;
   }
   m_context.isStable = (m_flipFlopCount < 5);
   
   return true;
}

//+------------------------------------------------------------------+
//| Calculate FSM Metrics                                            |
//+------------------------------------------------------------------+
void CFSM::CalculateMetrics(double currentPrice, double prevPrice, double volume) {
   double atr = m_emas.GetATR();
   if(atr <= 0) atr = 0.0001;
   
   // Effort: Volume normalized
   SFanMetrics fan = m_emas.GetFanMetrics();
   SEMAGroupMetrics slow = m_emas.GetSlowMetrics();
   
   // Simple volume normalization (would use rolling percentile in production)
   m_context.metrics.effort = volume / 10000.0;  // Placeholder normalization
   
   // Result: Price change normalized by ATR
   m_context.metrics.result = MathAbs(currentPrice - prevPrice) / atr;
   
   // Compression from EMA fan
   m_context.metrics.compression = fan.compression;
   
   // Slope from slow EMAs
   m_context.metrics.slope = slow.avgSlope;
   
   // Momentum (rate of change of result) - using FEAT if available
   if(m_feat != NULL) {
      SAccelMetrics accel = m_feat.GetAccel();
      m_context.metrics.momentum = accel.fanOpeningSpeed;
   } else {
      m_context.metrics.momentum = 0;
   }
   
   // Speed: Current price velocity
   m_context.metrics.speed = (currentPrice - prevPrice) / atr;
   
   m_context.metrics.isValid = true;
}

//+------------------------------------------------------------------+
//| Determine State Based on Metrics                                 |
//+------------------------------------------------------------------+
ENUM_MARKET_STATE CFSM::DetermineState() {
   SFSMMetrics m = m_context.metrics;
   SFEATResult feat = m_feat != NULL ? m_feat.GetResult() : SFEATResult();
   
   // Get percentile positions
   double effortPct = GetPercentile(m_effortBuffer, m.effort);
   double resultPct = GetPercentile(m_resultBuffer, m.result);
   
   // RESET: Quick mean reversion (high speed return to mean)
   if(MathAbs(m.speed) > m_thresholds.resetSpeed && 
      m.compression > 0.5 &&
      m.momentum < 0) {
      return STATE_RESET;
   }
   
   // ACCUMULATION: High effort, low result, compressed structure
   if(effortPct > m_thresholds.effortP80 && 
      resultPct < m_thresholds.resultP20 &&
      m.compression > m_thresholds.accumulationCompression) {
      return STATE_ACCUMULATION;
   }
   
   // EXPANSION: High result, moderate effort, active slope
   if(resultPct > m_thresholds.resultP80 &&
      effortPct < m_thresholds.effortP50 &&
      MathAbs(m.slope) > m_thresholds.expansionSlope) {
      
      // Verify with FEAT if available
      if(m_feat != NULL && feat.accel.type == ACCEL_VALID) {
         return STATE_EXPANSION;
      } else if(m_feat == NULL) {
         return STATE_EXPANSION;
      }
   }
   
   // DISTRIBUTION: Effort increasing, momentum decaying, away from value
   if(effortPct > m_thresholds.effortP50 &&
      m.momentum < m_thresholds.distributionMomentum &&
      m.compression < 0.5) {
      return STATE_DISTRIBUTION;
   }
   
   // Default to ACCUMULATION if no clear signal
   return STATE_ACCUMULATION;
}

//+------------------------------------------------------------------+
//| Calculate Confidence for Proposed State                          |
//+------------------------------------------------------------------+
double CFSM::CalculateConfidence(ENUM_MARKET_STATE proposedState) {
   SFSMMetrics m = m_context.metrics;
   double confidence = 50;  // Base
   
   double effortPct = GetPercentile(m_effortBuffer, m.effort);
   double resultPct = GetPercentile(m_resultBuffer, m.result);
   
   switch(proposedState) {
      case STATE_ACCUMULATION:
         // More confident if high effort, low result, high compression
         confidence = 50 + (effortPct - 0.5) * 30 + (0.5 - resultPct) * 30 + (m.compression - 0.5) * 40;
         break;
         
      case STATE_EXPANSION:
         // More confident if high result, proper slope, FEAT validation
         confidence = 50 + (resultPct - 0.5) * 40 + MathAbs(m.slope) * 30;
         if(m_feat != NULL && m_feat.GetAccel().type == ACCEL_VALID) confidence += 15;
         break;
         
      case STATE_DISTRIBUTION:
         // More confident if momentum decay visible
         confidence = 50 + MathAbs(m.momentum) * 40 + (effortPct - 0.5) * 20;
         break;
         
      case STATE_RESET:
         // More confident if high speed
         confidence = 50 + MathAbs(m.speed / m_thresholds.resetSpeed) * 40;
         break;
   }
   
   return MathMax(0, MathMin(100, confidence));
}

//+------------------------------------------------------------------+
//| Apply Hysteresis to Prevent Flip-Flop                            |
//+------------------------------------------------------------------+
void CFSM::ApplyHysteresis(ENUM_MARKET_STATE &proposedState) {
   // If confidence is in ambiguous zone, keep current state
   double proposedConf = CalculateConfidence(proposedState);
   double currentConf = CalculateConfidence(m_context.state);
   
   // Need significant improvement to change state
   if(proposedState != m_context.state) {
      if(proposedConf < currentConf + m_thresholds.hysteresisMargin * 100) {
         proposedState = m_context.state;  // Revert to current
      } else {
         m_flipFlopCount++;  // Track potential instability
      }
   }
}

//+------------------------------------------------------------------+
//| Record State Transition                                          |
//+------------------------------------------------------------------+
void CFSM::RecordTransition(ENUM_MARKET_STATE newState, string reason) {
   SStateTransition trans;
   trans.prevState = m_context.state;
   trans.newState = newState;
   trans.transitionTime = TimeCurrent();
   trans.confidence = m_context.confidence;
   trans.reason = reason;
   
   m_context.lastTransition = trans.transitionTime;
   m_context.lastTransitionInfo = trans;
   
   // Add to history
   int size = ArraySize(m_history);
   if(size >= m_historySize) {
      ArrayRemove(m_history, 0, 1);
      size--;
   }
   ArrayResize(m_history, size + 1);
   m_history[size] = trans;
}

//+------------------------------------------------------------------+
//| Update Dynamic Percentiles                                       |
//+------------------------------------------------------------------+
void CFSM::UpdatePercentiles() {
   if(!m_bufferFull) return;  // Need full buffer for accurate percentiles
   
   // Sort and find percentiles for effort
   double sortedEffort[];
   ArrayCopy(sortedEffort, m_effortBuffer);
   ArraySort(sortedEffort);
   
   int p20Idx = (int)(m_bufferSize * 0.2);
   int p50Idx = (int)(m_bufferSize * 0.5);
   int p80Idx = (int)(m_bufferSize * 0.8);
   
   m_thresholds.effortP20 = sortedEffort[p20Idx];
   m_thresholds.effortP50 = sortedEffort[p50Idx];
   m_thresholds.effortP80 = sortedEffort[p80Idx];
   
   // Sort and find percentiles for result
   double sortedResult[];
   ArrayCopy(sortedResult, m_resultBuffer);
   ArraySort(sortedResult);
   
   m_thresholds.resultP20 = sortedResult[p20Idx];
   m_thresholds.resultP50 = sortedResult[p50Idx];
   m_thresholds.resultP80 = sortedResult[p80Idx];
}

//+------------------------------------------------------------------+
//| Get Percentile Position of Value in Buffer                       |
//+------------------------------------------------------------------+
double CFSM::GetPercentile(const double &buffer[], double value) {
   int count = m_bufferFull ? m_bufferSize : m_bufferIndex;
   if(count == 0) return 0.5;
   
   int below = 0;
   for(int i = 0; i < count; i++) {
      if(buffer[i] < value) below++;
   }
   
   return (double)below / count;
}

//+------------------------------------------------------------------+
//| Get State as String                                              |
//+------------------------------------------------------------------+
string CFSM::GetStateString() const {
   switch(m_context.state) {
      case STATE_ACCUMULATION: return "ACCUMULATION";
      case STATE_EXPANSION:    return "EXPANSION";
      case STATE_DISTRIBUTION: return "DISTRIBUTION";
      case STATE_RESET:        return "RESET";
   }
   return "UNKNOWN";
}

//+------------------------------------------------------------------+
//| Get History Item                                                 |
//+------------------------------------------------------------------+
SStateTransition CFSM::GetHistoryItem(int index) const {
   SStateTransition empty;
   ZeroMemory(empty);
   if(index < 0 || index >= ArraySize(m_history)) return empty;
   return m_history[index];
}

#endif // CFSM_MQH
