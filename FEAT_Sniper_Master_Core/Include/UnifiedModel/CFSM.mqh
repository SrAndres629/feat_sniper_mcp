//+------------------------------------------------------------------+
//|                                                       CFSM.mqh |
//|                    FSM Layer - State Classification             |
//|        Multifractal Layer Logic: Micro, Oper, Macro, Bias       |
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
   
   double atr = m_ptrEmas.GetATR();
   double effort = volume / (atr * 100000); // Normalized effort
   double result = MathAbs(close - prevClose) / atr;
   
   m_effortBuffer[m_index] = effort;
   m_resultBuffer[m_index] = result;
   m_index = (m_index + 1) % m_bufferSize;
   if(m_index == 0) m_full = true;
   
   if(!m_full) { m_state = STATE_CALIBRATING; return true; }
   
   // --- Professional Layer Question Logic ---
   SEMAGroupMetrics micro = m_ptrEmas.GetMicroMetrics();
   SEMAGroupMetrics oper  = m_ptrEmas.GetOperationalMetrics();
   SEMAGroupMetrics macro = m_ptrEmas.GetMacroMetrics();
   SEMAGroupMetrics bias  = m_ptrEmas.GetBiasMetrics();
   
   // 1. Is Micro compressed?
   bool microCompressed = micro.compression > 0.7;
   
   // 2. Is Oper rejecting or absorbing?
   bool priceInOper = m_ptrEmas.GetPricePosition(close) < 0.5 && m_ptrEmas.GetPricePosition(close) > -0.5;
   
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
      SEMAGroupMetrics micro = m_ptrEmas.GetMicroMetrics();
      SEMAGroupMetrics oper  = m_ptrEmas.GetOperationalMetrics();
      m.effort = m_effortBuffer[(m_index + m_bufferSize - 1) % m_bufferSize];
      m.result = m_resultBuffer[(m_index + m_bufferSize - 1) % m_bufferSize];
      m.compression = micro.compression;
      m.slope = oper.avgSlope;
      m.speed = (micro.avgValue - oper.avgValue) / m_ptrEmas.GetATR();
   }
   return m;
}

#endif
