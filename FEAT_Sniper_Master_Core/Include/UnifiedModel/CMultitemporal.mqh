//+------------------------------------------------------------------+
//|                                              CMultitemporal.mqh |
//|                    Multi-Timeframe Analysis Module               |
//|           Validates confluences across M5/H1/H4/D1/W1            |
//+------------------------------------------------------------------+
#ifndef CMULTITEMPORAL_MQH
#define CMULTITEMPORAL_MQH

#include "CEMAs.mqh"
#include "CLiquidity.mqh"

//+------------------------------------------------------------------+
//| TIMEFRAME STATE STRUCTURE                                         |
//+------------------------------------------------------------------+
struct STimeframeState {
   ENUM_TIMEFRAMES tf;
   bool            bullish;           // Bullish structure
   bool            bearish;           // Bearish structure
   bool            expanding;         // EMA layers expanding
   bool            compressed;        // EMA layers compressed
   double          bias;              // -1 to 1 (bearish to bullish)
   double          score;             // Confluence score 0-100
   bool            hasFreshZone;      // Unmitigated zone in range
   bool            hasRecentBOS;      // BOS in last 10 bars
   bool            hasRecentCHoCH;    // CHoCH in last 10 bars
   datetime        lastUpdate;
};

//+------------------------------------------------------------------+
//| MULTITEMPORAL RESULT                                              |
//+------------------------------------------------------------------+
struct SMultitemporalResult {
   STimeframeState states[5];         // M5, H1, H4, D1, W1
   int             confluenceScore;   // 0-100 based on alignment
   bool            isAgainstBias;     // Current intent vs D1/W1
   string          dominantTrend;     // "BULLISH", "BEARISH", "NEUTRAL"
   int             alignedCount;      // How many TFs aligned
   bool            isValid;
};

//+------------------------------------------------------------------+
//| CMULTITEMPORAL CLASS                                              |
//+------------------------------------------------------------------+
class CMultitemporal {
private:
   string              m_symbol;
   SMultitemporalResult m_result;
   bool                m_initialized;
   
   ENUM_TIMEFRAMES     m_timeframes[5];
   
   bool AnalyzeTimeframe(ENUM_TIMEFRAMES tf, STimeframeState &state);
   void CalculateConfluence();

public:
   CMultitemporal();
   ~CMultitemporal() {}
   
   bool Init(string symbol);
   bool Calculate();
   
   // Getters
   SMultitemporalResult GetResult() const { return m_result; }
   int GetConfluenceScore() const { return m_result.confluenceScore; }
   bool IsAgainstBias() const { return m_result.isAgainstBias; }
   string GetDominantTrend() const { return m_result.dominantTrend; }
   STimeframeState GetState(int index) const { 
      if(index >= 0 && index < 5) return m_result.states[index]; 
      STimeframeState e; ZeroMemory(e); return e;
   }
   
   // Feature export for ML
   void GetFeatureVector(double &features[]);
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CMultitemporal::CMultitemporal() : m_initialized(false) {
   ZeroMemory(m_result);
   m_timeframes[0] = PERIOD_M5;
   m_timeframes[1] = PERIOD_H1;
   m_timeframes[2] = PERIOD_H4;
   m_timeframes[3] = PERIOD_D1;
   m_timeframes[4] = PERIOD_W1;
}

//+------------------------------------------------------------------+
//| Initialize                                                        |
//+------------------------------------------------------------------+
bool CMultitemporal::Init(string symbol) {
   m_symbol = symbol;
   m_initialized = true;
   return true;
}

//+------------------------------------------------------------------+
//| Main Calculation                                                  |
//+------------------------------------------------------------------+
bool CMultitemporal::Calculate() {
   if(!m_initialized) return false;
   
   // Analyze each timeframe
   for(int i = 0; i < 5; i++) {
      m_result.states[i].tf = m_timeframes[i];
      if(!AnalyzeTimeframe(m_timeframes[i], m_result.states[i])) {
         m_result.states[i].score = 0;
      }
   }
   
   // Calculate overall confluence
   CalculateConfluence();
   
   m_result.isValid = true;
   return true;
}

//+------------------------------------------------------------------+
//| Analyze Single Timeframe                                          |
//+------------------------------------------------------------------+
bool CMultitemporal::AnalyzeTimeframe(ENUM_TIMEFRAMES tf, STimeframeState &state) {
   state.lastUpdate = TimeCurrent();
   
   // Get OHLC data
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(m_symbol, tf, 0, 50, rates);
   if(copied < 20) return false;
   
   // Analyze structure via swing points
   double lastSwingHigh = 0, prevSwingHigh = 0;
   double lastSwingLow = DBL_MAX, prevSwingLow = DBL_MAX;
   
   for(int i = 5; i < copied - 5; i++) {
      bool isSH = (rates[i].high > rates[i-1].high && rates[i].high > rates[i-2].high &&
                   rates[i].high > rates[i+1].high && rates[i].high > rates[i+2].high);
      bool isSL = (rates[i].low < rates[i-1].low && rates[i].low < rates[i-2].low &&
                   rates[i].low < rates[i+1].low && rates[i].low < rates[i+2].low);
      
      if(isSH) {
         prevSwingHigh = lastSwingHigh;
         lastSwingHigh = rates[i].high;
      }
      if(isSL) {
         prevSwingLow = lastSwingLow;
         lastSwingLow = rates[i].low;
      }
   }
   
   // Determine structure
   state.bullish = (prevSwingHigh > 0 && lastSwingHigh > prevSwingHigh) && 
                   (prevSwingLow < DBL_MAX && lastSwingLow > prevSwingLow);
   state.bearish = (prevSwingHigh > 0 && lastSwingHigh < prevSwingHigh) && 
                   (prevSwingLow < DBL_MAX && lastSwingLow < prevSwingLow);
   
   // Check recent BOS/CHoCH
   double prevHigh = rates[0].high, prevLow = rates[0].low;
   state.hasRecentBOS = false;
   state.hasRecentCHoCH = false;
   
   for(int i = 1; i <= 10 && i < copied; i++) {
      if(rates[i].close > prevHigh && state.bullish) state.hasRecentBOS = true;
      if(rates[i].close < prevLow && state.bearish) state.hasRecentBOS = true;
      if(rates[i].close > prevHigh && !state.bullish) state.hasRecentCHoCH = true;
      if(rates[i].close < prevLow && !state.bearish) state.hasRecentCHoCH = true;
   }
   
   // Calculate EMA compression (using SMA as proxy)
   double fastMA = 0, slowMA = 0;
   for(int i = 0; i < 8; i++) fastMA += rates[i].close;
   fastMA /= 8;
   for(int i = 0; i < 21; i++) slowMA += rates[i].close;
   slowMA /= 21;
   
   double atr = 0;
   for(int i = 1; i < 15; i++) atr += rates[i].high - rates[i].low;
   atr /= 14;
   
   double separation = MathAbs(fastMA - slowMA) / (atr > 0 ? atr : 0.0001);
   state.expanding = (separation > 2.0);
   state.compressed = (separation < 0.5);
   
   // Calculate bias (-1 to 1)
   if(state.bullish) state.bias = 0.5 + (state.expanding ? 0.3 : 0) + (state.hasRecentBOS ? 0.2 : 0);
   else if(state.bearish) state.bias = -0.5 - (state.expanding ? 0.3 : 0) - (state.hasRecentBOS ? 0.2 : 0);
   else state.bias = (fastMA > slowMA) ? 0.2 : -0.2;
   
   // Calculate TF score
   state.score = 50.0;
   if(state.bullish || state.bearish) state.score += 20.0;
   if(state.hasRecentBOS) state.score += 15.0;
   if(state.hasRecentCHoCH) state.score += 10.0;
   if(state.expanding) state.score += 5.0;
   
   state.score = MathMin(100.0, MathMax(0.0, state.score));
   
   return true;
}

//+------------------------------------------------------------------+
//| Calculate Overall Confluence                                      |
//+------------------------------------------------------------------+
void CMultitemporal::CalculateConfluence() {
   int bullishCount = 0, bearishCount = 0;
   double totalScore = 0;
   
   // Weight by timeframe importance: M5=1, H1=2, H4=3, D1=4, W1=5
   double weights[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
   double weightSum = 0;
   
   for(int i = 0; i < 5; i++) {
      if(m_result.states[i].bullish) bullishCount++;
      if(m_result.states[i].bearish) bearishCount++;
      
      totalScore += m_result.states[i].score * weights[i];
      weightSum += weights[i];
   }
   
   // Confluence score
   m_result.confluenceScore = (int)(totalScore / weightSum);
   
   // Alignment count
   m_result.alignedCount = MathMax(bullishCount, bearishCount);
   
   // Dominant trend based on higher TFs (H4, D1, W1)
   int htfBullish = 0, htfBearish = 0;
   for(int i = 2; i < 5; i++) {  // H4, D1, W1
      if(m_result.states[i].bullish) htfBullish++;
      if(m_result.states[i].bearish) htfBearish++;
   }
   
   if(htfBullish >= 2) m_result.dominantTrend = "BULLISH";
   else if(htfBearish >= 2) m_result.dominantTrend = "BEARISH";
   else m_result.dominantTrend = "NEUTRAL";
   
   // Check if lower TF is against dominant bias
   bool m5Bullish = m_result.states[0].bullish;
   bool m5Bearish = m_result.states[0].bearish;
   
   m_result.isAgainstBias = (m_result.dominantTrend == "BULLISH" && m5Bearish) ||
                             (m_result.dominantTrend == "BEARISH" && m5Bullish);
   
   // Bonus for alignment
   if(m_result.alignedCount >= 4) {
      m_result.confluenceScore = MathMin(100, m_result.confluenceScore + 20);
   } else if(m_result.alignedCount >= 3) {
      m_result.confluenceScore = MathMin(100, m_result.confluenceScore + 10);
   }
   
   // Penalty for going against bias
   if(m_result.isAgainstBias) {
      m_result.confluenceScore = (int)(m_result.confluenceScore * 0.6);
   }
}

//+------------------------------------------------------------------+
//| Export Feature Vector for ML                                      |
//+------------------------------------------------------------------+
void CMultitemporal::GetFeatureVector(double &features[]) {
   ArrayResize(features, 15);
   
   // Per-timeframe features (5 TFs x 3 features = 15)
   for(int i = 0; i < 5; i++) {
      features[i*3 + 0] = m_result.states[i].bias;
      features[i*3 + 1] = m_result.states[i].score / 100.0;
      features[i*3 + 2] = m_result.states[i].bullish ? 1.0 : (m_result.states[i].bearish ? -1.0 : 0.0);
   }
}

#endif // CMULTITEMPORAL_MQH
