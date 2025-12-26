//+------------------------------------------------------------------+
//|                                                 CLiquidity.mqh |
//|                    Liquidity Layer - Causal Reading             |
//|          External, Internal, Imbalance Detection                |
//+------------------------------------------------------------------+
#ifndef CLIQUIDITY_MQH
#define CLIQUIDITY_MQH

#include "CEMAs.mqh"

//+------------------------------------------------------------------+
//| LIQUIDITY TYPES                                                  |
//+------------------------------------------------------------------+
enum ENUM_LIQUIDITY_TYPE {
   LIQ_EXTERNAL,         // Obvious highs/lows, session extremes
   LIQ_INTERNAL,         // Within range, micro-ranges
   LIQ_IMBALANCE         // Fair value gaps, displacement zones
};

enum ENUM_LIQUIDITY_SIDE {
   LIQ_ABOVE,            // Buy-side liquidity (stops above)
   LIQ_BELOW             // Sell-side liquidity (stops below)
};

//+------------------------------------------------------------------+
//| LIQUIDITY LEVEL STRUCTURE                                        |
//+------------------------------------------------------------------+
struct SLiquidityLevel {
   double               price;
   ENUM_LIQUIDITY_TYPE  type;
   ENUM_LIQUIDITY_SIDE  side;
   datetime             createdTime;
   bool                 mitigated;
   datetime             mitigatedTime;
   double               strength;        // 0-1 based on touches/confluence
   string               label;           // Descriptive label
};

//+------------------------------------------------------------------+
//| IMBALANCE (FVG) STRUCTURE                                        |
//+------------------------------------------------------------------+
struct SImbalance {
   double               high;
   double               low;
   double               midpoint;
   datetime             time;
   bool                 mitigated;
   double               fillPercent;     // 0-1 how much has been filled
   bool                 isBullish;       // Direction of move that created it
};

//+------------------------------------------------------------------+
//| LIQUIDITY CONTEXT                                                |
//+------------------------------------------------------------------+
struct SLiquidityContext {
   SLiquidityLevel      nearestAbove;
   SLiquidityLevel      nearestBelow;
   int                  totalAbove;
   int                  totalBelow;
   double               imbalanceAbove;  // Nearest unfilled FVG above
   double               imbalanceBelow;  // Nearest unfilled FVG below
   bool                 isValid;
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
   double               m_equalThreshold;  // Pips for equal high/low detection
   double               m_fvgMinSize;      // Minimum FVG size in ATR
   int                  m_imbalanceCount;
   
   double               m_atr;
   double               m_tickSize;
   
   // Private methods
   void                 DetectExternalLiquidity(const double &high[], const double &low[], 
                                                const datetime &time[], int count);
   void                 DetectInternalLiquidity(const double &high[], const double &low[],
                                                const double &close[], const datetime &time[], int count);
   void                 DetectImbalances(const double &high[], const double &low[],
                                         const double &open[], const double &close[],
                                         const datetime &time[], int count);
   void                 UpdateMitigation(double currentHigh, double currentLow);
   void                 BuildContext(double currentPrice);
   bool                 IsEqualLevel(double price1, double price2);
   void                 AddLevel(double price, ENUM_LIQUIDITY_TYPE type, ENUM_LIQUIDITY_SIDE side,
                                 datetime time, double strength, string label);
   
public:
                        CLiquidity();
                       ~CLiquidity();
   
   // Configuration
   bool                 Init(string symbol, ENUM_TIMEFRAMES tf, int maxLevels = 50, 
                             int lookback = 100, double equalPips = 5.0, double fvgMinATR = 0.5);
   void                 SetATR(double atr) { m_atr = atr; }
   
   // Calculation
   bool                 Calculate(const double &high[], const double &low[],
                                  const double &open[], const double &close[],
                                  const datetime &time[], int count, double currentPrice);
   
   // Getters
   SLiquidityContext    GetContext() const { return m_context; }
   int                  GetLevelCount() const { return ArraySize(m_levels); }
   SLiquidityLevel      GetLevel(int index) const;
   int                  GetImbalanceCount() const { return ArraySize(m_imbalances); }
   SImbalance           GetImbalance(int index) const;
   
   // Level queries
   bool                 GetNearestLevel(double price, ENUM_LIQUIDITY_SIDE side, SLiquidityLevel &level);
   bool                 GetNearestImbalance(double price, ENUM_LIQUIDITY_SIDE side, SImbalance &imb);
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CLiquidity::CLiquidity() {
   m_maxLevels = 50;
   m_lookbackBars = 100;
   m_equalThreshold = 0.0005;
   m_fvgMinSize = 0.5;
   m_atr = 0.001;
   m_tickSize = 0.00001;
   ZeroMemory(m_context);
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CLiquidity::~CLiquidity() {
   ArrayFree(m_levels);
   ArrayFree(m_imbalances);
}

//+------------------------------------------------------------------+
//| Initialize                                                       |
//+------------------------------------------------------------------+
bool CLiquidity::Init(string symbol, ENUM_TIMEFRAMES tf, int maxLevels = 50,
                      int lookback = 100, double equalPips = 5.0, double fvgMinATR = 0.5) {
   m_symbol = symbol;
   m_timeframe = tf;
   m_maxLevels = maxLevels;
   m_lookbackBars = lookback;
   m_fvgMinSize = fvgMinATR;
   
   m_tickSize = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_SIZE);
   m_equalThreshold = equalPips * m_tickSize * 10;  // Convert pips to price
   
   ArrayResize(m_levels, 0);
   ArrayResize(m_imbalances, m_maxLevels);
   m_imbalanceCount = 0;
   
   return true;
}

//+------------------------------------------------------------------+
//| Main Calculation                                                 |
//+------------------------------------------------------------------+
bool CLiquidity::Calculate(const double &high[], const double &low[],
                           const double &open[], const double &close[],
                           const datetime &time[], int count, double currentPrice) {
   
   if(count < 10) return false;
   
   int barsToProcess = MathMin(count, m_lookbackBars);
   
   // Clear old data
   ArrayResize(m_levels, 0);
   m_imbalanceCount = 0;
   
   // Detect all liquidity types
   DetectExternalLiquidity(high, low, time, barsToProcess);
   DetectInternalLiquidity(high, low, close, time, barsToProcess);
   DetectImbalances(high, low, open, close, time, barsToProcess);
   
   // Update mitigation status
   UpdateMitigation(high[0], low[0]);
   
   // Build context
   BuildContext(currentPrice);
   
   return true;
}

//+------------------------------------------------------------------+
//| Detect External Liquidity                                        |
//| Swing highs/lows, session extremes, equal levels                 |
//+------------------------------------------------------------------+
void CLiquidity::DetectExternalLiquidity(const double &high[], const double &low[],
                                          const datetime &time[], int count) {
   
   // Detect swing highs (liquidity above - buy stops)
   for(int i = 2; i < count - 2; i++) {
      // Swing high: higher than 2 bars before and after
      if(high[i] > high[i-1] && high[i] > high[i-2] &&
         high[i] > high[i+1] && high[i] > high[i+2]) {
         
         // Check for equal highs nearby
         double strength = 1.0;
         for(int j = i + 3; j < MathMin(i + 20, count); j++) {
            if(IsEqualLevel(high[i], high[j])) {
               strength += 0.5;  // Increase strength for equal levels
            }
         }
         strength = MathMin(3.0, strength) / 3.0;  // Normalize to 0-1
         
         AddLevel(high[i], LIQ_EXTERNAL, LIQ_ABOVE, time[i], strength, "SwingHigh");
      }
      
      // Swing low: lower than 2 bars before and after
      if(low[i] < low[i-1] && low[i] < low[i-2] &&
         low[i] < low[i+1] && low[i] < low[i+2]) {
         
         double strength = 1.0;
         for(int j = i + 3; j < MathMin(i + 20, count); j++) {
            if(IsEqualLevel(low[i], low[j])) {
               strength += 0.5;
            }
         }
         strength = MathMin(3.0, strength) / 3.0;
         
         AddLevel(low[i], LIQ_EXTERNAL, LIQ_BELOW, time[i], strength, "SwingLow");
      }
   }
   
   // Session extremes (simplified - based on bar index patterns)
   // In real implementation, would parse actual session times
   double sessionHigh = high[0], sessionLow = low[0];
   for(int i = 0; i < MathMin(24, count); i++) {
      if(high[i] > sessionHigh) sessionHigh = high[i];
      if(low[i] < sessionLow) sessionLow = low[i];
   }
   
   AddLevel(sessionHigh, LIQ_EXTERNAL, LIQ_ABOVE, time[0], 0.8, "SessionHigh");
   AddLevel(sessionLow, LIQ_EXTERNAL, LIQ_BELOW, time[0], 0.8, "SessionLow");
}

//+------------------------------------------------------------------+
//| Detect Internal Liquidity                                        |
//| Range-bound levels, micro-structures                             |
//+------------------------------------------------------------------+
void CLiquidity::DetectInternalLiquidity(const double &high[], const double &low[],
                                          const double &close[], const datetime &time[], int count) {
   
   // Find range boundaries (consolidation areas)
   for(int i = 5; i < count - 5; i++) {
      // Detect micro range highs (3-bar pattern)
      if(high[i] >= high[i-1] && high[i] >= high[i+1] &&
         high[i] < high[i-3] && high[i] < high[i+3]) {
         
         AddLevel(high[i], LIQ_INTERNAL, LIQ_ABOVE, time[i], 0.4, "MicroHigh");
      }
      
      // Detect micro range lows
      if(low[i] <= low[i-1] && low[i] <= low[i+1] &&
         low[i] > low[i-3] && low[i] > low[i+3]) {
         
         AddLevel(low[i], LIQ_INTERNAL, LIQ_BELOW, time[i], 0.4, "MicroLow");
      }
   }
}

//+------------------------------------------------------------------+
//| Detect Imbalances (Fair Value Gaps)                              |
//+------------------------------------------------------------------+
void CLiquidity::DetectImbalances(const double &high[], const double &low[],
                                   const double &open[], const double &close[],
                                   const datetime &time[], int count) {
   


      // Bullish FVG
      if(low[i-1] > high[i+1]) {
         double gapSize = low[i-1] - high[i+1];
         if(gapSize > m_atr * m_fvgMinSize) {
            SImbalance imb;
            imb.high = low[i-1];
            imb.low = high[i+1];
            imb.midpoint = (imb.high + imb.low) / 2;
            imb.time = time[i];
            imb.mitigated = false;
            imb.fillPercent = 0;
            imb.isBullish = true;
            
            // Add or overwrite (ring buffer style implies removing oldest start, but here we just append)
            // Since we cleared at start of calculate using m_imbalanceCount = 0
            if(m_imbalanceCount < m_maxLevels) {
               m_imbalances[m_imbalanceCount] = imb;
               m_imbalanceCount++;
            } else {
               // If full, we should technically remove the oldest.
               // For now, let's just shift or stop adding. 
               // Shift is O(N), expensive.
               // Better strategy: We are scanning history. Usually we want most recent.
               // We should scan BACKWARDS if we want most recent first and stop at max.
               // But detecting patterns iterates forward. 
               // Let's implement simple ring overwrite or shift.
               // Optimization: Shift is acceptable if maxLevels is small (50).
               ArrayCopy(m_imbalances, m_imbalances, 0, 1, m_maxLevels - 1);
               m_imbalances[m_maxLevels - 1] = imb;
            }
         }
      }
      
      // Bearish FVG
      if(high[i-1] < low[i+1]) {
         double gapSize = low[i+1] - high[i-1];
         if(gapSize > m_atr * m_fvgMinSize) {
            SImbalance imb;
            imb.high = low[i+1];
            imb.low = high[i-1];
            imb.midpoint = (imb.high + imb.low) / 2;
            imb.time = time[i];
            imb.mitigated = false;
            imb.fillPercent = 0;
            imb.isBullish = false;
            
            if(m_imbalanceCount < m_maxLevels) {
               m_imbalances[m_imbalanceCount] = imb;
               m_imbalanceCount++;
            } else {
               ArrayCopy(m_imbalances, m_imbalances, 0, 1, m_maxLevels - 1);
               m_imbalances[m_maxLevels - 1] = imb;
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Update Mitigation Status                                         |
//+------------------------------------------------------------------+
void CLiquidity::UpdateMitigation(double currentHigh, double currentLow) {
   // Update levels
   for(int i = 0; i < ArraySize(m_levels); i++) {
      if(m_levels[i].mitigated) continue;
      
      if(m_levels[i].side == LIQ_ABOVE && currentHigh >= m_levels[i].price) {
         m_levels[i].mitigated = true;
         m_levels[i].mitigatedTime = TimeCurrent();
      }
      if(m_levels[i].side == LIQ_BELOW && currentLow <= m_levels[i].price) {
         m_levels[i].mitigated = true;
         m_levels[i].mitigatedTime = TimeCurrent();
      }
   }
   
   // Update imbalances
   for(int i = 0; i < ArraySize(m_imbalances); i++) {
      if(m_imbalances[i].mitigated) continue;
      
      // Calculate fill percentage
      if(m_imbalances[i].isBullish) {
         // Price needs to come down into the gap
         if(currentLow <= m_imbalances[i].high) {
            double fill = (m_imbalances[i].high - MathMax(currentLow, m_imbalances[i].low)) /
                          (m_imbalances[i].high - m_imbalances[i].low);
            m_imbalances[i].fillPercent = MathMax(m_imbalances[i].fillPercent, fill);
            if(m_imbalances[i].fillPercent >= 1.0) m_imbalances[i].mitigated = true;
         }
      } else {
         // Price needs to come up into the gap
         if(currentHigh >= m_imbalances[i].low) {
            double fill = (MathMin(currentHigh, m_imbalances[i].high) - m_imbalances[i].low) /
                          (m_imbalances[i].high - m_imbalances[i].low);
            m_imbalances[i].fillPercent = MathMax(m_imbalances[i].fillPercent, fill);
            if(m_imbalances[i].fillPercent >= 1.0) m_imbalances[i].mitigated = true;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Build Liquidity Context                                          |
//+------------------------------------------------------------------+
void CLiquidity::BuildContext(double currentPrice) {
   m_context.totalAbove = 0;
   m_context.totalBelow = 0;
   m_context.nearestAbove.price = 0;
   m_context.nearestBelow.price = 0;
   m_context.imbalanceAbove = 0;
   m_context.imbalanceBelow = 0;
   
   double nearestAboveDist = DBL_MAX;
   double nearestBelowDist = DBL_MAX;
   
   // Find nearest levels
   for(int i = 0; i < ArraySize(m_levels); i++) {
      if(m_levels[i].mitigated) continue;
      
      if(m_levels[i].side == LIQ_ABOVE && m_levels[i].price > currentPrice) {
         m_context.totalAbove++;
         double dist = m_levels[i].price - currentPrice;
         if(dist < nearestAboveDist) {
            nearestAboveDist = dist;
            m_context.nearestAbove = m_levels[i];
         }
      }
      if(m_levels[i].side == LIQ_BELOW && m_levels[i].price < currentPrice) {
         m_context.totalBelow++;
         double dist = currentPrice - m_levels[i].price;
         if(dist < nearestBelowDist) {
            nearestBelowDist = dist;
            m_context.nearestBelow = m_levels[i];
         }
      }
   }
   
   // Find nearest imbalances
   nearestAboveDist = DBL_MAX;
   nearestBelowDist = DBL_MAX;
   
   for(int i = 0; i < ArraySize(m_imbalances); i++) {
      if(m_imbalances[i].mitigated) continue;
      
      if(m_imbalances[i].midpoint > currentPrice) {
         double dist = m_imbalances[i].midpoint - currentPrice;
         if(dist < nearestAboveDist) {
            nearestAboveDist = dist;
            m_context.imbalanceAbove = m_imbalances[i].midpoint;
         }
      } else {
         double dist = currentPrice - m_imbalances[i].midpoint;
         if(dist < nearestBelowDist) {
            nearestBelowDist = dist;
            m_context.imbalanceBelow = m_imbalances[i].midpoint;
         }
      }
   }
   
   m_context.isValid = true;
}

//+------------------------------------------------------------------+
//| Check if two prices are equal (within threshold)                 |
//+------------------------------------------------------------------+
bool CLiquidity::IsEqualLevel(double price1, double price2) {
   return MathAbs(price1 - price2) <= m_equalThreshold;
}

//+------------------------------------------------------------------+
//| Add Liquidity Level                                              |
//+------------------------------------------------------------------+
void CLiquidity::AddLevel(double price, ENUM_LIQUIDITY_TYPE type, ENUM_LIQUIDITY_SIDE side,
                           datetime time, double strength, string label) {
   
   // Check for duplicate
   for(int i = 0; i < ArraySize(m_levels); i++) {
      if(IsEqualLevel(m_levels[i].price, price) && m_levels[i].side == side) {
         // Merge: increase strength
         m_levels[i].strength = MathMin(1.0, m_levels[i].strength + strength * 0.5);
         return;
      }
   }
   
   // Add new level
   if(ArraySize(m_levels) >= m_maxLevels) {
      // Remove oldest/weakest level
      int weakestIdx = 0;
      double weakestStr = m_levels[0].strength;
      for(int i = 1; i < ArraySize(m_levels); i++) {
         if(m_levels[i].strength < weakestStr) {
            weakestStr = m_levels[i].strength;
            weakestIdx = i;
         }
      }
      ArrayRemove(m_levels, weakestIdx, 1);
   }
   
   SLiquidityLevel lvl;
   lvl.price = price;
   lvl.type = type;
   lvl.side = side;
   lvl.createdTime = time;
   lvl.mitigated = false;
   lvl.mitigatedTime = 0;
   lvl.strength = strength;
   lvl.label = label;
   
   int size = ArraySize(m_levels);
   ArrayResize(m_levels, size + 1);
   m_levels[size] = lvl;
}

//+------------------------------------------------------------------+
//| Get Level by Index                                               |
//+------------------------------------------------------------------+
SLiquidityLevel CLiquidity::GetLevel(int index) const {
   SLiquidityLevel empty;
   ZeroMemory(empty);
   if(index < 0 || index >= ArraySize(m_levels)) return empty;
   return m_levels[index];
}

//+------------------------------------------------------------------+
//| Get Imbalance by Index                                           |
//+------------------------------------------------------------------+
SImbalance CLiquidity::GetImbalance(int index) const {
   SImbalance empty;
   ZeroMemory(empty);
   if(index < 0 || index >= ArraySize(m_imbalances)) return empty;
   return m_imbalances[index];
}

//+------------------------------------------------------------------+
//| Get Nearest Level on Side                                        |
//+------------------------------------------------------------------+
bool CLiquidity::GetNearestLevel(double price, ENUM_LIQUIDITY_SIDE side, SLiquidityLevel &level) {
   double nearestDist = DBL_MAX;
   bool found = false;
   
   for(int i = 0; i < ArraySize(m_levels); i++) {
      if(m_levels[i].mitigated || m_levels[i].side != side) continue;
      
      double dist = MathAbs(m_levels[i].price - price);
      if(dist < nearestDist) {
         nearestDist = dist;
         level = m_levels[i];
         found = true;
      }
   }
   
   return found;
}

//+------------------------------------------------------------------+
//| Get Nearest Imbalance on Side                                    |
//+------------------------------------------------------------------+
bool CLiquidity::GetNearestImbalance(double price, ENUM_LIQUIDITY_SIDE side, SImbalance &imb) {
   double nearestDist = DBL_MAX;
   bool found = false;
   
   for(int i = 0; i < ArraySize(m_imbalances); i++) {
      if(m_imbalances[i].mitigated) continue;
      
      bool isAbove = (m_imbalances[i].midpoint > price);
      if((side == LIQ_ABOVE && !isAbove) || (side == LIQ_BELOW && isAbove)) continue;
      
      double dist = MathAbs(m_imbalances[i].midpoint - price);
      if(dist < nearestDist) {
         nearestDist = dist;
         imb = m_imbalances[i];
         found = true;
      }
   }
   
   return found;
}

#endif // CLIQUIDITY_MQH
