//+------------------------------------------------------------------+
//|                                                 CLiquidity.mqh |
//|                    Liquidity Layer - Causal Reading             |
//|          Institutional Zones, FVG, OB, ZS, PC Detection         |
//+------------------------------------------------------------------+
#ifndef CLIQUIDITY_MQH
#define CLIQUIDITY_MQH

#include "CEMAs.mqh"

//+------------------------------------------------------------------+
//| LIQUIDITY TYPES                                                  |
//+------------------------------------------------------------------+
enum ENUM_LIQUIDITY_TYPE { LIQ_EXTERNAL, LIQ_INTERNAL, LIQ_IMBALANCE, LIQ_POOL };
enum ENUM_LIQUIDITY_SIDE { LIQ_ABOVE, LIQ_BELOW };

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
   double               equilibrium;    
   bool                 isPremium;      
   bool                 isDiscount;     
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
//| INSTITUTIONAL ZONES (FVG, OB, ZS, PC)                            |
//+------------------------------------------------------------------+
enum ENUM_ZONE_TYPE { ZONE_FVG, ZONE_OB, ZONE_ZS, ZONE_PC, ZONE_CONFLUENCE, ZONE_NONE };

struct SInstitutionalZone {
   ENUM_ZONE_TYPE type;
   double         top;
   double         bottom;
   double         mid;
   datetime       time;
   bool           isBullish;
   bool           mitigated;
   double         strength;
   string         label;
};

//+------------------------------------------------------------------+
//| CLIQUIDITY CLASS                                                 |
//+------------------------------------------------------------------+
class CLiquidity {
private:
   SLiquidityLevel      m_levels[];
   SImbalance           m_imbalances[];
   SInstitutionalZone   m_zones[];
   SLiquidityContext    m_context;
   
   string               m_symbol;
   ENUM_TIMEFRAMES      m_timeframe;
   
   int                  m_maxLevels;
   int                  m_lookbackBars;
   double               m_equalThreshold;  
   double               m_fvgMinSize;      
   int                  m_imbalanceCount;
   int                  m_zoneCount;
   
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
   void                 DetectInstitutionalZones(const double &open[], const double &high[], const double &low[], const double &close[], const datetime &time[], int count);
   void                 DetectConfluence();
   void                 UpdateMitigation(double currentHigh, double currentLow);
   void                 BuildContext(double currentPrice);
   bool                 IsEqualLevel(double price1, double price2);
   void                 AddLevel(double price, ENUM_LIQUIDITY_TYPE type, ENUM_LIQUIDITY_SIDE side, datetime time, double strength, string label);
   void                 AddZone(ENUM_ZONE_TYPE type, double t, double b, datetime tm, bool bull, string lbl);
   
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
   int                  GetZoneCount() const { return m_zoneCount; }
   SInstitutionalZone   GetZone(int index) const { if(index>=0 && index<m_zoneCount) return m_zones[index]; SInstitutionalZone e; ZeroMemory(e); return e; }
   
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
   ArrayResize(m_zones, m_maxLevels * 4); m_zoneCount = 0;
   ArrayResize(m_structures, m_maxLevels); m_structCount = 0;
   ArrayResize(m_swings, m_maxLevels); m_swingCount = 0;
   ArrayResize(m_patterns, m_maxLevels); m_patternCount = 0;
   return true;
}

bool CLiquidity::Calculate(const double &high[], const double &low[], const double &open[], const double &close[], const datetime &time[], int count, double currentPrice) {
   if(count < 10) return false;
   int barsToProcess = MathMin(count, m_lookbackBars);
   ArrayResize(m_levels, 0); m_imbalanceCount = 0; m_zoneCount = 0; m_structCount = 0; m_swingCount = 0; m_patternCount = 0;
   DetectHighsLows(high, low, time, barsToProcess);
   DetectStructure(high, low, close, time, barsToProcess);
   DetectPatterns(close, time, barsToProcess);
   DetectExternalLiquidity(high, low, time, barsToProcess);
   DetectInternalLiquidity(high, low, close, time, barsToProcess);
   DetectImbalances(high, low, open, close, time, barsToProcess);
   DetectInstitutionalZones(open, high, low, close, time, barsToProcess);
   DetectConfluence();
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
   // GC Pattern from CHoCH
   for(int i = 0; i < m_structCount; i++) {
      if(m_structures[i].type == STRUCT_CHOCH) {
         SPatternEvent p; p.type = PAT_GC; p.time = m_structures[i].time; p.price = m_structures[i].price; p.isBullish = m_structures[i].isBullish; p.description = "GC";
         if(m_patternCount < m_maxLevels) m_patterns[m_patternCount++] = p;
      }
   }
   
   // M/W Pattern Detection from Swing Sequence
   if(m_swingCount >= 4) {
      // Look at most recent 4 swings
      SSwingPoint s0 = m_swings[0], s1 = m_swings[1], s2 = m_swings[2], s3 = m_swings[3];
      
      // M-Pattern (Bearish Double Top): LH after HH sequence
      // s3=HL, s2=HH, s1=LH(lower than HH), s0=LL or break
      if(s2.type == SWING_HH && s1.type == SWING_LH && s1.price < s2.price * 0.995) {
         SPatternEvent p; p.type = PAT_M; p.time = s1.time; p.price = s2.price; p.isBullish = false;
         p.description = "M-Pattern (Double Top)";
         bool exists = false;
         for(int i=0; i<m_patternCount; i++) if(m_patterns[i].type == PAT_M && m_patterns[i].time == p.time) exists = true;
         if(!exists && m_patternCount < m_maxLevels) m_patterns[m_patternCount++] = p;
      }
      
      // W-Pattern (Bullish Double Bottom): HL after LL sequence
      // s3=LH, s2=LL, s1=HL(higher than LL), s0=HH or break
      if(s2.type == SWING_LL && s1.type == SWING_HL && s1.price > s2.price * 1.005) {
         SPatternEvent p; p.type = PAT_W; p.time = s1.time; p.price = s2.price; p.isBullish = true;
         p.description = "W-Pattern (Double Bottom)";
         bool exists = false;
         for(int i=0; i<m_patternCount; i++) if(m_patterns[i].type == PAT_W && m_patterns[i].time == p.time) exists = true;
         if(!exists && m_patternCount < m_maxLevels) m_patterns[m_patternCount++] = p;
      }
   }
}

void CLiquidity::DetectExternalLiquidity(const double &high[], const double &low[], const datetime &time[], int count) {
   for(int i = 2; i < count - 2; i++) {
      if(high[i] > high[i-1] && high[i] > high[i-2] && high[i] > high[i+1] && high[i] > high[i+2]) {
         AddLevel(high[i], LIQ_EXTERNAL, LIQ_ABOVE, time[i], 0.8, "SwingHigh");
         // Liquidity Pool (Equal Highs)
         for(int j=i-1; j>MathMax(0,i-20); j--) {
            if(MathAbs(high[i]-high[j]) < m_equalThreshold) { AddLevel(high[i], LIQ_POOL, LIQ_ABOVE, time[i], 1.2, "LiquidityPool(EQH)"); break; }
         }
      }
      if(low[i] < low[i-1] && low[i] < low[i-2] && low[i] < low[i+1] && low[i] < low[i+2]) {
         AddLevel(low[i], LIQ_EXTERNAL, LIQ_BELOW, time[i], 0.8, "SwingLow");
         // Liquidity Pool (Equal Lows)
         for(int j=i-1; j>MathMax(0,i-20); j--) {
            if(MathAbs(low[i]-low[j]) < m_equalThreshold) { AddLevel(low[i], LIQ_POOL, LIQ_BELOW, time[i], 1.2, "LiquidityPool(EQL)"); break; }
         }
      }
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

void CLiquidity::DetectInstitutionalZones(const double &open[], const double &high[], const double &low[], const double &close[], const datetime &time[], int count) {
   for(int i = 2; i < count - 2; i++) {
      double body = MathAbs(close[i] - open[i]), r = high[i] - low[i];
      double uW = high[i] - MathMax(open[i], close[i]), lW = MathMin(open[i], close[i]) - low[i];
      if(r > m_atr * 0.5) {
         if(uW > body * 2 && uW > m_atr * 0.3) AddZone(ZONE_ZS, high[i], MathMax(open[i], close[i]), time[i], false, "ShadowZone(H)");
         if(lW > body * 2 && lW > m_atr * 0.3) AddZone(ZONE_ZS, MathMin(open[i], close[i]), low[i], time[i], true, "ShadowZone(L)");
      }
      if(body < r * 0.2 && r > m_atr * 0.2) AddZone(ZONE_PC, high[i], low[i], time[i], (close[i] > open[i]), "CriticalPoint");
      if(i > 0) {
         double moveNext = MathAbs(close[i-1] - open[i-1]);
         if(moveNext > r * 2.5 && moveNext > m_atr * 0.8) {
            bool isBullishOB = (close[i-1] > open[i-1] && close[i] < open[i]);
            bool isBearishOB = (close[i-1] < open[i-1] && close[i] > open[i]);
            if(isBullishOB) AddZone(ZONE_OB, high[i], low[i], time[i], true, "BullishOB");
            if(isBearishOB) AddZone(ZONE_OB, high[i], low[i], time[i], false, "BearishOB");
         }
      }
   }
   for(int i = 0; i < m_imbalanceCount; i++) AddZone(ZONE_FVG, m_imbalances[i].high, m_imbalances[i].low, m_imbalances[i].time, m_imbalances[i].isBullish, "FVG");
}

void CLiquidity::DetectConfluence() {
   int count = m_zoneCount;
   for(int i=0; i<count; i++) {
      for(int j=i+1; j<count; j++) {
         if(m_zones[i].mitigated || m_zones[j].mitigated) continue;
         if(m_zones[i].isBullish != m_zones[j].isBullish) continue;
         double overlapT = MathMin(m_zones[i].top, m_zones[j].top);
         double overlapB = MathMax(m_zones[i].bottom, m_zones[j].bottom);
         if(overlapT > overlapB) {
            AddZone(ZONE_CONFLUENCE, overlapT, overlapB, TimeCurrent(), m_zones[i].isBullish, "CONFLUENCE("+m_zones[i].label+"+"+m_zones[j].label+")");
         }
      }
   }
}

void CLiquidity::UpdateMitigation(double cH, double cL) {
   for(int i=0; i<ArraySize(m_levels); i++) {
      if(!m_levels[i].mitigated) {
         if(m_levels[i].side==LIQ_ABOVE && cH >= m_levels[i].price) m_levels[i].mitigated=true;
         if(m_levels[i].side==LIQ_BELOW && cL <= m_levels[i].price) m_levels[i].mitigated=true;
      }
   }
   for(int i=0; i<m_zoneCount; i++) {
      if(!m_zones[i].mitigated) {
         if(m_zones[i].isBullish && cL <= m_zones[i].bottom) m_zones[i].mitigated=true;
         if(!m_zones[i].isBullish && cH >= m_zones[i].top) m_zones[i].mitigated=true;
      }
   }
}

void CLiquidity::BuildContext(double currentPrice) {
   m_context.totalAbove = 0; m_context.totalBelow = 0; m_context.nearestAbove.price = 0; m_context.nearestBelow.price = 0;
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
   m_context.equilibrium = 0; m_context.isPremium = false; m_context.isDiscount = false;
   if(m_swingCount >= 2) {
      double hh = 0, ll = DBL_MAX; for(int i=0; i<MathMin(m_swingCount,10); i++) { hh = MathMax(hh, m_swings[i].price); ll = MathMin(ll, m_swings[i].price); }
      m_context.equilibrium = (hh + ll) / 2.0;
      m_context.isPremium = (currentPrice > m_context.equilibrium);
      m_context.isDiscount = (currentPrice < m_context.equilibrium);
   }
   m_context.isValid = true;
}

void CLiquidity::AddLevel(double price, ENUM_LIQUIDITY_TYPE type, ENUM_LIQUIDITY_SIDE side, datetime time, double strength, string label) {
   for(int i = 0; i < ArraySize(m_levels); i++) if(MathAbs(m_levels[i].price - price) < m_equalThreshold && m_levels[i].side == side) return;
   if(ArraySize(m_levels) >= m_maxLevels) ArrayRemove(m_levels, 0, 1);
   SLiquidityLevel lvl; lvl.price = price; lvl.type = type; lvl.side = side; lvl.createdTime = time; lvl.mitigated = false; lvl.strength = strength; lvl.label = label;
   int sz = ArraySize(m_levels); ArrayResize(m_levels, sz + 1); m_levels[sz] = lvl;
}

void CLiquidity::AddZone(ENUM_ZONE_TYPE type, double t, double b, datetime tm, bool bull, string lbl) {
   if(m_zoneCount >= ArraySize(m_zones)) return;
   SInstitutionalZone z; z.type = type; z.top = t; z.bottom = b; z.mid = (t+b)/2; z.time = tm; z.isBullish = bull; z.mitigated = false; z.strength = (type==ZONE_CONFLUENCE?2.0:1.0); z.label = lbl;
   m_zones[m_zoneCount++] = z;
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

#endif
