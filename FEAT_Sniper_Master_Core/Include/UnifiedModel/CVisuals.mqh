//+------------------------------------------------------------------+
//|                                                    CVisuals.mqh |
//|                    Visualization Layer                          |
//|            Panel, EMAs Ribbon, Liquidity Zones                  |
//+------------------------------------------------------------------+
#ifndef CVISUALS_MQH
#define CVISUALS_MQH

#include "CEMAs.mqh"
#include "CFEAT.mqh"
#include "CLiquidity.mqh"
#include "CFSM.mqh"

//+------------------------------------------------------------------+
//| COLOR SCHEME                                                     |
//+------------------------------------------------------------------+
struct SColorScheme {
   // EMA Groups
   color    emaFast;
   color    emaMedium;
   color    emaSlow;
   
   // States
   color    stateAccumulation;
   color    stateExpansion;
   color    stateDistribution;
   color    stateReset;
   
   // Liquidity
   color    liqExternal;
   color    liqInternal;
   color    liqImbalance;
   
   // Panel
   color    panelBg;
   color    panelText;
   color    panelHighlight;
};

//+------------------------------------------------------------------+
//| CVISUALS CLASS                                                   |
//+------------------------------------------------------------------+
class CVisuals {
private:
   string            m_prefix;
   long              m_chartId;
   SColorScheme      m_colors;
   
   bool              m_drawEMAs;
   bool              m_drawPanel;
   bool              m_drawLiquidity;
   bool              m_drawFEAT;
   
   int               m_panelX;
   int               m_panelY;
   int               m_panelWidth;
   int               m_lineHeight;
   int               m_fontSize;
   
   // Component references
   CEMAs*            m_emas;
   CFEAT*            m_feat;
   CLiquidity*       m_liquidity;
   CFSM*             m_fsm;
   
   // Helper methods
   void              CreateLabel(string name, int x, int y, string text, color clr, int size = 9);
   void              CreateRectangle(string name, int x, int y, int width, int height, color bg, color border);
   void              CreateLine(string name, datetime t1, double p1, datetime t2, double p2, color clr, int width = 1, ENUM_LINE_STYLE style = STYLE_SOLID);
   void              CreateRectanglePrice(string name, datetime t1, double p1, datetime t2, double p2, color clr, bool fill = true);
   color             GetStateColor(ENUM_MARKET_STATE state);
   string            GetFormString(ENUM_FORM_TYPE type);
   string            GetSpaceString(ENUM_SPACE_TYPE type);
   string            GetAccelString(ENUM_ACCEL_TYPE type);
   
public:
                     CVisuals();
                    ~CVisuals();
   
   // Configuration
   void              Init(string prefix, long chartId = 0);
   void              SetComponents(CEMAs* emas, CFEAT* feat, CLiquidity* liq, CFSM* fsm);
   void              SetColorScheme(const SColorScheme &scheme);
   void              SetDrawOptions(bool emas, bool panel, bool liquidity, bool feat);
   void              SetPanelPosition(int x, int y);
   
   // Drawing
   void              Draw(datetime currentTime, double currentPrice);
   void              DrawEMAs(datetime startTime, datetime endTime);
   void              DrawPanel();
   void              DrawLiquidity(datetime startTime, datetime endTime);
   void              DrawFEATInfo();
   
   // Cleanup
   void              Clear();
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CVisuals::CVisuals() {
   m_prefix = "UM_";
   m_chartId = 0;
   
   m_drawEMAs = true;
   m_drawPanel = true;
   m_drawLiquidity = true;
   m_drawFEAT = true;
   
   m_panelX = 10;
   m_panelY = 30;
   m_panelWidth = 250;
   m_lineHeight = 18;
   m_fontSize = 9;
   
   // Default color scheme
   m_colors.emaFast = clrYellow;
   m_colors.emaMedium = clrOrange;
   m_colors.emaSlow = clrRed;
   
   m_colors.stateAccumulation = clrDodgerBlue;
   m_colors.stateExpansion = clrLime;
   m_colors.stateDistribution = clrOrange;
   m_colors.stateReset = clrMagenta;
   
   m_colors.liqExternal = clrCrimson;
   m_colors.liqInternal = clrGold;
   m_colors.liqImbalance = clrDarkViolet;
   
   m_colors.panelBg = C'20,20,30';
   m_colors.panelText = clrWhite;
   m_colors.panelHighlight = clrGold;
   
   m_emas = NULL;
   m_feat = NULL;
   m_liquidity = NULL;
   m_fsm = NULL;
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CVisuals::~CVisuals() {
   Clear();
}

//+------------------------------------------------------------------+
//| Initialize                                                       |
//+------------------------------------------------------------------+
void CVisuals::Init(string prefix, long chartId = 0) {
   m_prefix = prefix;
   m_chartId = chartId;
}

//+------------------------------------------------------------------+
//| Set Components                                                   |
//+------------------------------------------------------------------+
void CVisuals::SetComponents(CEMAs* emas, CFEAT* feat, CLiquidity* liq, CFSM* fsm) {
   m_emas = emas;
   m_feat = feat;
   m_liquidity = liq;
   m_fsm = fsm;
}

//+------------------------------------------------------------------+
//| Set Color Scheme                                                 |
//+------------------------------------------------------------------+
void CVisuals::SetColorScheme(const SColorScheme &scheme) {
   m_colors = scheme;
}

//+------------------------------------------------------------------+
//| Set Draw Options                                                 |
//+------------------------------------------------------------------+
void CVisuals::SetDrawOptions(bool emas, bool panel, bool liquidity, bool feat) {
   m_drawEMAs = emas;
   m_drawPanel = panel;
   m_drawLiquidity = liquidity;
   m_drawFEAT = feat;
}

//+------------------------------------------------------------------+
//| Set Panel Position                                               |
//+------------------------------------------------------------------+
void CVisuals::SetPanelPosition(int x, int y) {
   m_panelX = x;
   m_panelY = y;
}

//+------------------------------------------------------------------+
//| Main Draw Method                                                 |
//+------------------------------------------------------------------+
void CVisuals::Draw(datetime currentTime, double currentPrice) {
   Clear();
   
   datetime startTime = currentTime - PeriodSeconds(PERIOD_CURRENT) * 100;
   datetime endTime = currentTime + PeriodSeconds(PERIOD_CURRENT) * 5;
   
   if(m_drawLiquidity && m_liquidity != NULL) DrawLiquidity(startTime, endTime);
   if(m_drawPanel) DrawPanel();
   
   ChartRedraw(m_chartId);
}

//+------------------------------------------------------------------+
//| Draw EMAs as Ribbons                                             |
//+------------------------------------------------------------------+
void CVisuals::DrawEMAs(datetime startTime, datetime endTime) {
   if(m_emas == NULL || !m_emas.IsReady()) return;
   
   // Note: In practice, EMAs are drawn as indicator buffers
   // This method can be used for additional visual elements
}

//+------------------------------------------------------------------+
//| Draw Info Panel                                                  |
//+------------------------------------------------------------------+
void CVisuals::DrawPanel() {
   int x = m_panelX;
   int y = m_panelY;
   int h = m_lineHeight;
   int panelHeight = h * 14 + 15;
   
   // Background
   CreateRectangle(m_prefix + "PanelBG", x - 5, y - 5, m_panelWidth, panelHeight, m_colors.panelBg, clrGray);
   
   // Title
   CreateLabel(m_prefix + "Title", x, y, "=== UNIFIED MODEL ===", m_colors.panelHighlight, 10);
   y += h + 5;
   
   // State
   if(m_fsm != NULL) {
      ENUM_MARKET_STATE state = m_fsm.GetState();
      color stateClr = GetStateColor(state);
      CreateLabel(m_prefix + "State", x, y, "State: " + m_fsm.GetStateString(), stateClr, m_fontSize);
      y += h;
      
      CreateLabel(m_prefix + "Conf", x, y, StringFormat("Confidence: %.1f%%", m_fsm.GetConfidence()), m_colors.panelText, m_fontSize);
      y += h;
      
      SFSMMetrics metrics = m_fsm.GetMetrics();
      CreateLabel(m_prefix + "Effort", x, y, StringFormat("Effort: %.3f", metrics.effort), m_colors.panelText, m_fontSize);
      y += h;
      CreateLabel(m_prefix + "Result", x, y, StringFormat("Result: %.3f", metrics.result), m_colors.panelText, m_fontSize);
      y += h;
   }
   
   // FEAT
   y += 5;
   CreateLabel(m_prefix + "FeatTitle", x, y, "--- FEAT ---", clrSilver, m_fontSize);
   y += h;
   
   if(m_feat != NULL) {
      SFormMetrics form = m_feat.GetForm();
      SSpaceMetrics space = m_feat.GetSpace();
      SAccelMetrics accel = m_feat.GetAccel();
      STimeMetrics time = m_feat.GetTime();
      
      CreateLabel(m_prefix + "Form", x, y, "F: " + GetFormString(form.type), m_colors.panelText, m_fontSize);
      y += h;
      CreateLabel(m_prefix + "Space", x, y, "E: " + GetSpaceString(space.type), m_colors.panelText, m_fontSize);
      y += h;
      CreateLabel(m_prefix + "Accel", x, y, "A: " + GetAccelString(accel.type), m_colors.panelText, m_fontSize);
      y += h;
      CreateLabel(m_prefix + "Time", x, y, StringFormat("T: %s [%.1fx]", time.activeSession, time.tfMultiplier), m_colors.panelText, m_fontSize);
      y += h;
      
      color scoreClr = (m_feat.GetCompositeScore() > 60) ? clrLime : ((m_feat.GetCompositeScore() < 40) ? clrRed : clrYellow);
      CreateLabel(m_prefix + "FeatScore", x, y, StringFormat("FEAT Score: %.0f", m_feat.GetCompositeScore()), scoreClr, m_fontSize);
      y += h;
   }
   
   // Liquidity
   y += 5;
   CreateLabel(m_prefix + "LiqTitle", x, y, "--- LIQUIDITY ---", clrSilver, m_fontSize);
   y += h;
   
   if(m_liquidity != NULL) {
      SLiquidityContext ctx = m_liquidity.GetContext();
      string aboveStr = (ctx.nearestAbove.price > 0) ? DoubleToString(ctx.nearestAbove.price, _Digits) : "---";
      string belowStr = (ctx.nearestBelow.price > 0) ? DoubleToString(ctx.nearestBelow.price, _Digits) : "---";
      
      CreateLabel(m_prefix + "LiqAbove", x, y, "Above: " + aboveStr, m_colors.liqExternal, m_fontSize);
      y += h;
      CreateLabel(m_prefix + "LiqBelow", x, y, "Below: " + belowStr, m_colors.liqExternal, m_fontSize);
   }
}

//+------------------------------------------------------------------+
//| Draw Liquidity Levels                                            |
//+------------------------------------------------------------------+
void CVisuals::DrawLiquidity(datetime startTime, datetime endTime) {
   if(m_liquidity == NULL) return;
   
   int count = m_liquidity.GetLevelCount();
   for(int i = 0; i < MathMin(count, 30); i++) {
      SLiquidityLevel lvl = m_liquidity.GetLevel(i);
      if(lvl.mitigated) continue;
      
      color clr = (lvl.type == LIQ_EXTERNAL) ? m_colors.liqExternal :
                  (lvl.type == LIQ_INTERNAL) ? m_colors.liqInternal : m_colors.liqImbalance;
      
      ENUM_LINE_STYLE style = (lvl.type == LIQ_EXTERNAL) ? STYLE_SOLID : STYLE_DOT;
      int width = (lvl.strength > 0.7) ? 2 : 1;
      
      string name = m_prefix + "Liq_" + IntegerToString(i);
      CreateLine(name, startTime, lvl.price, endTime, lvl.price, clr, width, style);
   }
   
   // Draw imbalances
   int imbCount = m_liquidity.GetImbalanceCount();
   for(int i = 0; i < MathMin(imbCount, 10); i++) {
      SImbalance imb = m_liquidity.GetImbalance(i);
      if(imb.mitigated) continue;
      
      string name = m_prefix + "Imb_" + IntegerToString(i);
      CreateRectanglePrice(name, imb.time, imb.high, TimeCurrent(), imb.low, m_colors.liqImbalance, false);
   }
}

//+------------------------------------------------------------------+
//| Draw FEAT Info                                                   |
//+------------------------------------------------------------------+
void CVisuals::DrawFEATInfo() {
   // Additional FEAT visualization can be added here
}

//+------------------------------------------------------------------+
//| Clear All Objects                                                |
//+------------------------------------------------------------------+
void CVisuals::Clear() {
   ObjectsDeleteAll(m_chartId, m_prefix);
}

//+------------------------------------------------------------------+
//| Create Label Helper (Optimized)                                  |
//+------------------------------------------------------------------+
void CVisuals::CreateLabel(string name, int x, int y, string text, color clr, int size = 9) {
   if(ObjectFind(m_chartId, name) < 0) {
      ObjectCreate(m_chartId, name, OBJ_LABEL, 0, 0, 0);
      ObjectSetString(m_chartId, name, OBJPROP_FONT, "Consolas");
      ObjectSetInteger(m_chartId, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   }
   
   ObjectSetInteger(m_chartId, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(m_chartId, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(m_chartId, name, OBJPROP_TEXT, text);
   ObjectSetInteger(m_chartId, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(m_chartId, name, OBJPROP_FONTSIZE, size);
}

//+------------------------------------------------------------------+
//| Create Rectangle Helper (Optimized)                              |
//+------------------------------------------------------------------+
void CVisuals::CreateRectangle(string name, int x, int y, int width, int height, color bg, color border) {
   if(ObjectFind(m_chartId, name) < 0) {
      ObjectCreate(m_chartId, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
      ObjectSetInteger(m_chartId, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(m_chartId, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(m_chartId, name, OBJPROP_BACK, false);
   }
   
   ObjectSetInteger(m_chartId, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(m_chartId, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(m_chartId, name, OBJPROP_XSIZE, width);
   ObjectSetInteger(m_chartId, name, OBJPROP_YSIZE, height);
   ObjectSetInteger(m_chartId, name, OBJPROP_BGCOLOR, bg);
   ObjectSetInteger(m_chartId, name, OBJPROP_COLOR, border);
}

//+------------------------------------------------------------------+
//| Create Line Helper (Optimized)                                   |
//+------------------------------------------------------------------+
void CVisuals::CreateLine(string name, datetime t1, double p1, datetime t2, double p2, color clr, int width = 1, ENUM_LINE_STYLE style = STYLE_SOLID) {
   if(ObjectFind(m_chartId, name) < 0) {
      ObjectCreate(m_chartId, name, OBJ_TREND, 0, t1, p1, t2, p2);
      ObjectSetInteger(m_chartId, name, OBJPROP_RAY_RIGHT, false);
      ObjectSetInteger(m_chartId, name, OBJPROP_BACK, true);
   } else {
      ObjectSetInteger(m_chartId, name, OBJPROP_TIME, 0, t1);
      ObjectSetDouble(m_chartId, name, OBJPROP_PRICE, 0, p1);
      ObjectSetInteger(m_chartId, name, OBJPROP_TIME, 1, t2);
      ObjectSetDouble(m_chartId, name, OBJPROP_PRICE, 1, p2);
   }
   
   ObjectSetInteger(m_chartId, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(m_chartId, name, OBJPROP_WIDTH, width);
   ObjectSetInteger(m_chartId, name, OBJPROP_STYLE, style);
}

//+------------------------------------------------------------------+
//| Create Price Rectangle Helper (Optimized)                        |
//+------------------------------------------------------------------+
void CVisuals::CreateRectanglePrice(string name, datetime t1, double p1, datetime t2, double p2, color clr, bool fill = true) {
   if(ObjectFind(m_chartId, name) < 0) {
      ObjectCreate(m_chartId, name, OBJ_RECTANGLE, 0, t1, p1, t2, p2);
      ObjectSetInteger(m_chartId, name, OBJPROP_BACK, true);
   } else {
      ObjectSetInteger(m_chartId, name, OBJPROP_TIME, 0, t1);
      ObjectSetDouble(m_chartId, name, OBJPROP_PRICE, 0, p1);
      ObjectSetInteger(m_chartId, name, OBJPROP_TIME, 1, t2);
      ObjectSetDouble(m_chartId, name, OBJPROP_PRICE, 1, p2);
   }
   
   ObjectSetInteger(m_chartId, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(m_chartId, name, OBJPROP_FILL, fill);
}

//+------------------------------------------------------------------+
//| Get State Color                                                  |
//+------------------------------------------------------------------+
color CVisuals::GetStateColor(ENUM_MARKET_STATE state) {
   switch(state) {
      case STATE_ACCUMULATION: return m_colors.stateAccumulation;
      case STATE_EXPANSION:    return m_colors.stateExpansion;
      case STATE_DISTRIBUTION: return m_colors.stateDistribution;
      case STATE_RESET:        return m_colors.stateReset;
   }
   return m_colors.panelText;
}

//+------------------------------------------------------------------+
//| Get Form Type as String                                          |
//+------------------------------------------------------------------+
string CVisuals::GetFormString(ENUM_FORM_TYPE type) {
   switch(type) {
      case FORM_CONSTRUCTION: return "Construction";
      case FORM_IMPULSE:      return "Impulse";
      case FORM_EXHAUSTION:   return "Exhaustion";
   }
   return "Undefined";
}

//+------------------------------------------------------------------+
//| Get Space Type as String                                         |
//+------------------------------------------------------------------+
string CVisuals::GetSpaceString(ENUM_SPACE_TYPE type) {
   switch(type) {
      case SPACE_EXPANDED:   return "Expanded";
      case SPACE_COMPRESSED: return "Compressed";
      case SPACE_VOID:       return "Void";
   }
   return "Normal";
}

//+------------------------------------------------------------------+
//| Get Accel Type as String                                         |
//+------------------------------------------------------------------+
string CVisuals::GetAccelString(ENUM_ACCEL_TYPE type) {
   switch(type) {
      case ACCEL_VALID: return "Valid";
      case ACCEL_FAKE:  return "Fake";
   }
   return "None";
}

#endif // CVISUALS_MQH
