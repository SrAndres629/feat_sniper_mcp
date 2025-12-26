//+------------------------------------------------------------------+
//|                                                   CVisuals.mqh |
//|                    Visualization Engine - Dashboard + Levels      |
//+------------------------------------------------------------------+
#ifndef CVISUALS_MQH
#define CVISUALS_MQH

#include "CEMAs.mqh"
#include "CFEAT.mqh"
#include "CLiquidity.mqh"
#include "CFSM.mqh"

class CVisuals {
private:
   string         m_prefix;
   long           m_chart;
   CEMAs*         m_ptrEmas;
   CFEAT*         m_ptrFeat;
   CLiquidity*    m_ptrLiq;
   CFSM*          m_ptrFsm;
   
   bool           m_showEMAs;
   bool           m_showDashboard;
   bool           m_showLiquidity;
   bool           m_showStructure;

   void           CreateLabel(string name, int x, int y, string text, color clr, int size);
   void           CreateHLine(string name, double price, color clr, ENUM_LINE_STYLE style, int width);
   void           CreateBox(string name, datetime t1, double p1, datetime t2, double p2, color clr, int width, ENUM_LINE_STYLE style, bool fill);
   void           CreateArrow(string name, datetime t, double price, int code, color clr);

public:
   CVisuals();
   ~CVisuals() { Clear(); }

   void Init(string prefix, long chartID) { m_prefix = prefix; m_chart = chartID; }
   void SetComponents(CEMAs* e, CFEAT* f, CLiquidity* l, CFSM* sm) { m_ptrEmas = e; m_ptrFeat = f; m_ptrLiq = l; m_ptrFsm = sm; }
   void SetDrawOptions(bool emas, bool dash, bool liq, bool structr) { m_showEMAs = emas; m_showDashboard = dash; m_showLiquidity = liq; m_showStructure = structr; }

   void Draw(datetime t, double close);
   void Clear();
};

CVisuals::CVisuals() : m_ptrEmas(NULL), m_ptrFeat(NULL), m_ptrLiq(NULL), m_ptrFsm(NULL), m_showEMAs(true), m_showDashboard(true), m_showLiquidity(true), m_showStructure(true) {}

void CVisuals::Clear() { ObjectsDeleteAll(m_chart, m_prefix); }

void CVisuals::Draw(datetime t, double close) {
   ObjectsDeleteAll(m_chart, m_prefix);
   int x = 20, y = 30, h = 18;

   if(m_showDashboard && m_ptrFsm != NULL && m_ptrEmas != NULL && m_ptrFeat != NULL && m_ptrLiq != NULL) {
      CreateLabel(m_prefix+"Title", x, y, "FEAT SNIPER V3 - INSTITUTIONAL", clrGold, 10); y += h + 5;
      
      // Context: Premium/Discount
      SLiquidityContext lctx = m_ptrLiq.GetContext();
      string pdLabel = "EQUILIBRIUM";
      color pdClr = clrSilver;
      if(lctx.isPremium) { pdLabel = "PREMIUM"; pdClr = clrTomato; }
      if(lctx.isDiscount) { pdLabel = "DISCOUNT"; pdClr = clrLime; }
      CreateLabel(m_prefix+"PD_Status", x, y, "CONTEXT: " + pdLabel, pdClr, 9); y += h;
      
      // State
      CreateLabel(m_prefix+"State", x, y, "STATE: " + m_ptrFsm.GetStateString(), clrDodgerBlue, 8); y += h;
      
      // Time Module
      STimeMetrics tm = m_ptrFeat.GetTime();
      string kzLabel = tm.isKillzone ? (tm.isLondonKZ ? "[LONDON KZ]" : "[NY KZ]") : "";
      color kzClr = tm.isKillzone ? clrCyan : clrGray;
      CreateLabel(m_prefix+"Session", x, y, "SESSION: " + tm.activeSession + " " + kzLabel, kzClr, 8); y += h;
      
      // Acceleration
      SAccelMetrics acm = m_ptrFeat.GetAccel();
      color accelClr = acm.isInstitutional ? clrCyan : (acm.velocity > 1.0 ? clrLime : clrWhite);
      CreateLabel(m_prefix+"Accel", x, y, StringFormat("ACCEL: %.2f | Vel: %.2f", acm.momentum, acm.velocity), accelClr, 8); y += h;
      
      // Alerts
      if(acm.isInstitutional) {
         CreateLabel(m_prefix+"Intent", x, y, ">> INTENCION INSTITUCIONAL", clrCyan, 9); y += h;
      }
      if(acm.isExhausted) {
         CreateLabel(m_prefix+"Exhaust", x, y, "!! AGOTAMIENTO DETECTADO", clrOrange, 9); y += h;
      }
      
      // Composite Score
      double score = m_ptrFeat.GetCompositeScore();
      color scoreClr = (score > 70) ? clrLime : (score > 40 ? clrYellow : clrRed);
      CreateLabel(m_prefix+"Score", x, y, StringFormat("SCORE: %.0f/100", score), scoreClr, 9); y += h + 5;
      
      // Equilibrium Line
      if(lctx.equilibrium > 0) {
         CreateHLine(m_prefix+"EQ", lctx.equilibrium, clrDarkGray, STYLE_DASHDOT, 1);
      }
   }

   if(m_ptrLiq != NULL) {
      // Structure Lines (BOS/CHoCH)
      if(m_showStructure) {
         int sCount = m_ptrLiq.GetStructureCount();
         for(int i = 0; i < sCount; i++) {
            SStructureEvent e = m_ptrLiq.GetStructureEvent(i);
            if(e.active) {
               string name = m_prefix + "STRUCT_" + IntegerToString(i);
               color c = e.isBullish ? clrLime : clrRed;
               ENUM_LINE_STYLE st = (e.type == STRUCT_BOS) ? STYLE_SOLID : STYLE_DASH;
               ObjectCreate(m_chart, name, OBJ_TREND, 0, e.time, e.price, TimeCurrent(), e.price);
               ObjectSetInteger(m_chart, name, OBJPROP_COLOR, c);
               ObjectSetInteger(m_chart, name, OBJPROP_STYLE, st);
               ObjectSetInteger(m_chart, name, OBJPROP_RAY_RIGHT, true);
               
               string lblName = name + "_LBL";
               ObjectCreate(m_chart, lblName, OBJ_TEXT, 0, e.time, e.price);
               ObjectSetString(m_chart, lblName, OBJPROP_TEXT, (e.type == STRUCT_BOS ? " BOS" : " CHoCH"));
               ObjectSetInteger(m_chart, lblName, OBJPROP_COLOR, c);
               ObjectSetInteger(m_chart, lblName, OBJPROP_FONTSIZE, 7);
            }
         }
         
         // Patterns (M/W/GC)
         int pCount = m_ptrLiq.GetPatternCount();
         for(int i = 0; i < pCount; i++) {
            SPatternEvent p = m_ptrLiq.GetPatternEvent(i);
            string name = m_prefix + "PAT_" + IntegerToString(i);
            color c = p.isBullish ? clrLime : clrRed;
            int code = p.isBullish ? 233 : 234; // Arrow up/down
            CreateArrow(name, p.time, p.price, code, c);
            
            string lblName = name + "_LBL";
            ObjectCreate(m_chart, lblName, OBJ_TEXT, 0, p.time, p.price);
            ObjectSetString(m_chart, lblName, OBJPROP_TEXT, " " + p.description);
            ObjectSetInteger(m_chart, lblName, OBJPROP_COLOR, c);
            ObjectSetInteger(m_chart, lblName, OBJPROP_FONTSIZE, 8);
         }
      }
      
      // Zones
      if(m_showLiquidity) {
         int zCount = m_ptrLiq.GetZoneCount();
         for(int i = 0; i < zCount; i++) {
            SInstitutionalZone z = m_ptrLiq.GetZone(i);
            if(z.mitigated) continue;
            string name = m_prefix + "ZONE_" + IntegerToString(i);
            color clrZone = clrDimGray;
            if(z.type == ZONE_FVG) clrZone = z.isBullish ? C'30,50,30' : C'50,30,30';
            if(z.type == ZONE_OB) clrZone = z.isBullish ? C'0,100,0' : C'100,0,0';
            if(z.type == ZONE_CONFLUENCE) clrZone = clrFuchsia;
            if(z.type == ZONE_PC) clrZone = clrGoldenrod;
            
            CreateBox(name, z.time, z.top, TimeCurrent(), z.bottom, clrZone, 1, STYLE_SOLID, true);
            
            string lblName = name + "_LBL";
            ObjectCreate(m_chart, lblName, OBJ_TEXT, 0, z.time, z.top);
            ObjectSetString(m_chart, lblName, OBJPROP_TEXT, " " + z.label);
            ObjectSetInteger(m_chart, lblName, OBJPROP_COLOR, clrWhite);
            ObjectSetInteger(m_chart, lblName, OBJPROP_FONTSIZE, 7);
         }
         
         // Liquidity Levels
         for(int i=0; i<m_ptrLiq.GetLevelCount(); i++) {
            SLiquidityLevel lvl = m_ptrLiq.GetLevel(i);
            if(lvl.mitigated) continue;
            string name = m_prefix + "LVL_" + IntegerToString(i);
            color c = (lvl.side==LIQ_ABOVE ? clrOrangeRed : clrMediumSeaGreen);
            int w = 1;
            if(lvl.type == LIQ_POOL) { w = 2; c = clrYellow; }
            CreateHLine(name, lvl.price, c, STYLE_DOT, w);
         }
      }
   }
   
   ChartRedraw(m_chart);
}

void CVisuals::CreateLabel(string name, int x, int y, string text, color clr, int size) {
   ObjectCreate(m_chart, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(m_chart, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(m_chart, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(m_chart, name, OBJPROP_TEXT, text);
   ObjectSetInteger(m_chart, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(m_chart, name, OBJPROP_FONTSIZE, size);
}

void CVisuals::CreateHLine(string name, double price, color clr, ENUM_LINE_STYLE style, int width) {
   ObjectCreate(m_chart, name, OBJ_HLINE, 0, 0, price);
   ObjectSetInteger(m_chart, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(m_chart, name, OBJPROP_STYLE, style);
   ObjectSetInteger(m_chart, name, OBJPROP_WIDTH, width);
}

void CVisuals::CreateBox(string name, datetime t1, double p1, datetime t2, double p2, color clr, int width, ENUM_LINE_STYLE style, bool fill) {
   ObjectCreate(m_chart, name, OBJ_RECTANGLE, 0, t1, p1, t2, p2);
   ObjectSetInteger(m_chart, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(m_chart, name, OBJPROP_WIDTH, width);
   ObjectSetInteger(m_chart, name, OBJPROP_STYLE, style);
   ObjectSetInteger(m_chart, name, OBJPROP_FILL, fill);
   ObjectSetInteger(m_chart, name, OBJPROP_BACK, true);
}

void CVisuals::CreateArrow(string name, datetime t, double price, int code, color clr) {
   ObjectCreate(m_chart, name, OBJ_ARROW, 0, t, price);
   ObjectSetInteger(m_chart, name, OBJPROP_ARROWCODE, code);
   ObjectSetInteger(m_chart, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(m_chart, name, OBJPROP_WIDTH, 2);
}

#endif
