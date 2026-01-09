//+------------------------------------------------------------------+
//|                                                   CVisuals.mqh |
//|             FEAT SNIPER HUD - Professional Dashboard              |
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
   CEMAs* m_ptrEmas;
   CFEAT* m_ptrFeat;
   CLiquidity* m_ptrLiq;
   CFSM* m_ptrFsm;
   bool           m_showDashboard;

   // Helpers gráficos
   void CreateRect(string name, int x, int y, int w, int h, color bg, color border);
   void CreateText(string name, int x, int y, string text, color clr, int size, string font="Consolas", bool bold=false);
   
public:
   CVisuals();
   ~CVisuals() { Clear(); }
   void Init(string prefix, long chartID) { m_prefix = prefix; m_chart = chartID; }
   void SetComponents(CEMAs* e, CFEAT* f, CLiquidity* l, CFSM* sm) { m_ptrEmas = e; m_ptrFeat = f; m_ptrLiq = l; m_ptrFsm = sm; }
   void SetDrawOptions(bool emas, bool dash, bool liq, bool structr) { m_showDashboard = dash; }
   void Draw(datetime t, double close);
   void Clear();
};

CVisuals::CVisuals() : m_ptrEmas(NULL), m_ptrFeat(NULL), m_ptrLiq(NULL), m_ptrFsm(NULL), m_showDashboard(true) {}

void CVisuals::Clear() { ObjectsDeleteAll(m_chart, m_prefix); }

void CVisuals::Draw(datetime t, double close) {
   // Clear previous HUD elements
   ObjectsDeleteAll(m_chart, m_prefix + "HUD");

   if(!m_showDashboard || m_ptrFeat == NULL) return;

   // 1. GET INTELLIGENCE
   SEngineerReport engineer = m_ptrFeat.GetEngineer();
   STacticianReport tactician = m_ptrFeat.GetTactician();
   SSniperReport sniper = m_ptrFeat.GetSniper();
   
   // 2. LAYOUT CONFIG
   int startX = 20;
   int startY = 40;
   int col1W = 200; 
   int col2W = 180;
   int padding = 5;
   
   // --- HEADER: SNIPER DECISION ---
   color headerColor = C'40,40,40';
   if(sniper.decision == "DISPARAR") {
      headerColor = (sniper.order.action == "BUY") ? C'0,180,60' : C'220,40,40'; 
   } else if(sniper.confidence > 50) {
      headerColor = C'150,120,0'; // Warning/Preparing
   }
   
   CreateRect(m_prefix+"HUD_HeadBG", startX, startY, col1W+col2W+padding, 40, headerColor, clrBlack);
   string title = "SNIPER: " + sniper.decision + " (" + DoubleToString(sniper.confidence, 0) + "%)";
   if(sniper.decision == "DISPARAR") title += " -> " + sniper.order.action;
   
   CreateText(m_prefix+"HUD_Title", startX + (col1W+col2W)/2, startY+20, title, clrWhite, 12, "Impact", true);

   // --- COLUMN 1: QUANTITATIVE ENGINEER (Physics) ---
   int y = startY + 45;
   CreateRect(m_prefix+"HUD_Col1BG", startX, y, col1W, 160, C'20,20,20', C'60,60,60');
   CreateText(m_prefix+"HUD_L_Eng", startX+5, y+5, "INGENIERO CUANTITATIVO", clrGold, 9, "Arial", true);
   
   y += 20;
   CreateText(m_prefix+"HUD_Trend", startX+5, y, "Vector: " + engineer.trend, clrWhite, 8);
   y += 15;
   CreateText(m_prefix+"HUD_Pres", startX+5, y, "Presion: " + engineer.pressure, clrSilver, 8);
   y += 15;
   CreateText(m_prefix+"HUD_RSI", startX+5, y, "RSI: " + engineer.rsiState, clrWhite, 8);
   y += 15;
   CreateText(m_prefix+"HUD_Path", startX+5, y, "Ruta: " + engineer.criticalPath, clrAqua, 8);
   y += 15;
   CreateText(m_prefix+"HUD_Energy", startX+5, y, "Energia: " + engineer.energyState, clrWhite, 8);
   y += 15;
   CreateText(m_prefix+"HUD_DFlow", startX+5, y, "DeltaFlow: " + DoubleToString(sniper.deltaFlow, 2), (sniper.deltaFlow>0?clrLime:clrRed), 8);
   y += 15;
   CreateText(m_prefix+"HUD_Ord", startX+5, y, "ORDEN: " + engineer.engineerOrder, clrGold, 8, "Consolas", true);

   // --- COLUMN 2: TACTICIAN (Space-Time) ---
   int x2 = startX + col1W + padding;
   y = startY + 45;
   CreateRect(m_prefix+"HUD_Col2BG", x2, y, col2W, 160, C'20,20,20', C'60,60,60');
   CreateText(m_prefix+"HUD_L_Tac", x2+5, y+5, "TACTICO (TIEMPO-ESPACIO)", clrSilver, 9, "Arial", true);
   
   y += 20;
   color timeColor = tactician.isOperableTime ? clrLime : clrRed;
   CreateText(m_prefix+"HUD_Time", x2+5, y, "Hora: " + tactician.currentTime + " (" + EnumToString(tactician.sessionState)+")", timeColor, 8);
   y += 15;
   CreateText(m_prefix+"HUD_POI", x2+5, y, "POI: " + tactician.poiDetected, (tactician.poiDetected!="NONE"?clrGold:clrSilver), 8);
   y += 15;
   CreateText(m_prefix+"HUD_Loc", x2+5, y, "Ubic: " + tactician.locationRelative, clrWhite, 8);
   y += 15;
   CreateText(m_prefix+"HUD_Sep", x2+5, y, "Separacion: " + DoubleToString(tactician.layerSeparation, 5), clrSilver, 8);
   y += 30;
   CreateText(m_prefix+"HUD_Act", x2+5, y, "VERDICTO: " + tactician.action, (tactician.action=="BUSCAR_GATILLO"?clrLime:clrWhite), 9, "Arial", true);

   // --- FOOTER: DIAGNOSIS ---
   y = startY + 45 + 160 + padding;
   CreateRect(m_prefix+"HUD_FootBG", startX, y, col1W+col2W+padding, 25, C'10,10,10', C'60,60,60');
   CreateText(m_prefix+"HUD_Diag", startX+5, y+5, "DIAGNOSTICO: " + sniper.finalReason, clrWhite, 8);
   
   // --- SNIPER TRIGGER PANEL (IF SHOOT) ---
   if(sniper.decision == "DISPARAR") {
      y += 30;
      CreateRect(m_prefix+"HUD_SnipBG", startX, y, col1W+col2W+padding, 40, C'0,50,0', clrLime);
      string trade = "ENTRY: " + DoubleToString(sniper.order.entryPrice, _Digits) + 
                     "  SL: " + DoubleToString(sniper.order.slPrice, _Digits) + 
                     "  TP: " + DoubleToString(sniper.order.tpPrice, _Digits);
      CreateText(m_prefix+"HUD_Trade", startX+200, y+20, trade, clrWhite, 9, "Consolas", true);
   }

   ChartRedraw(m_chart);
}

// --- HELPERS ---
void CVisuals::CreateRect(string name, int x, int y, int w, int h, color bg, color border) {
   ObjectCreate(m_chart, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(m_chart, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(m_chart, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(m_chart, name, OBJPROP_XSIZE, w);
   ObjectSetInteger(m_chart, name, OBJPROP_YSIZE, h);
   ObjectSetInteger(m_chart, name, OBJPROP_BGCOLOR, bg);
   ObjectSetInteger(m_chart, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(m_chart, name, OBJPROP_COLOR, border);
   ObjectSetInteger(m_chart, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(m_chart, name, OBJPROP_BACK, false);
}

void CVisuals::CreateText(string name, int x, int y, string text, color clr, int size, string font, bool bold) {
   ObjectCreate(m_chart, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(m_chart, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(m_chart, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(m_chart, name, OBJPROP_TEXT, text);
   ObjectSetInteger(m_chart, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(m_chart, name, OBJPROP_FONTSIZE, size);
   ObjectSetString(m_chart, name, OBJPROP_FONT, font);
   ObjectSetInteger(m_chart, name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
   if(StringFind(name, "Title") >= 0 || StringFind(name, "Trade") >= 0)
       ObjectSetInteger(m_chart, name, OBJPROP_ANCHOR, ANCHOR_CENTER);
}
#endif
