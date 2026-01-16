
//+------------------------------------------------------------------+
//|                                                     CVisuals.mqh |
//|                    Senior Glass HUD V2 - FEAT SNIPER             |
//|           "Precision is not an act, it's a habit"                |
//+------------------------------------------------------------------+
#ifndef CVISUALS_MQH
#define CVISUALS_MQH

#include <ChartObjects\ChartObjectsTxtControls.mqh>
#include <ChartObjects\ChartObjectsShapes.mqh>
#include "CFEAT.mqh"
#include "CFSM.mqh"

// Colors
#define HUD_BG_COLOR      C'20,20,30'
#define HUD_BORDER_COLOR  clrDarkSlateGray
#define HUD_ACCENT        clrCyan
#define HUD_NEURAL_COLOR  clrMediumSpringGreen
#define HUD_ALARM_COLOR   clrOrangeRed

class CVisuals {
private:
   CChartObjectRectLabel m_bg;
   CChartObjectLabel     m_lblTitle;
   CChartObjectLabel     m_lblPilot;      // Who's driving? (Neural / LLM / Manual)
   CChartObjectLabel     m_lblConfidence; // Progress bar representation?
   CChartObjectLabel     m_lblState;
   CChartObjectLabel     m_lblScore;
   CChartObjectLabel     m_lblVault;      // Vault Status
   CChartObjectLabel     m_lblRegime;     // Market Regime
   
   string            m_symbol;
   long              m_chartId;
   int               m_x;
   int               m_y;
   int               m_w;
   int               m_h;

public:
   CVisuals();
   ~CVisuals();

   void Init(string symbol, int x = 20, int y = 50);
   void Update(SFEATResult &feat, string state);
   void SetPilot(string pilot, double confidence);
   void SetVault(double balance, string status);
   void Clear();
   
private:
   void CreateLabel(CChartObjectLabel &lbl, string name, int x, int y, int size, color col, string text);
};

CVisuals::CVisuals() : m_chartId(0), m_x(20), m_y(50), m_w(220), m_h(180) {}

CVisuals::~CVisuals() {
   Clear();
}

void CVisuals::Init(string symbol, int x, int y) {
   m_symbol = symbol;
   m_chartId = ChartID();
   m_x = x;
   m_y = y;

   // 1. Background (Glass Effect)
   m_bg.Create(m_chartId, "FEAT_HUD_BG", 0, m_x-10, m_y-10, m_w, m_h);
   m_bg.BackColor(HUD_BG_COLOR);
   m_bg.BorderType(BORDER_FLAT);
   m_bg.Color(HUD_BORDER_COLOR);
   m_bg.Width(1);
   m_bg.Selectable(false);
   m_bg.Z_Order(0);

   int curY = m_y;
   int step = 20;

   // 2. Header
   CreateLabel(m_lblTitle, "FEAT_HUD_Title", m_x, curY, 12, HUD_ACCENT, ">> FEAT SNIPER COCKPIT V2");
   curY += step + 5;

   // 3. Pilot Status
   CreateLabel(m_lblPilot, "FEAT_HUD_Pilot", m_x, curY, 10, clrWhite, "PILOT: CO-PILOT (LLM)");
   curY += step;
   
   CreateLabel(m_lblConfidence, "FEAT_HUD_Conf", m_x, curY, 9, clrGray, "CONFIDENCE: [----------] 0%");
   curY += step + 8;

   // 4. Market Physics
   CreateLabel(m_lblState, "FEAT_HUD_State", m_x, curY, 10, clrWhite, "STATE: SCANNING...");
   curY += step;
   
   CreateLabel(m_lblScore, "FEAT_HUD_Score", m_x, curY, 14, clrGold, "SCORE: 0.0");
   curY += step + 5;

   CreateLabel(m_lblRegime, "FEAT_HUD_Regime", m_x, curY, 9, clrSkyBlue, "REGIME: CALIBRATING");
   curY += step;

   // 5. Vault Area
   CreateLabel(m_lblVault, "FEAT_HUD_Vault", m_x, curY, 9, clrMediumSpringGreen, "VAULT: $0.00 (LOCKED)");
}

void CVisuals::CreateLabel(CChartObjectLabel &lbl, string name, int x, int y, int size, color col, string text) {
   lbl.Create(m_chartId, name, 0, x, y);
   lbl.Color(col);
   lbl.FontSize(size);
   lbl.Description(text);
   lbl.Selectable(false);
   lbl.Font("Consolas");
}

void CVisuals::Update(SFEATResult &feat, string state) {
   // Update State
   m_lblState.Description("STATE: " + state);
   
   // Update Score with dynamic coloring
   m_lblScore.Description(StringFormat("SCORE: %.1f", feat.compositeScore));
   if(feat.compositeScore > 70) m_lblScore.Color(clrLime);
   else if(feat.compositeScore < 30) m_lblScore.Color(clrOrangeRed);
   else m_lblScore.Color(clrGold);

   ChartRedraw(m_chartId);
}

void CVisuals::SetPilot(string pilot, double confidence) {
   m_lblPilot.Description("PILOT: " + pilot);
   
   // Confidence Bar
   int bars = (int)MathRound(confidence * 10);
   string barStr = "[";
   for(int i=0; i<10; i++) barStr += (i < bars) ? "#" : "-";
   barStr += StringFormat("] %.0f%%", confidence * 100);
   
   m_lblConfidence.Description("CONFIDENCE: " + barStr);
   m_lblConfidence.Color(confidence > 0.75 ? HUD_NEURAL_COLOR : (confidence < 0.3 ? HUD_ALARM_COLOR : clrGray));
}

void CVisuals::SetVault(double balance, string status) {
   m_lblVault.Description(StringFormat("VAULT: $%.2f (%s)", balance, status));
   m_lblVault.Color(status == "LOCKED" ? clrGold : clrMediumSpringGreen);
}

void CVisuals::Clear() {
   m_bg.Delete();
   m_lblTitle.Delete();
   m_lblPilot.Delete();
   m_lblConfidence.Delete();
   m_lblState.Delete();
   m_lblScore.Delete();
   m_lblVault.Delete();
   m_lblRegime.Delete();
}

#endif
