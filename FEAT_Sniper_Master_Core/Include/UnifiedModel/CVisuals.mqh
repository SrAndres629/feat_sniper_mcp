//+------------------------------------------------------------------+
//|                                                     CVisuals.mqh |
//|                    Visual HUD for FEAT System                   |
//|               Minimalist Debugging Interface (Local)             |
//+------------------------------------------------------------------+
#ifndef CVISUALS_MQH
#define CVISUALS_MQH

#include <ChartObjects\ChartObjectsTxtControls.mqh>
#include "CFEAT.mqh"
#include "CFSM.mqh"

class CVisuals {
private:
   CChartObjectLabel m_lblScore;
   CChartObjectLabel m_lblState;
   CChartObjectLabel m_lblAction;
   CChartObjectLabel m_lblZone;

   string            m_symbol;
   long              m_chartId;
   int               m_yOffset;

public:
   CVisuals();
   ~CVisuals();

   void Init(string symbol, int yOffset = 20);
   void Update(SFEATResult &feat, string state);
   void Clear();
};

CVisuals::CVisuals() : m_chartId(0), m_yOffset(20) {}

CVisuals::~CVisuals() {
   Clear();
}

void CVisuals::Init(string symbol, int yOffset) {
   m_symbol = symbol;
   m_chartId = ChartID();
   m_yOffset = yOffset;

   int x = 20;
   int y = yOffset;
   int step = 20;

   m_lblScore.Create(m_chartId, "FeatScore", 0, x, y);
   m_lblScore.Color(clrWhite);
   m_lblScore.FontSize(12);
   m_lblScore.Description("Score: --");

   y += step;
   m_lblState.Create(m_chartId, "FeatState", 0, x, y);
   m_lblState.Color(clrGray);
   m_lblState.FontSize(10);
   m_lblState.Description("State: --");

   y += step;
   m_lblAction.Create(m_chartId, "FeatAction", 0, x, y);
   m_lblAction.Color(clrGray);
   m_lblAction.FontSize(10);
   m_lblAction.Description("Action: --");

   y += step;
   m_lblZone.Create(m_chartId, "FeatZone", 0, x, y);
   m_lblZone.Color(clrGray);
   m_lblZone.FontSize(10);
   m_lblZone.Description("Zone: --");
}

void CVisuals::Update(SFEATResult &feat, string state) {
   string scoreText = StringFormat("FEAT Score: %.1f", feat.compositeScore);
   m_lblScore.Description(scoreText);
   if(feat.compositeScore > 75) m_lblScore.Color(clrLime);
   else if(feat.compositeScore < 25) m_lblScore.Color(clrRed);
   else m_lblScore.Color(clrWhite);

   m_lblState.Description("FSM: " + state);

   string action = "WAIT";
   color actionColor = clrGray;

   if(feat.accel.isInstitutional) {
       action = "INSTITUTIONAL";
       actionColor = clrGold;
   }
   m_lblAction.Description("Mode: " + action);
   m_lblAction.Color(actionColor);

   string zone = feat.space.atZone ? feat.space.activeZoneType : "No Zone";
   m_lblZone.Description("Zone: " + zone);
   m_lblZone.Color(feat.space.atZone ? clrAqua : clrGray);

   ChartRedraw(m_chartId);
}

void CVisuals::Clear() {
   m_lblScore.Delete();
   m_lblState.Delete();
   m_lblAction.Delete();
   m_lblZone.Delete();
}

#endif
