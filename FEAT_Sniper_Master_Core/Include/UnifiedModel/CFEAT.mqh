//+------------------------------------------------------------------+
//|                                                      CFEAT.mqh |
//|             FEAT Layer - Senior Quantitative Engineer           |
//|      Gas (Micro), Water (Oper), Wall (Struct), Bedrock (Macro)  |
//+------------------------------------------------------------------+
#ifndef CFEAT_MQH
#define CFEAT_MQH

#include "CEMAs.mqh"
#include "CLiquidity.mqh"

//+------------------------------------------------------------------+
//| 1. ENGINEER ROLE - PHYSICS & FORM                                |
//+------------------------------------------------------------------+
enum ENUM_MARKET_PHASE { PHASE_ACCUMULATION, PHASE_MANIPULATION, PHASE_DISTRIBUTION, PHASE_EXPANSION };
enum ENUM_PHYSICS_INTERACTION { PHYS_NONE, PHYS_BOUNCE_WALL, PHYS_PIERCE_RIVER, PHYS_MAGNET_PULL };

struct SEngineerReport {
   string               trend;            
   string               pressure;         
   string               rsiState;         
   ENUM_MARKET_PHASE    phase;
   string               energyState;      
   string               structure;        
   string               criticalPath;     
   double               targetPrice;
   string               engineerOrder;    
   string               diagnosis;        
   double               avgCurvature;
   double               compressionRatio;
   double               momentumVal;
};

//+------------------------------------------------------------------+
//| 2. TACTICIAN ROLE - TIME & SPACE                                 |
//+------------------------------------------------------------------+
enum ENUM_SESSION_STATE { STATE_KZ_LONDON, STATE_KZ_NY, STATE_LUNCH, STATE_ASIA, STATE_DEAD_ZONE };
enum ENUM_ZONE_QUALITY { QUALITY_HIGH, QUALITY_MEDIUM, QUALITY_LOW };

struct STacticianReport {
   string               currentTime;
   ENUM_SESSION_STATE   sessionState;
   bool                 isOperableTime;
   string               poiDetected;      
   string               locationRelative; 
   ENUM_ZONE_QUALITY    zoneQuality;
   string               action;           
   string               reason;
   double               distToZone;
   double               layerSeparation;
};

//+------------------------------------------------------------------+
//| 3. SNIPER ROLE - EXECUTION                                       |
//+------------------------------------------------------------------+
struct SSniperOrder {
   string               action;     
   double               entryPrice;
   double               slPrice;
   double               tpPrice;
   double               riskReward;
   double               lotSize;    
};

struct SSniperReport {
   string               decision;         
   double               confidence;       
   string               finalReason;
   SSniperOrder         order;
   double               velocity;         
   bool                 isInstitutional;
   double               deltaFlow;
   double               rsi;
   double               macdHist;
   double               ao;
   double               ac;
   bool                 isExhausted;
};

// Global Structs for Compatibility
struct STimeMetrics { bool isKillzone; string activeSession; };
struct SAccelMetrics { double velocity; bool isInstitutional; double momentum; bool isExhausted; double deltaFlow; double rsi; double macdHist; double ao; double ac; };
struct SFormMetrics { bool hasBOS; bool hasCHoCH; bool hasHCH; bool isIntentCandle; double curvatureScore; double compressionRatio; };
struct SSpaceMetrics { bool atZone; double proximityScore; string activeZoneType; double fastMediumGap; double mediumSlowGap; };
struct SFEATResult { SFormMetrics form; SSpaceMetrics space; SAccelMetrics accel; STimeMetrics time; double compositeScore; };

//+------------------------------------------------------------------+
//| MAIN FEAT CLASS                                                  |
//+------------------------------------------------------------------+
class CFEAT {
private:
   CEMAs*           m_ptrEmas;
   CLiquidity*      m_ptrLiq;
   SEngineerReport  m_engineer;
   STacticianReport m_tactician;
   SSniperReport    m_sniper;
   string           m_symbol;
   ENUM_TIMEFRAMES  m_timeframe;
   double           m_tickSize;
   int              m_rsiHandle;
   int              m_macdHandle;
   int              m_aoHandle;
   double           m_prevMomentum;
   double           m_prevVelocity;
   double           m_atr;
   
   void             RunEngineer(double close);
   void             RunTactician(datetime time, double close);
   void             RunSniper(double open, double close, double high, double low, double volume);
   bool             InitIndicators();
   
public:
   CFEAT();
   ~CFEAT();
   bool Init(string symbol, ENUM_TIMEFRAMES tf);
   void Deinit();
   void SetEMAs(CEMAs* e) { m_ptrEmas = e; }
   void SetLiquidity(CLiquidity* l) { m_ptrLiq = l; }
   bool Calculate(ENUM_TIMEFRAMES tf, datetime t, double open, double high, double low, double close, double volume);
   SEngineerReport  GetEngineer() const { return m_engineer; }
   STacticianReport GetTactician() const { return m_tactician; }
   SSniperReport    GetSniper() const { return m_sniper; }
   double           GetCompositeScore() const { return m_sniper.confidence; }
   SFEATResult      GetResult();
};

CFEAT::CFEAT() : m_ptrEmas(NULL), m_ptrLiq(NULL), m_prevMomentum(0), m_prevVelocity(0), m_atr(0) {
   ZeroMemory(m_engineer);
   ZeroMemory(m_tactician);
   ZeroMemory(m_sniper);
   m_rsiHandle = INVALID_HANDLE;
   m_macdHandle = INVALID_HANDLE;
   m_aoHandle = INVALID_HANDLE;
}

CFEAT::~CFEAT() { Deinit(); }

bool CFEAT::Init(string symbol, ENUM_TIMEFRAMES tf) {
   m_symbol = symbol;
   m_timeframe = tf;
   m_tickSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   return InitIndicators();
}

bool CFEAT::InitIndicators() {
   m_rsiHandle = iRSI(m_symbol, m_timeframe, 14, PRICE_CLOSE);
   m_macdHandle = iMACD(m_symbol, m_timeframe, 12, 26, 9, PRICE_CLOSE);
   m_aoHandle = iAO(m_symbol, m_timeframe);
   return (m_rsiHandle != INVALID_HANDLE && m_macdHandle != INVALID_HANDLE && m_aoHandle != INVALID_HANDLE);
}

void CFEAT::Deinit() {
   if(m_rsiHandle != INVALID_HANDLE) { IndicatorRelease(m_rsiHandle); m_rsiHandle = INVALID_HANDLE; }
   if(m_macdHandle != INVALID_HANDLE) { IndicatorRelease(m_macdHandle); m_macdHandle = INVALID_HANDLE; }
   if(m_aoHandle != INVALID_HANDLE) { IndicatorRelease(m_aoHandle); m_aoHandle = INVALID_HANDLE; }
}

bool CFEAT::Calculate(ENUM_TIMEFRAMES tf, datetime t, double open, double high, double low, double close, double volume) {
   if(m_ptrEmas == NULL || !m_ptrEmas.IsReady()) return false;
   m_atr = m_ptrEmas.GetATR();
   RunEngineer(close);
   RunTactician(t, close);
   RunSniper(open, close, high, low, volume);
   return true;
}

SFEATResult CFEAT::GetResult() {
   SFEATResult r;
   r.compositeScore = m_sniper.confidence;
   r.time.isKillzone = m_tactician.isOperableTime;
   r.time.activeSession = EnumToString(m_tactician.sessionState);
   r.accel.velocity = m_sniper.velocity;
   r.accel.isInstitutional = m_sniper.isInstitutional;
   r.accel.momentum = m_engineer.momentumVal;
   r.accel.deltaFlow = m_sniper.deltaFlow;
   r.accel.rsi = m_sniper.rsi;
   r.accel.macdHist = m_sniper.macdHist;
   r.accel.ao = m_sniper.ao;
   r.accel.ac = m_sniper.ac;
   r.accel.isExhausted = m_sniper.isExhausted;
   r.form.hasBOS = (StringFind(m_engineer.structure, "BOS") >= 0);
   r.form.hasCHoCH = (StringFind(m_engineer.structure, "CHOCH") >= 0);
   r.form.isIntentCandle = m_sniper.isInstitutional;
   r.form.curvatureScore = m_engineer.avgCurvature;
   r.form.compressionRatio = m_engineer.compressionRatio;
   r.space.atZone = (m_tactician.poiDetected != "NONE");
   r.space.activeZoneType = m_tactician.poiDetected;
   r.space.proximityScore = (r.space.atZone ? 1.0 : 0.0);
   r.space.fastMediumGap = m_tactician.layerSeparation;
   return r;
}

void CFEAT::RunEngineer(double close) {
   double gas = m_ptrEmas.GetWind();       
   double water = m_ptrEmas.GetRiver();    
   double wall = m_ptrEmas.GetWall();      
   double bedrock = m_ptrEmas.GetMagnet(); 
   double wallSlope = m_ptrEmas.GetWallSlope();
   double gasSlope = m_ptrEmas.GetWindSlope();
   if(wallSlope > 0.05) {
      m_engineer.trend = "ALCISTA (PISO)";
      m_engineer.pressure = (close > wall) ? "FLUJO LIBRE" : "FRICCION EN MURO";
   } else if(wallSlope < -0.05) {
      m_engineer.trend = "BAJISTA (TECHO)";
      m_engineer.pressure = (close < wall) ? "FLUJO LIBRE" : "FRICCION EN MURO";
   } else {
      m_engineer.trend = "LATERAL";
      m_engineer.pressure = "RUIDO TERMICO";
   }
   m_engineer.momentumVal = gasSlope;
   double rsi = 50.0;
   double buff[1];
   if(m_rsiHandle != INVALID_HANDLE && CopyBuffer(m_rsiHandle, 0, 0, 1, buff)>0) rsi = buff[0];
   m_sniper.rsi = rsi;
   if(rsi > 70) m_engineer.rsiState = (rsi > 80) ? "CRITICO > 80" : "SOBRE-COMPRA";
   else if(rsi < 30) m_engineer.rsiState = (rsi < 20) ? "CRITICO < 20" : "SOBRE-VENTA";
   else m_engineer.rsiState = "NEUTRO";
   SFanMetrics fan = m_ptrEmas.GetFanMetrics();
   m_engineer.compressionRatio = fan.compression;
   m_engineer.energyState = (fan.compression > 0.8) ? "COMPRESION (SQZ)" : "EXPANSION (ABANICO)";
   double target = 0;
   string targetName = "VACIO";
   if(m_ptrLiq != NULL) {
       SLiquidityContext ctx = m_ptrLiq.GetContext();
       if(m_engineer.trend == "ALCISTA (PISO)") {
           if(ctx.nearestAbove.price > 0 && ctx.nearestAbove.price > close) {
               target = ctx.nearestAbove.price;
               targetName = ctx.nearestAbove.label;
           } else {
               target = close + (m_atr * 50); 
               targetName = "EXPANSION";
           }
       } else if (m_engineer.trend == "BAJISTA (TECHO)") {
           if(ctx.nearestBelow.price > 0 && ctx.nearestBelow.price < close) {
               target = ctx.nearestBelow.price;
               targetName = ctx.nearestBelow.label;
           } else {
               target = close - (m_atr * 50);
               targetName = "EXPANSION";
           }
       }
   }
   m_engineer.targetPrice = target;
   if(target > 0) m_engineer.criticalPath = StringFormat("%.5f -> %s (%.5f)", close, targetName, target);
   else m_engineer.criticalPath = "ESPERANDO ESTRUCTURA";
   m_engineer.engineerOrder = "ESPERAR";
   if(m_engineer.trend == "ALCISTA (PISO)" && m_engineer.pressure == "FLUJO LIBRE") {
       m_engineer.diagnosis = "El precio debe buscar liquidez superior.";
       m_engineer.engineerOrder = "COMPRAR EN PULLBACK";
   } else if(m_engineer.trend == "BAJISTA (TECHO)" && m_engineer.pressure == "FLUJO LIBRE") {
       m_engineer.diagnosis = "Trayectoria obligatoria hacia zona inferior.";
       m_engineer.engineerOrder = "VENDER EN PULLBACK";
   } else {
       m_engineer.diagnosis = "Acumulacion de energia. No operar ruido.";
       m_engineer.engineerOrder = "OBSERVAR";
   }
}

void CFEAT::RunTactician(datetime time, double close) {
   MqlDateTime dt;
   TimeToStruct(time, dt);
   m_tactician.currentTime = StringFormat("%02d:%02d", dt.hour, dt.min);
   m_tactician.isOperableTime = false;
   if(dt.hour >= 2 && dt.hour < 5) { m_tactician.sessionState = STATE_KZ_LONDON; m_tactician.isOperableTime = true; }
   else if(dt.hour >= 7 && dt.hour < 11) { m_tactician.sessionState = STATE_KZ_NY; m_tactician.isOperableTime = true; }
   else if(dt.hour >= 0 && dt.hour < 9) m_tactician.sessionState = STATE_ASIA; 
   else if(dt.hour >= 12 && dt.hour < 14) m_tactician.sessionState = STATE_LUNCH;
   else m_tactician.sessionState = STATE_DEAD_ZONE;
   m_tactician.poiDetected = "NONE";
   m_tactician.zoneQuality = QUALITY_LOW;
   if(m_ptrLiq != NULL) {
      double minDist = DBL_MAX;
      int zoneIdx = -1;
      for(int i=0; i<m_ptrLiq.GetZoneCount(); i++) {
         SInstitutionalZone z = m_ptrLiq.GetZone(i);
         if(z.mitigated) continue;
         double dist = 0;
         if(close > z.top) dist = close - z.top; else if(close < z.bottom) dist = z.bottom - close;
         if(dist < minDist) { minDist = dist; zoneIdx = i; }
      }
      m_tactician.distToZone = (minDist < DBL_MAX && m_atr > 0) ? minDist/m_atr : 999;
      if(zoneIdx != -1 && minDist < m_atr * 2.0) { 
         SInstitutionalZone z = m_ptrLiq.GetZone(zoneIdx);
         if(z.type == ZONE_FVG) m_tactician.poiDetected = "FVG";
         else if(z.type == ZONE_OB) m_tactician.poiDetected = "OB";
         else if(z.type == ZONE_CONFLUENCE) { m_tactician.poiDetected = "CONFLUENCE"; m_tactician.zoneQuality = QUALITY_HIGH; }
         m_tactician.poiDetected += (z.isBullish ? "_BULL" : "_BEAR");
      }
   }
   if(m_tactician.isOperableTime && m_tactician.poiDetected != "NONE") {
       m_tactician.action = "BUSCAR_GATILLO";
       m_tactician.reason = "Confluencia Espacio-Tiempo Confirmada.";
   } else if(!m_tactician.isOperableTime) {
       m_tactician.action = "ABORTAR_HORARIO";
       m_tactician.reason = "Fuera de Killzone.";
   } else {
       m_tactician.action = "ESPERAR_ZONA";
       m_tactician.reason = "Precio en Tierra de Nadie.";
   }
}

void CFEAT::RunSniper(double open, double close, double high, double low, double volume) {
   double body = MathAbs(close - open);
   double rsi = m_sniper.rsi;
   m_sniper.velocity = (m_atr > 0) ? (body / m_atr) : 0;
   m_sniper.isInstitutional = (m_sniper.velocity > 1.1); 
   double direction = (close > open) ? 1.0 : -1.0;
   m_sniper.deltaFlow = direction * body * (volume / 1000.0) / (m_atr > 0 ? m_atr : 0.0001);
   m_sniper.isExhausted = (m_prevVelocity > 1.5 && m_sniper.velocity < 0.3 * m_prevVelocity);
   m_prevVelocity = m_sniper.velocity;
   m_sniper.decision = "ESPERAR";
   m_sniper.confidence = 0;
   bool tacticianOk = (m_tactician.action == "BUSCAR_GATILLO");
   bool engineerBuy = (m_engineer.engineerOrder == "COMPRAR EN PULLBACK");
   bool engineerSell = (m_engineer.engineerOrder == "VENDER EN PULLBACK");
   bool setupBuy = (engineerBuy && StringFind(m_tactician.poiDetected, "BULL") >= 0 && rsi < 75);
   bool setupSell = (engineerSell && StringFind(m_tactician.poiDetected, "BEAR") >= 0 && rsi > 25);
   if(tacticianOk) {
      if(setupBuy && close > open && m_sniper.isInstitutional && m_sniper.deltaFlow > 0.5) {
          m_sniper.decision = "DISPARAR";
          m_sniper.order.action = "BUY";
          m_sniper.finalReason = "Aceleracion + Flujo + Zona Demanda.";
          double conf = 70.0;
          if(m_tactician.zoneQuality == QUALITY_HIGH) conf += 15.0;
          if(m_sniper.deltaFlow > 2.0) conf += 10.0;
          m_sniper.confidence = MathMin(99.0, conf);
          m_sniper.order.entryPrice = close;
          double zoneLow = (m_ptrLiq != NULL) ? m_ptrLiq.GetContext().nearestBelow.price : low;
          m_sniper.order.slPrice = MathMin(low - (m_atr * 0.5), zoneLow - (m_atr * 0.2));
          double risk = close - m_sniper.order.slPrice;
          if(risk < m_atr * 1.0) { m_sniper.order.slPrice = close - (m_atr * 1.0); risk = m_atr * 1.0; }
          m_sniper.order.tpPrice = close + (risk * 2.5);
          m_sniper.order.riskReward = 2.5;
      } else if(setupSell && close < open && m_sniper.isInstitutional && m_sniper.deltaFlow < -0.5) {
          m_sniper.decision = "DISPARAR";
          m_sniper.order.action = "SELL";
          m_sniper.finalReason = "Aceleracion + Flujo + Zona Oferta.";
          double conf = 70.0;
          if(m_tactician.zoneQuality == QUALITY_HIGH) conf += 15.0;
          if(m_sniper.deltaFlow < -2.0) conf += 10.0;
          m_sniper.confidence = MathMin(99.0, conf);
          m_sniper.order.entryPrice = close;
          double zoneHigh = (m_ptrLiq != NULL) ? m_ptrLiq.GetContext().nearestAbove.price : high;
          m_sniper.order.slPrice = MathMax(high + (m_atr * 0.5), zoneHigh + (m_atr * 0.2));
          double risk = m_sniper.order.slPrice - close;
          if(risk < m_atr * 1.0) { m_sniper.order.slPrice = close + (m_atr * 1.0); risk = m_atr * 1.0; }
          m_sniper.order.tpPrice = close - (risk * 2.5);
          m_sniper.order.riskReward = 2.5;
      } else {
          if(m_sniper.isExhausted) m_sniper.finalReason = "Agotamiento Termico detectado.";
          else if(rsi > 75 || rsi < 25) m_sniper.finalReason = "Sobre-Extension de Energia (RSI).";
          else m_sniper.finalReason = "Esperando Inyeccion de Gas (Volumen).";
          m_sniper.confidence = 40.0;
      }
   } else {
      m_sniper.decision = "ABORTAR";
      m_sniper.finalReason = m_tactician.reason;
      m_sniper.confidence = 10.0;
   }
}

#endif
