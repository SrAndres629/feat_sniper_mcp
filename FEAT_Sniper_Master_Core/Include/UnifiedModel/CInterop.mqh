//+------------------------------------------------------------------+
//|                                                   CInterop.mqh |
//|                    MT5 <> Python Interoperability               |
//|         Upgraded for Full FEAT + EMAs + Multitemporal Export    |
//|         NOW WITH ZMQ SUPPORT (Port 5555)                        |
//+------------------------------------------------------------------+
#ifndef CINTEROP_MQH
#define CINTEROP_MQH

#include <Zmq/ZmqMsg.mqh>
#include <Zmq/Socket.mqh>
#include <Zmq/Context.mqh>

#include "CEMAs.mqh"
#include "CFEAT.mqh"
#include "CMultitemporal.mqh"

//+------------------------------------------------------------------+
//| COMPREHENSIVE FEATURE EXPORT STRUCTURE                           |
//+------------------------------------------------------------------+
struct SBarDataExport {
   // Time & Price
   datetime time;
   double open, high, low, close, volume;
   
   // EMA Layer Metrics
   double microComp, microSlope, microCurvature;
   double operComp, operSlope;
   double macroSlope, biasSlope;
   double layerSep12;    // micro - oper
   double layerSep23;    // oper - macro
   bool   fanBullish, fanBearish;
   
   // FEAT Form
   bool hasBOS, hasCHoCH, hasHCH, isIntentCandle;
   double curvatureScore, compressionRatio;
   
   // FEAT Space
   bool atZone;
   double proximityScore;
   string activeZoneType;
   
   // FEAT Acceleration  
   double velocity, momentum, deltaFlow;
   double rsi, macdHist, ao, ac;
   bool isInstitutional, isExhausted;
   
   // FEAT Time
   bool isKillzone, isLondonKZ, isNYKZ, isAgainstH4;
   int h4Direction;
   string activeSession;
   
   // FSM State
   string marketState;
   double compositeScore;
   
   // Multitemporal
   string dominantTrend;
   int mtfConfluence;
   bool mtfAgainstBias;
   double m5Bias, h1Bias, h4Bias, d1Bias;
};

//+------------------------------------------------------------------+
//| CINTEROP CLASS                                                    |
//+------------------------------------------------------------------+
class CInterop {
private:
   bool   m_enabled;
   Context *m_context;
   Socket  *m_socket;
   bool     m_zmqInitialised;
   string   m_dataPath;
   string   m_filename;

public:
   CInterop();
   ~CInterop();
   
   void SetEnabled(bool e) { m_enabled = e; }
   void SetDataPath(string path) { m_dataPath = path; }
   void SetFilename(string name) { m_filename = name; }
   
   // Initialize ZMQ connection
   bool InitZMQ();

   // NEW: Send features via ZMQ
   bool SendFeaturesZMQ(SBarDataExport &data, string symbol);
   
   // Helper to format JSON string (MQL5 doesn't have native JSON builder yet)
   string BuildJson(SBarDataExport &data, string symbol);
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CInterop::CInterop() : m_enabled(true), m_zmqInitialised(false) {
   m_context = NULL;
   m_socket = NULL;
   m_dataPath = "";
   m_filename = "";
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CInterop::~CInterop() {
   if(m_socket != NULL) { delete m_socket; }
   if(m_context != NULL) { delete m_context; }
}

//+------------------------------------------------------------------+
//| Initialize ZMQ                                                    |
//+------------------------------------------------------------------+
bool CInterop::InitZMQ() {
   if(m_zmqInitialised) return true;
   
   // Create ZMQ Context
   if(m_context == NULL) {
      m_context = new Context();
   }
   
   if(m_context == NULL) {
      Print("CInterop: Failed to create ZMQ Context");
      return false;
   }

   // Create PUB Socket (Streaming to Python)
   // The instruction's snippet for this part was syntactically incorrect.
   // Reconstructing based on intent to add a Print statement.
   if(m_socket == NULL) {
      Print("[ZMQ] Initializing for ", _Symbol); // Added Print statement
      m_socket = new Socket(m_context, ZMQ_PUB);
   }
   
   if(m_socket == NULL) {
      Print("CInterop: Failed to create ZMQ Socket");
      return false;
   }

   // Connect to Docker brain
   string addr = "tcp://localhost:5555";
   if(m_socket != NULL && !(*m_socket).connect(addr)) { 
      Print("CInterop: Failed to connect to ", addr);
      return false;
   }
   
   // Set Linger to 0 to avoid blocking on close
   if(m_socket != NULL) (*m_socket).setLinger(0);

   Print("CInterop: ZMQ Bridge Connected on Port 5555");
   m_zmqInitialised = true;
   return true;
}

//+------------------------------------------------------------------+
//| Send Features via ZMQ                                             |
//+------------------------------------------------------------------+
bool CInterop::SendFeaturesZMQ(SBarDataExport &data, string symbol) {
   if(!m_enabled) return false;
   if(!m_zmqInitialised) {
      if(!InitZMQ()) return false;
   }

   string json = BuildJson(data, symbol);
   
   ZmqMsg request(json);
   // Non-blocking send
   // The instruction's snippet for this part was syntactically incorrect.
   // Reconstructing based on intent to add a Print statement and use the existing ZmqMsg.
   bool success = false;
   if(m_socket != NULL) {
      success = (*m_socket).send(request, true); // Use the ZmqMsg 'request'
   }

   if(success) {
      // Print("[ZMQ] Data sent for ", symbol); // Silent by default, uncomment for heavy debug
   } else {
      Print("[ZMQ] ERROR: Send failed for ", symbol);
   }
   
   return success;
}

//+------------------------------------------------------------------+
//| Manual JSON Builder (Fast & Lightweight)                         |
//+------------------------------------------------------------------+
string CInterop::BuildJson(SBarDataExport &data, string symbol) {
   string json = "{";
   
   // Header
   json += "\"type\":\"MKT_SNAPSHOT\",";
   json += "\"symbol\":\"" + symbol + "\",";
   json += "\"time\":\"" + TimeToString(data.time) + "\",";
   
   // Price
   json += "\"price\":{";
   json += "\"o\":" + DoubleToString(data.open, 5) + ",";
   json += "\"h\":" + DoubleToString(data.high, 5) + ",";
   json += "\"l\":" + DoubleToString(data.low, 5) + ",";
   json += "\"c\":" + DoubleToString(data.close, 5) + ",";
   json += "\"v\":" + DoubleToString(data.volume, 2);
   json += "},";
   
   // FEAT Metrics
   json += "\"feat\":{";
   json += "\"velocity\":" + DoubleToString(data.velocity, 4) + ",";
   json += "\"momentum\":" + DoubleToString(data.momentum, 4) + ",";
   json += "\"deltaFlow\":" + DoubleToString(data.deltaFlow, 4) + ",";
   json += "\"rsi\":" + DoubleToString(data.rsi, 2) + ",";
   json += "\"macdHist\":" + DoubleToString(data.macdHist, 6) + ",";
   json += "\"isInstitutional\":" + (data.isInstitutional ? "true" : "false") + ",";
   json += "\"isExhausted\":" + (data.isExhausted ? "true" : "false");
   json += "},";
   
   // State
   json += "\"state\":\"" + data.marketState + "\",";
   json += "\"score\":" + DoubleToString(data.compositeScore, 2);
   
   json += "}";
   return json;
}

#endif // CINTEROP_MQH
