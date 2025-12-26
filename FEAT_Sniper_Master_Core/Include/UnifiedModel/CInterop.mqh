//+------------------------------------------------------------------+
//|                                                   CInterop.mqh |
//|                    MT5 <> Python Interoperability               |
//|         Upgraded for Full FEAT + EMAs + Multitemporal Export    |
//+------------------------------------------------------------------+
#ifndef CINTEROP_MQH
#define CINTEROP_MQH

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
   string m_path;
   string m_filename;
   bool   m_headerWritten;

public:
   CInterop() : m_enabled(true), m_path(""), m_filename("UnifiedModel_Features.csv"), m_headerWritten(false) {}
   void SetEnabled(bool e) { m_enabled = e; }
   void SetDataPath(string p) { m_path = p; }
   void SetFilename(string f) { m_filename = f; }

   bool AppendBarData(string filename, datetime time, double open, double high, double low, double close,
                      double effort, double result, double compression, double slope, double speed, 
                      double confidence, string state);
   
   // NEW: Full feature export
   bool ExportFeatures(SBarDataExport &data);
   
   // Write CSV header
   bool WriteHeader(int handle);
};

//+------------------------------------------------------------------+
//| Legacy AppendBarData (backward compatibility)                     |
//+------------------------------------------------------------------+
bool CInterop::AppendBarData(string filename, datetime time, double o, double h, double l, double c,
                            double eff, double res, double comp, double slp, double spd, 
                            double conf, string state) {
   if(!m_enabled) return false;
   int handle = FileOpen(filename, FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON);
   if(handle == INVALID_HANDLE) {
      handle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON);
      if(handle != INVALID_HANDLE)
         FileWrite(handle, "time,open,high,low,close,effort,result,microComp,operSlope,layerSep,state");
   }
   if(handle == INVALID_HANDLE) return false;
   FileSeek(handle, 0, SEEK_END);
   FileWrite(handle, TimeToString(time) + "," + DoubleToString(o, 5) + "," + DoubleToString(h, 5) + "," +
                     DoubleToString(l, 5) + "," + DoubleToString(c, 5) + "," + DoubleToString(eff, 5) + "," +
                     DoubleToString(res, 5) + "," + DoubleToString(comp, 5) + "," + DoubleToString(slp, 5) + "," +
                     DoubleToString(spd, 5) + "," + state);
   FileClose(handle);
   return true;
}

//+------------------------------------------------------------------+
//| Write CSV Header                                                  |
//+------------------------------------------------------------------+
bool CInterop::WriteHeader(int handle) {
   if(handle == INVALID_HANDLE) return false;
   FileWrite(handle, 
      "time,open,high,low,close,volume," +
      "microComp,microSlope,microCurv,operComp,operSlope,macroSlope,biasSlope,layerSep12,layerSep23,fanBull,fanBear," +
      "hasBOS,hasCHoCH,hasHCH,isIntent,curvScore,compRatio," +
      "atZone,proxScore,zoneType," +
      "velocity,momentum,deltaFlow,rsi,macdHist,ao,ac,isInst,isExhaust," +
      "isKZ,isLondonKZ,isNYKZ,isAgainstH4,h4Dir,session," +
      "state,score," +
      "domTrend,mtfConf,mtfAgainst,m5Bias,h1Bias,h4Bias,d1Bias"
   );
   return true;
}

//+------------------------------------------------------------------+
//| Full Feature Export                                               |
//+------------------------------------------------------------------+
bool CInterop::ExportFeatures(SBarDataExport &data) {
   if(!m_enabled) return false;
   
   string fullPath = m_path + m_filename;
   int handle = FileOpen(fullPath, FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON);
   
   if(handle == INVALID_HANDLE) {
      handle = FileOpen(fullPath, FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON);
      if(handle != INVALID_HANDLE) {
         WriteHeader(handle);
         m_headerWritten = true;
      }
   }
   
   if(handle == INVALID_HANDLE) return false;
   FileSeek(handle, 0, SEEK_END);
   
   // Build CSV row
   string row = TimeToString(data.time) + "," +
                DoubleToString(data.open, 5) + "," +
                DoubleToString(data.high, 5) + "," +
                DoubleToString(data.low, 5) + "," +
                DoubleToString(data.close, 5) + "," +
                DoubleToString(data.volume, 2) + "," +
                // EMA Layers
                DoubleToString(data.microComp, 4) + "," +
                DoubleToString(data.microSlope, 4) + "," +
                DoubleToString(data.microCurvature, 4) + "," +
                DoubleToString(data.operComp, 4) + "," +
                DoubleToString(data.operSlope, 4) + "," +
                DoubleToString(data.macroSlope, 4) + "," +
                DoubleToString(data.biasSlope, 4) + "," +
                DoubleToString(data.layerSep12, 4) + "," +
                DoubleToString(data.layerSep23, 4) + "," +
                IntegerToString(data.fanBullish ? 1 : 0) + "," +
                IntegerToString(data.fanBearish ? 1 : 0) + "," +
                // FEAT Form
                IntegerToString(data.hasBOS ? 1 : 0) + "," +
                IntegerToString(data.hasCHoCH ? 1 : 0) + "," +
                IntegerToString(data.hasHCH ? 1 : 0) + "," +
                IntegerToString(data.isIntentCandle ? 1 : 0) + "," +
                DoubleToString(data.curvatureScore, 4) + "," +
                DoubleToString(data.compressionRatio, 4) + "," +
                // FEAT Space
                IntegerToString(data.atZone ? 1 : 0) + "," +
                DoubleToString(data.proximityScore, 4) + "," +
                data.activeZoneType + "," +
                // FEAT Acceleration
                DoubleToString(data.velocity, 4) + "," +
                DoubleToString(data.momentum, 4) + "," +
                DoubleToString(data.deltaFlow, 4) + "," +
                DoubleToString(data.rsi, 2) + "," +
                DoubleToString(data.macdHist, 6) + "," +
                DoubleToString(data.ao, 4) + "," +
                DoubleToString(data.ac, 4) + "," +
                IntegerToString(data.isInstitutional ? 1 : 0) + "," +
                IntegerToString(data.isExhausted ? 1 : 0) + "," +
                // FEAT Time
                IntegerToString(data.isKillzone ? 1 : 0) + "," +
                IntegerToString(data.isLondonKZ ? 1 : 0) + "," +
                IntegerToString(data.isNYKZ ? 1 : 0) + "," +
                IntegerToString(data.isAgainstH4 ? 1 : 0) + "," +
                IntegerToString(data.h4Direction) + "," +
                data.activeSession + "," +
                // FSM
                data.marketState + "," +
                DoubleToString(data.compositeScore, 2) + "," +
                // Multitemporal
                data.dominantTrend + "," +
                IntegerToString(data.mtfConfluence) + "," +
                IntegerToString(data.mtfAgainstBias ? 1 : 0) + "," +
                DoubleToString(data.m5Bias, 3) + "," +
                DoubleToString(data.h1Bias, 3) + "," +
                DoubleToString(data.h4Bias, 3) + "," +
                DoubleToString(data.d1Bias, 3);
   
   FileWrite(handle, row);
   FileClose(handle);
   return true;
}

#endif // CINTEROP_MQH
