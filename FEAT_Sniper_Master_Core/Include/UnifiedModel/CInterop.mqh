//+------------------------------------------------------------------+
//|                                                   CInterop.mqh |
//|                    MT5 <-> Python Interoperability              |
//|            Load/Save Calibration Data via JSON/CSV              |
//+------------------------------------------------------------------+
#ifndef CINTEROP_MQH
#define CINTEROP_MQH

//+------------------------------------------------------------------+
//| CALIBRATION DATA STRUCTURE                                       |
//+------------------------------------------------------------------+
struct SCalibrationData {
   // Effort/Result Percentiles
   double   effortP20, effortP50, effortP80;
   double   resultP20, resultP50, resultP80;
   
   // FSM Thresholds
   double   accumulationCompression;
   double   expansionSlope;
   double   distributionMomentum;
   double   resetSpeed;
   double   hysteresisMargin;
   int      minBarsInState;
   
   // FEAT Thresholds
   double   curvatureThreshold;
   double   compressionThreshold;
   double   accelThreshold;
   double   gapThreshold;
   
   // Metadata
   string   symbol;
   string   timeframe;
   datetime calibrationTime;
   string   version;
   
   bool     isValid;
};

//+------------------------------------------------------------------+
//| CINTEROP CLASS                                                   |
//+------------------------------------------------------------------+
class CInterop {
private:
   string            m_dataPath;
   SCalibrationData  m_calibration;
   bool              m_enabled;
   
   // Parsing helpers
   bool              ParseKeyValue(string line, string &key, string &value);
   double            SafeStringToDouble(string str);
   int               SafeStringToInt(string str);
   
public:
                     CInterop();
                    ~CInterop();
   
   // Configuration
   void              SetDataPath(string path) { m_dataPath = path; }
   void              SetEnabled(bool enabled) { m_enabled = enabled; }
   bool              IsEnabled() const { return m_enabled; }
   
   // Load calibration from Python output
   bool              LoadCalibration(string filename);
   bool              LoadCalibrationAuto(string symbol, ENUM_TIMEFRAMES tf);
   
   // Append single bar data for real-time collection
   bool              AppendBarData(string filename, datetime time, double open, double high, double low, double close,
                                  double effort, double result, double compression, double slope, double speed, 
                                  double confidence, string state);
   
   // Getters
   SCalibrationData  GetCalibration() const { return m_calibration; }
   bool              HasValidCalibration() const { return m_calibration.isValid; }
   
   // Export thresholds for FSM
   bool              ExportToFSMThresholds(SFSMThresholds &thresholds);
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CInterop::CInterop() {
   m_dataPath = "UnifiedModel\\"; // Relative to Common/Files
   m_enabled = false;
   ZeroMemory(m_calibration);
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CInterop::~CInterop() {
}

//+------------------------------------------------------------------+
//| Load Calibration from File                                       |
//| Format: key=value (one per line)                                 |
//+------------------------------------------------------------------+
bool CInterop::LoadCalibration(string filename) {
   if(!m_enabled) return false;
   
   // Try Common folder first (Bridge deployment target)
   int flags = FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON;
   int handle = FileOpen(filename, flags);
   
   // Fallback to local folder
   if(handle == INVALID_HANDLE) {
       handle = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI);
   }
   
   if(handle == INVALID_HANDLE) {
      // Quiet fail is okay, might not exist yet
      return false;
   }
   
   ZeroMemory(m_calibration);
   
   while(!FileIsEnding(handle)) {
      string line = FileReadString(handle);
      if(StringLen(line) == 0) continue;
      
      string key, value;
      if(!ParseKeyValue(line, key, value)) continue;
      
      // Percentiles
      if(key == "effortP20") m_calibration.effortP20 = SafeStringToDouble(value);
      else if(key == "effortP50") m_calibration.effortP50 = SafeStringToDouble(value);
      else if(key == "effortP80") m_calibration.effortP80 = SafeStringToDouble(value);
      else if(key == "resultP20") m_calibration.resultP20 = SafeStringToDouble(value);
      else if(key == "resultP50") m_calibration.resultP50 = SafeStringToDouble(value);
      else if(key == "resultP80") m_calibration.resultP80 = SafeStringToDouble(value);
      
      // FSM Thresholds
      else if(key == "accumulationCompression") m_calibration.accumulationCompression = SafeStringToDouble(value);
      else if(key == "expansionSlope") m_calibration.expansionSlope = SafeStringToDouble(value);
      else if(key == "distributionMomentum") m_calibration.distributionMomentum = SafeStringToDouble(value);
      else if(key == "resetSpeed") m_calibration.resetSpeed = SafeStringToDouble(value);
      else if(key == "hysteresisMargin") m_calibration.hysteresisMargin = SafeStringToDouble(value);
      else if(key == "minBarsInState") m_calibration.minBarsInState = SafeStringToInt(value);
      
      // FEAT Thresholds
      else if(key == "curvatureThreshold") m_calibration.curvatureThreshold = SafeStringToDouble(value);
      else if(key == "compressionThreshold") m_calibration.compressionThreshold = SafeStringToDouble(value);
      else if(key == "accelThreshold") m_calibration.accelThreshold = SafeStringToDouble(value);
      else if(key == "gapThreshold") m_calibration.gapThreshold = SafeStringToDouble(value);
      
      // Metadata
      else if(key == "symbol") m_calibration.symbol = value;
      else if(key == "timeframe") m_calibration.timeframe = value;
      else if(key == "version") m_calibration.version = value;
   }
   
   FileClose(handle);
   m_calibration.calibrationTime = TimeCurrent();
   m_calibration.isValid = true;
   
   Print("[CInterop] Loaded calibration: ", filename, " Version: ", m_calibration.version);
   return true;
}

//+------------------------------------------------------------------+
//| Auto-load Calibration Based on Symbol/TF                         |
//+------------------------------------------------------------------+
bool CInterop::LoadCalibrationAuto(string symbol, ENUM_TIMEFRAMES tf) {
   string tfStr = "";
   switch(tf) {
      case PERIOD_M1:  tfStr = "M1"; break;
      case PERIOD_M5:  tfStr = "M5"; break;
      case PERIOD_M15: tfStr = "M15"; break;
      case PERIOD_M30: tfStr = "M30"; break;
      case PERIOD_H1:  tfStr = "H1"; break;
      case PERIOD_H4:  tfStr = "H4"; break;
      case PERIOD_D1:  tfStr = "D1"; break;
      default: tfStr = "H1";
   }
   
   // Try optuna optimized file first
   if(LoadCalibration("optuna_calibration.txt")) return true;
   
   // Fallback to specific file
   string filename = "calibration_" + symbol + "_" + tfStr + ".txt";
   return LoadCalibration(filename);
}



//+------------------------------------------------------------------+
//| Append Single Bar Data (Full Schema)                             |
//+------------------------------------------------------------------+
bool CInterop::AppendBarData(string filename, datetime time, double open, double high, double low, double close,
                            double effort, double result, double compression, double slope, double speed, 
                            double confidence, string state) {
   if(!m_enabled) return false;
   
   // Use Common folder for bridge visibility
   int flags = FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON;
   int handle = FileOpen(filename, flags);
   
   if(handle == INVALID_HANDLE) {
      // Try to create if doesn't exist
      handle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON);
      if(handle != INVALID_HANDLE) {
         // Write Header
         FileWrite(handle, "time,open,high,low,close,effort,result,compression,slope,speed,confidence,state");
      }
   }
   
   if(handle == INVALID_HANDLE) {
       Print("[CInterop] Error opening/creating export file: ", filename);
       return false;
   }
   
   // Seek to end
   FileSeek(handle, 0, SEEK_END);
   
   string timeStr = TimeToString(time, TIME_DATE|TIME_MINUTES);
   
   FileWrite(handle, timeStr + "," + 
                     DoubleToString(open, 5) + "," +
                     DoubleToString(high, 5) + "," +
                     DoubleToString(low, 5) + "," +
                     DoubleToString(close, 5) + "," +
                     DoubleToString(effort, 6) + "," +
                     DoubleToString(result, 6) + "," +
                     DoubleToString(compression, 6) + "," +
                     DoubleToString(slope, 6) + "," +
                     DoubleToString(speed, 6) + "," +
                     DoubleToString(confidence, 2)+ "," +
                     state);
                     
   FileClose(handle);
   return true;
}

//+------------------------------------------------------------------+
//| Export to FSM Thresholds Structure                               |
//+------------------------------------------------------------------+
bool CInterop::ExportToFSMThresholds(SFSMThresholds &thresholds) {
   if(!m_calibration.isValid) return false;
   
   thresholds.effortP20 = m_calibration.effortP20;
   thresholds.effortP50 = m_calibration.effortP50;
   thresholds.effortP80 = m_calibration.effortP80;
   thresholds.resultP20 = m_calibration.resultP20;
   thresholds.resultP50 = m_calibration.resultP50;
   thresholds.resultP80 = m_calibration.resultP80;
   
   thresholds.accumulationCompression = m_calibration.accumulationCompression;
   thresholds.expansionSlope = m_calibration.expansionSlope;
   thresholds.distributionMomentum = m_calibration.distributionMomentum;
   thresholds.resetSpeed = m_calibration.resetSpeed;
   thresholds.hysteresisMargin = m_calibration.hysteresisMargin;
   thresholds.minBarsInState = m_calibration.minBarsInState;
   
   return true;
}

//+------------------------------------------------------------------+
//| Parse Key=Value Line                                             |
//+------------------------------------------------------------------+
bool CInterop::ParseKeyValue(string line, string &key, string &value) {
   int pos = StringFind(line, "=");
   if(pos <= 0) return false;
   
   key = StringSubstr(line, 0, pos);
   value = StringSubstr(line, pos + 1);
   
   StringTrimLeft(key);
   StringTrimRight(key);
   StringTrimLeft(value);
   StringTrimRight(value);
   
   return (StringLen(key) > 0 && StringLen(value) > 0);
}

//+------------------------------------------------------------------+
//| Safe String to Double                                            |
//+------------------------------------------------------------------+
double CInterop::SafeStringToDouble(string str) {
   StringReplace(str, ",", ".");
   return StringToDouble(str);
}

//+------------------------------------------------------------------+
//| Safe String to Int                                               |
//+------------------------------------------------------------------+
int CInterop::SafeStringToInt(string str) {
   return (int)StringToInteger(str);
}

#endif // CINTEROP_MQH
