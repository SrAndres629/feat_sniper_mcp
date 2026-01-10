#property indicator_chart_window
#include <UnifiedModel\CEMAs.mqh>
CEMAs g_emas;
int OnInit() { g_emas.Init(_Symbol, _Period); return INIT_SUCCEEDED; }
int OnCalculate(const int total, const int prev, const datetime &time[], const double &open[], const double &high[], const double &low[], const double &close[], const long &tick[], const long &real[], const int &spread[]) {
   g_emas.Calculate(0);
   return total;
}
