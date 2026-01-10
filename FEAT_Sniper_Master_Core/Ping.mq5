//+------------------------------------------------------------------+
//|                                                         Ping.mq5 |
//|                                  Diagnostic Ping Tool           |
//+------------------------------------------------------------------+
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

int OnInit() {
   Print("== [PING] Indicator Loaded Successfully ==");
   Print("== [PING] Symbol: ", _Symbol, " Period: ", _Period, " ==");
   return(INIT_SUCCEEDED);
}

int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[]) {
   Print("== [PING] Tick Received at ", TimeToString(TimeCurrent()));
   return(rates_total);
}
