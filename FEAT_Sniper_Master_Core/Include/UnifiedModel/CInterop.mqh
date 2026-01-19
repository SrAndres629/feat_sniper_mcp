//+------------------------------------------------------------------+
//|                                                     CInterop.mqh |
//|                                     QUANT_ARCHITECT_OMEGA : ZMQ  |
//|                                     Protocol: Hyperbridge (x64)  |
//+------------------------------------------------------------------+
#property copyright "FEAT Systems AI"
#property strict

// --- ZMQ CONSTANTS ---
// --- ZMQ CONSTANTS ---
#define ZMQ_REQ       3
#define ZMQ_REP       4
#define ZMQ_DEALER    5
#define ZMQ_ROUTER    6
#define ZMQ_PULL      7
#define ZMQ_PUSH      8
#define ZMQ_XPUB      9
#define ZMQ_SUB       2    // CORRECT: 2
#define ZMQ_PUB       1    // CORRECT: 1
#define ZMQ_SUBSCRIBE 6    // NEW: Option for filtering

#define ZMQ_NOBLOCK   1
#define ZMQ_DONTWAIT  1
#define ZMQ_SNDHWM    23
#define ZMQ_RCVHWM    24

// --- DLL IMPORTS (CANONICAL x64) ---
#import "libzmq.dll"
   long zmq_ctx_new();
   int  zmq_ctx_term(long context);
   long zmq_socket(long context, int type);
   int  zmq_close(long socket);
   
   // Array References are correct for MQL5
   int  zmq_bind(long socket, uchar &endpoint[]);
   int  zmq_connect(long socket, uchar &endpoint[]);
   
   int  zmq_send(long socket, uchar &buf[], ulong len, int flags);
   int  zmq_recv(long socket, uchar &buf[], ulong len, int flags);
   
   int  zmq_setsockopt(long socket, int option_name, uchar &option_value[], ulong option_len);
   
   int  zmq_errno();
#import

// --- WRAPPER CLASS ---
class CInterop
{
private:
   long m_context; 
   long m_socket; 
   bool m_connected;

public:
                     CInterop();
                    ~CInterop();
   
   bool              Init(bool is_publisher, int port);
   void              Shutdown();
   bool              Send(string message, bool non_blocking=true);
   string            ReceiveHUD(bool non_blocking=true);
   int               GetError() { return zmq_errno(); }
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CInterop::CInterop() : m_context(0), m_socket(0), m_connected(false)
{
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CInterop::~CInterop()
{
   Shutdown();
}

//+------------------------------------------------------------------+
//| Init                                                             |
//+------------------------------------------------------------------+
bool CInterop::Init(bool is_publisher, int port)
{
   if(m_connected) Shutdown();

   Print("[HYPERBRIDGE] Allocating ZMQ Context (x64 Array)...");
   m_context = zmq_ctx_new();
   if(m_context == 0) {
      Print("[HYPERBRIDGE] Failed to create context.");
      return false;
   }

   // [FIX] Ensure correct socket type (PUSH=8, SUB=2)
   int type = is_publisher ? ZMQ_PUSH : ZMQ_SUB;
   m_socket = zmq_socket(m_context, type);
   
   if(m_socket == 0) {
      Print("[HYPERBRIDGE] Failed to create socket.");
      return false;
   }

   uchar endpoint[];
   int rc = 0;

   string addr = StringFormat("tcp://127.0.0.1:%d", port);
   StringToCharArray(addr, endpoint, 0, WHOLE_ARRAY, CP_UTF8);

   if(is_publisher) {
      rc = zmq_connect(m_socket, endpoint);
      if(rc != 0) {
          Print("<<< [FORENSIC] PUSH CONNECT FAIL (", port, "). Error: ", zmq_errno());
          return false;
      }
   } else {
      rc = zmq_connect(m_socket, endpoint);
      
      if(rc != 0) {
         Print("<<< [FORENSIC] SUB CONNECT FAIL (", port, "). Error: ", zmq_errno());
         return false;
      }
      
      // SUBSCRIBE ALL (Option 6)
      uchar filter[1]; 
      filter[0] = 0;
      rc = zmq_setsockopt(m_socket, ZMQ_SUBSCRIBE, filter, (ulong)0); 
      
      if(rc != 0) {
         Print("<<< [FORENSIC] SUB SETSOCKOPT FAIL (Filter). Error: ", zmq_errno());
         return false;
      }
   }

   union IntUnion {
      int val;
      uchar bytes[4];
   };
   
   // HWM Protection (1000 messages max buffer)
   IntUnion u_hwm;
   u_hwm.val = 1000;
   zmq_setsockopt(m_socket, ZMQ_SNDHWM, u_hwm.bytes, (ulong)4);
   zmq_setsockopt(m_socket, ZMQ_RCVHWM, u_hwm.bytes, (ulong)4);

   IntUnion u_linger;
   u_linger.val = 0; 
   
   // Pass array
   zmq_setsockopt(m_socket, 17, u_linger.bytes, (ulong)4); 

   m_connected = true;
   Print("[HYPERBRIDGE] Link Established. Socket: ", m_socket);
   return true;
}

//+------------------------------------------------------------------+
//| Shutdown                                                         |
//+------------------------------------------------------------------+
void CInterop::Shutdown()
{
   if(m_socket != 0) {
      zmq_close(m_socket);
      m_socket = 0;
   }
   if(m_context != 0) {
      zmq_ctx_term(m_context);
      m_context = 0;
   }
   m_connected = false;
}

//+------------------------------------------------------------------+
//| Send                                                             |
//+------------------------------------------------------------------+
bool CInterop::Send(string message, bool non_blocking=true)
{
   if(!m_connected) return false;
   
   uchar data[];
   int len = StringToCharArray(message, data) - 1; 
   if(len < 0) len = 0;
   
   int flags = non_blocking ? ZMQ_DONTWAIT : 0;
   
   int rc = zmq_send(m_socket, data, (ulong)len, flags);
   
   return (rc != -1);
}

//+------------------------------------------------------------------+
//| ReceiveHUD (Renamed from Receive to match Visualizer)            |
//+------------------------------------------------------------------+
string CInterop::ReceiveHUD(bool non_blocking=true)
{
   if(!m_connected) return "";
   
   uchar buf[4096];
   int flags = non_blocking ? ZMQ_DONTWAIT : 0;
   
   int rc = zmq_recv(m_socket, buf, (ulong)4096, flags);
   
   if(rc <= 0) return "";
   
   return CharArrayToString(buf, 0, rc);
}
