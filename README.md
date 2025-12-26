# 🧠 MT5 Neural Sentinel

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-2.0+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MetaTrader5](https://img.shields.io/badge/MetaTrader_5-Compatible-orange?style=for-the-badge)
![MCP](https://img.shields.io/badge/MCP_Protocol-Ready-blueviolet?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-ready MCP Server that bridges AI assistants (Claude, Cursor, n8n) with MetaTrader 5 for autonomous algorithmic trading.**

[Features](#-features) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [MCP Tools](#-mcp-tools) • [Architecture](#-architecture)

</div>

---

## ✨ Features

🔌 **Dual Interface** — Native MCP protocol + REST API for maximum compatibility  
📊 **Real-time Market Data** — OHLCV candles, spreads, volatility metrics  
🎯 **Trade Execution** — Market orders, pending orders, position management  
📈 **Technical Analysis** — RSI, MACD, Moving Averages, Bollinger Bands, ATR  
👁️ **Vision Mode** — Screen capture of MT5 charts for visual pattern analysis  
🛡️ **Risk Management** — Account metrics, margin monitoring, trade history analytics  
📅 **Economic Calendar** — Upcoming events with importance filtering  
🔧 **Quant Coder** — Create custom MQL5 indicators directly from the AI  

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **MetaTrader 5** installed and logged in
- Windows OS (MT5 requirement)

### Installation

```bash
# Clone the repository
git clone https://github.com/SrAndres629/feat_sniper_mcp.git
cd feat_sniper_mcp

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install fastmcp fastapi uvicorn MetaTrader5 pydantic pillow numpy
```

### Configuration

Create a `.env` file in the project root:

```env
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
```

### Running the Server

**Option 1: MCP Server (for Claude Desktop, Cursor)**
```bash
python mcp_server.py
```

**Option 2: REST API (for n8n, webhooks)**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Option 3: Windows Batch Script**
```bash
start_server.bat
```

---

## 🛠️ MCP Tools

When connected via MCP protocol, these tools are available to AI assistants:

| Tool | Description |
|------|-------------|
| `get_market_data` | Fetch OHLCV candlestick data (JSON or CSV format) |
| `get_account_status` | Get balance, equity, margin, and profit |
| `get_market_panorama` | Capture MT5 screen for visual analysis |
| `execute_trade` | Execute BUY/SELL orders (market or pending) |
| `manage_trade` | Close, modify, or delete existing positions |
| `get_technical_indicator` | Calculate RSI, MACD, MA, ATR, Bollinger |
| `get_trade_performance` | Win rate, profit factor, and trade analytics |
| `get_economic_calendar` | Upcoming economic events |
| `create_mql5_indicator` | Write and compile custom MQL5 indicators |

---

## 📡 API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Connection status check |
| `POST` | `/market/candles` | Get OHLCV data |
| `GET` | `/market/account` | Account metrics |
| `POST` | `/market/volatility` | ATR and spread metrics |
| `POST` | `/market/indicators` | Technical indicators |
| `POST` | `/market/calendar` | Economic calendar |
| `POST` | `/market/create_indicator` | Create MQL5 indicator |
| `POST` | `/vision/panorama` | Screen capture |
| `POST` | `/trade/order` | Execute trade |
| `POST` | `/trade/manage` | Manage position |
| `POST` | `/account/history` | Trade history and KPIs |

### Example Request

```bash
curl -X POST http://localhost:8000/market/candles \
  -H "Content-Type: application/json" \
  -d '{"symbol": "EURUSD", "timeframe": "H1", "n_candles": 100}'
```

### Response Format

All endpoints return a standardized envelope:

```json
{
  "status": "success",
  "data": { ... },
  "error": null,
  "timestamp": "2024-12-25T20:00:00Z"
}
```

---

## 🏗️ Architecture

```
feat_sniper_mcp/
├── mcp_server.py          # MCP Protocol Server (Claude, Cursor)
├── run.py                 # Alternative entry point
├── start_server.bat       # Windows launcher
└── app/
    ├── main.py            # FastAPI REST Gateway
    ├── core/
    │   ├── config.py      # Environment configuration
    │   └── mt5_conn.py    # MT5 connection manager
    ├── models/
    │   └── schemas.py     # Pydantic request/response models
    └── skills/
        ├── market.py      # Market data operations
        ├── vision.py      # Screen capture
        ├── execution.py   # Trade execution
        ├── trade_mgmt.py  # Position management
        ├── indicators.py  # Technical analysis
        ├── history.py     # Trade history
        ├── calendar.py    # Economic calendar
        └── quant_coder.py # MQL5 code generation
```

---

## 🔗 Integration Examples

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mt5": {
      "command": "python",
      "args": ["C:/path/to/mcp_server.py"]
    }
  }
}
```

### n8n Workflow

Use the HTTP Request node to call REST endpoints:

```
POST http://localhost:8000/trade/order
{
  "symbol": "EURUSD",
  "action": "BUY",
  "volume": 0.1,
  "sl": 1.0500,
  "tp": 1.0700
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for the AI Trading Community**

</div>
