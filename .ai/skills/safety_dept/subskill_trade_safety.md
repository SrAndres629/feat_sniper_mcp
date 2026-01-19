# 🛡️ Skill: Trade Safety (.ai/skills/trade_safety.md)

## 🎯 Objective
Zero-Casualty Execution. Every order must survive the institutional "Gauntlet of Truth".

## 🛠️ Pre-Trade Verification Protocol
Antes de enviar una orden a MetaTrader 5, el sistema DEBE validar los siguientes sensores:

### 1. Spread Toxicity Check
- **Rule**: Spread actual < 1.5x `avg_spread`.
- **Reason**: Evitar slippage excesivo en regímenes de baja liquidez o manipulación.

### 2. Physical Inertia Check
- **Rule**: `layer_alignment` != 0 AND `micro_slope` debe estar en dirección del trade.
- **Reason**: No se lucha contra el momentum cinético inmediato.

### 3. Institutional News Check
- **Rule**: Si faltan < 5 min para una noticia de **High Impact**, la entrada está bloqueada.
- **Reason**: La física se vuelve estocásticamente impredecible durante eventos macro.

### 4. Neural Coherence Veto
- **Rule**: `uncertainty` < 0.05 AND `execute_trade` == True.
- **Reason**: Solo operamos cuando el nexo es "Claro y Decidido".

## 🚧 Blocked Scenarios
- **Martingala**: Queda terminante prohibido aumentar el riesgo tras una pérdida.
- **Overtrading**: No más de 3 posiciones concurrentes por símbolo.
