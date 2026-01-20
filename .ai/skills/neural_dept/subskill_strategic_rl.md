# Subskill: STRATEGIC POLICY OPTIMIZER (Game Theory Specialist)

## 🧠 Description
This specialist manages the **Strategic Cortex**, a Reinforcement Learning (PPO) module designed to transcend rigid "if-then" rules. Instead of hardcoded logic, it learns an optimal policy for capital deployment based on market states ($20 vs $1000) and microstructure regimes (Entropy/OFI).

## 🛠️ Protocols

### 1. State Encoding Protocol
The Agent observes a 16-dimensional Tensor representing the battlefield:
- **Account**: Balance, Growth Phase.
- **Physics**: Titanium Floors, Kinetic Acceleration.
- **Microstructure**: Shannon Entropy (Noise), Order Flow Imbalance (Intent).
- **Neural**: Probability confidence from the Deep Brain.

### 2. Action Space (The Move Set)
The Agent selects one of 4 strategic stances:
- `TWIN_SNIPER` (Aggressive): Deploys 2 units. Logic: "High Certitude + Low Noise = Maximum Expansion".
- `STANDARD` (Balanced): Deploys 1 unit. Logic: "Standard opportunity, preserve ammo".
- `DEFENSIVE` (Survival): Deploys Reduced unit with Tight Stops. Logic: "High Volatility but possible edge".
- `HOLD` (Discipline): Deploys 0 units. Logic: "Entropy is high, waiting is profitable".

### 3. Shadow Training Mode
Currently, the Agent operates in **Shadow Mode**:
- It observes live data and makes decisions.
- Decisions are logged but NOT executed (Legacy Logic rules).
- **Goal**: Collect 1000 "Shadow Experiences" to calibrate the PPO network before giving it the nuclear codes (Real Money Control).

### 4. Reward Function (Constitution)
The Agent is trained on the "Gain without Dying" function:
$$R = Profit - (Loss \times 2.5) - (Drawdown \times 10) + (EntropyAvoidance)$$
It is heavily punished for losing the $20 base capital.

## 🎯 Objective
Evolve from a "Rule-Based Algorithm" to a "Capital Allocator Intelligence" that intuitively knows when to bet big and when to fold.
