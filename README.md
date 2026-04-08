# 📈 OpenEnv NIFTY 50 Options Trading Simulator

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://github.com/openenv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 📌 Project Overview
This project implements a high-fidelity **OpenEnv environment** for professional options trading, specifically modeled on the **NIFTY 50 (NSE India)** market. Unlike "toy" environments, this simulator enforces real-world constraints including **slippage (0.1%)**, **brokerage (₹20/trade)**, and **dynamic volatility regimes**.

---

## 🎯 Motivation & Real-World Utility
Trading agents often fail in production because they are trained on "clean" data. This environment bridges that gap by modeling:
* **Volatility Injection:** Agents must adapt to changing market conditions (Low vs. High volatility).
* **Risk Discipline:** Enforced stop-loss thresholds (2% to 10%) and drawdown monitoring.
* **Capital Preservation:** A reward function that penalizes inactivity and over-trading.

---

## 🧠 Environment Design

### 🔁 Core API (OpenEnv Standard)
* `reset()`: Restarts the environment at a randomized market window.
* `step(Action)`: Executes a trade, calculates slippage/brokerage, and returns `(Observation, Reward, Done, Info)`.
* `state()`: Provides full transparency into account equity, current drawdown, and trade count.

### 📊 Observation Space (Typed Pydantic Model)
| Feature | Type | Description |
| :--- | :--- | :--- |
| `price` | `float` | Real-time NIFTY 50 price with synthetic noise. |
| `rsi` | `float` | Relative Strength Index (14-period). |
| `trend` | `str` | `bullish`, `bearish`, or `sideways`. |
| `time_to_expiry` | `int` | Remaining steps in the trading session. |
| `position` | `str` | Current holding: `none`, `call`, or `put`. |
| `equity` | `float` | Current account value (Cash + Unrealized PnL). |

### 🎮 Action Space
* `BUY_CALL`: Enter a long position on market upside.
* `BUY_PUT`: Enter a long position on market downside.
* `HOLD`: Maintain current state (slight penalty to prevent "do-nothing" bias).
* `EXIT`: Close current position and realize PnL.

---

## 🧪 Tasks & Grading Criteria
We provide a curated progression of difficulty to test agent robustness:

| Task | Market Type | Volatility | Success Criteria (Grader) |
| :--- | :--- | :--- | :--- |
| **🟢 Easy** | Trending | Low (0.5x) | Basic profitability (>5% return). |
| **🟡 Medium** | Sideways | Med (1.2x) | Profitability + Trade efficiency (penalty for over-trading). |
| **🔴 Hard** | Volatile | High (3.0x) | **Survival.** High weight on Max Drawdown (<15%). |

---

## 🏆 Reward Shaping
The reward function utilizes a **Hyperbolic Tangent (tanh)** squash to normalize equity changes:
$$Reward = \tanh\left(\frac{Equity_t - Equity_{t-1}}{InitialBalance}\right)$$
* **Partial Progress:** Rewards small gains at every step.
* **Inactivity Penalty:** A -0.001 penalty for `HOLD` actions when `position="none"` forces the agent to seek opportunities.

---

## ⚙️ Installation & Usage

### 1. Setup Environment
```bash
# Clone the repo
git clone <your-repo-url>
cd project-root

# Install dependencies
pip install -r requirements.txt