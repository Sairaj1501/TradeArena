import math
import random
from typing import Optional
from server.models import Observation, Action, Reward

# Reproducibility
random.seed(42)

class TradingEnvironment:
    def __init__(self, data, task_config):
        self.data = data
        self.task_config = task_config
        self.episode_length = task_config.get("episode_length", 200) #

        self.initial_balance = 100000.0
        self.brokerage = 20.0
        self.stop_loss_limit = task_config.get("stop_loss_threshold", 0.05)

        self.reset()

    # ===============================
    # 🔁 RESET
    # ===============================
    def reset(self) -> Observation:
        # Prevent index out of bounds
        max_start = max(0, len(self.data) - self.episode_length - 1)
        self.start_index = random.randint(0, max_start)
        self.current_index = self.start_index

        self.position = "none"
        self.entry_price = None
        self.trade_count = 0

        self.balance = self.initial_balance
        self.position_size = 0.0
        self.equity = self.balance

        self.current_price = None
        self.peak_equity = self.equity
        self.max_drawdown = 0.0
        
        return self._get_observation()

    # ===============================
    # 📊 OBSERVATION (Safe Fetch)
    # ===============================
    def _get_observation(self):
        # Handle both DataFrame and List of Dicts
        if hasattr(self.data, "iloc"):
            row = self.data.iloc[self.current_index]
        else:
            row = self.data[self.current_index]

        # FIX: Robust price fetching to avoid 'close' error
        price = float(row.get('price') or row.get('close') or row.get('Close') or row.get('Price') or 0)
        
        return {
            "price": float(self.current_price if self.current_price else price),
            "rsi": float(row.get('rsi', 50)),
            "trend": str(row.get('trend', 'neutral')),
            "time_to_expiry": int(self.episode_length - (self.current_index - self.start_index)),
            "position": str(self.position),
            "entry_price": float(self.entry_price) if self.entry_price else None,
            "balance": float(self.balance),
            "equity": float(self.equity),
        }

    # ===============================
    # ⚡ STEP
    # ===============================
    def step(self, action: Action) -> tuple[Optional[Observation], Reward]:
        action_str = action.action if hasattr(action, 'action') else str(action).upper()

        prev_equity = self.equity
        done = False

        # Safe Price Fetch for current step
        if hasattr(self.data, "iloc"):
            row = self.data.iloc[self.current_index]
        else:
            row = self.data[self.current_index]
            
        base_price = float(row.get('price') or row.get('close') or row.get('Close') or 0)

        # Volatility Logic
        vol_scale = {"low": 0.5, "medium": 1.2, "high": 3.0}
        vol_key = self.task_config.get("volatility", "medium")
        vol_val = vol_scale.get(vol_key, 1.2)

        self.current_price = base_price + random.uniform(-1 * vol_val, 1 * vol_val)

        # Action Validation
        if action_str in ["BUY_CALL", "BUY_PUT"] and self.position != "none":
            action_str = "HOLD"
        if action_str == "EXIT" and self.position == "none":
            action_str = "HOLD"

        # Stop Loss
        if self.position != "none":
            pnl_pct = (self.current_price - self.entry_price) / self.entry_price if self.position == "call" else (self.entry_price - self.current_price) / self.entry_price
            if pnl_pct <= -self.stop_loss_limit:
                action_str = "EXIT"

        # Execution Logic
        if action_str in ["BUY_CALL", "BUY_PUT"] and self.position == "none":
            self.position = action_str.replace("BUY_", "").lower()
            self.position_size = 0.1 * self.balance
            self.entry_price = self.current_price
            self.trade_count += 1
            self.balance -= (self.position_size + self.brokerage)

        elif action_str == "EXIT" and self.position != "none":
            profit = (self.current_price - self.entry_price) / self.entry_price if self.position == "call" else (self.entry_price - self.current_price) / self.entry_price
            pnl = self.position_size * profit
            self.balance += (self.position_size + pnl - self.brokerage)
            self.position = "none"
            self.entry_price = None
            self.position_size = 0.0

        # Equity Update
        if self.position == "none":
            self.equity = self.balance
        else:
            unrealized = (self.current_price - self.entry_price) / self.entry_price if self.position == "call" else (self.entry_price - self.current_price) / self.entry_price
            self.equity = self.balance + self.position_size + (self.position_size * unrealized)

        # Drawdown Update
        self.peak_equity = max(self.peak_equity, self.equity)
        current_dd = (self.peak_equity - self.equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, current_dd)

        # Move Forward
        self.current_index += 1
        if (self.current_index - self.start_index) >= self.episode_length or self.current_index >= len(self.data):
            done = True

        # ===============================
        # 🛡️ REWARD NORMALIZATION (0.0 to 1.0)
        # ===============================
        raw_reward = (self.equity - prev_equity) / self.initial_balance
        trade_penalty = -0.0005 * self.trade_count
        drawdown_penalty = -self.max_drawdown * 0.05

        # Strictly 0.0 to 1.0 range
        reward_value = (math.tanh(raw_reward + trade_penalty + drawdown_penalty) + 1) / 2

        reward = Reward(
            value=float(reward_value),
            done=done,
            info={
                "equity": self.equity,
                "trade_count": self.trade_count,
                "drawdown": self.max_drawdown,
            },
        )

        obs_dict = self._get_observation()
        observation = Observation(**obs_dict) if not done else None
        return observation, reward

    def state(self):
        return {
            "equity": float(self.equity),
            "balance": float(self.balance),
            "position": str(self.position),
            "trade_count": int(self.trade_count),
            "max_drawdown": float(self.max_drawdown),
            "step": int(self.current_index - self.start_index)
        }