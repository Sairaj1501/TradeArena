import random
import math
from typing import Optional
from server.models import Observation, Action, Reward
import random
random.seed(42)


class TradingEnvironment:
    def __init__(self, data, task_config):
        self.data = data
        self.task_config = task_config
        self.episode_length = task_config["episode_length"]

        self.initial_balance = 100000
        self.brokerage = 20
        self.slippage_percent = 0.001
        self.stop_loss_limit = task_config.get("stop_loss_threshold", 0.05)

        self.reset()

    # ===============================
    # 🔁 RESET
    # ===============================
    def reset(self) -> Observation:
        print("START: new_episode")
        self.start_index = random.randint(0, len(self.data) - self.episode_length - 1)
        self.current_index = self.start_index

        self.position = "none"
        self.entry_price = None
        self.trade_count = 0

        self.balance = self.initial_balance
        self.position_size = 0
        self.equity = self.balance

        self.current_price = None
        self.peak_equity = self.equity
        self.max_drawdown = 0
       

    # ===============================
    # 📊 OBSERVATION
    # ===============================
    def _get_observation(self):
        row = self.data[self.current_index]

        return {
            "price": float(self.current_price if self.current_price else row["price"]),
            "rsi": float(row["rsi"]),
            "trend": str(row["trend"]),
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

        action = action.action  # extract string

        prev_equity = self.equity
        done = False

        row = self.data[self.current_index]
        base_price = row["price"]

        # volatility
        vol_scale = {"low": 0.5, "medium": 1.2, "high": 3.0}
        vol_val = vol_scale[self.task_config["volatility"]]

        price = base_price + random.uniform(-1 * vol_val, 1 * vol_val)
        self.current_price = price

        # ===============================
        # ACTION VALIDATION
        # ===============================
        if action in ["BUY_CALL", "BUY_PUT"] and self.position != "none":
            action = "HOLD"

        if action == "EXIT" and self.position == "none":
            action = "HOLD"

        # ===============================
        # STOP LOSS
        # ===============================
        if self.position != "none":
            if self.position == "call":
                pnl_pct = (self.current_price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - self.current_price) / self.entry_price

            if pnl_pct <= -self.stop_loss_limit:
                action = "EXIT"

        # ===============================
        # EXECUTION
        # ===============================
        if action == "BUY_CALL" and self.position == "none":
            self.position = "call"
            self.position_size = 0.1 * self.balance
            self.entry_price = self.current_price
            self.trade_count += 1
            self.balance -= (self.position_size + self.brokerage)

        elif action == "BUY_PUT" and self.position == "none":
            self.position = "put"
            self.position_size = 0.1 * self.balance
            self.entry_price = self.current_price
            self.trade_count += 1
            self.balance -= (self.position_size + self.brokerage)

        elif action == "EXIT" and self.position != "none":
            if self.position == "call":
                profit = (self.current_price - self.entry_price) / self.entry_price
            else:
                profit = (self.entry_price - self.current_price) / self.entry_price

            pnl = self.position_size * profit
            self.balance += (self.position_size + pnl - self.brokerage)

            self.position = "none"
            self.entry_price = None
            self.position_size = 0

        # ===============================
        # MOVE FORWARD
        # ===============================
        self.current_index += 1
        if (self.current_index - self.start_index) >= self.episode_length:
            done = True

        # ===============================
        # EQUITY UPDATE
        # ===============================
        if self.position == "none":
            self.equity = self.balance
        else:
            if self.position == "call":
                unrealized = (self.current_price - self.entry_price) / self.entry_price
            else:
                unrealized = (self.entry_price - self.current_price) / self.entry_price

            self.equity = self.balance + self.position_size + (self.position_size * unrealized)

        # ===============================
        # REWARD
        # ===============================
        raw_reward = (self.equity - prev_equity) / self.initial_balance

        # penalty for overtrading
        trade_penalty = -0.001 * self.trade_count

        # penalty for drawdown
        drawdown_penalty = -self.max_drawdown * 0.1

        reward_value = math.tanh(raw_reward + trade_penalty + drawdown_penalty)

        reward = Reward(
            value=float(reward_value),
            done=done,
            info={
                "equity": self.equity,
                "trade_count": self.trade_count,
                "drawdown": self.max_drawdown,
            },
        )

        observation = Observation(**self._get_observation()) if not done else None
        print(f"STEP: action={action} equity={self.equity} trades={self.trade_count}")
        return observation, reward

    # ===============================
    # 📌 STATE
    # ===============================
    def state(self):
        return {
            "equity": float(self.equity),
            "balance": float(self.balance),
            "position": str(self.position),
            "trade_count": int(self.trade_count),
            "max_drawdown": float(self.max_drawdown),
            "step": int(self.current_index)
        }
