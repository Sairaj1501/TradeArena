import random
import math
from env.models import Observation, Reward  # Added Reward here


class TradingEnv:
    def __init__(self, data, task_config):
        self.data = data
        self.task_config = task_config
        self.episode_length = task_config["episode_length"]
        self.initial_balance = 100000
        self.brokerage = 20
        self.slippage_percent = 0.001
        self.stop_loss_limit = task_config.get("stop_loss_threshold", 0.05)

    def reset(self):
        self.equity_curve = []
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

        obs_dict = self._get_observation()
        return Observation(**obs_dict)

    def _get_observation(self):
        row = self.data[self.current_index]
        return {
            "price": float(self.current_price if self.current_price is not None else row["price"]),
            "rsi": float(row["rsi"]),
            "trend": str(row["trend"]),
            "time_to_expiry": int(self.episode_length - (self.current_index - self.start_index)),
            "position": str(self.position),
            "entry_price": float(self.entry_price) if self.entry_price is not None else None,
            "balance": float(self.balance),
            "equity": float(self.equity)
        }

    def step(self, action):
        prev_equity = self.equity
        done = False
        unrealized_pnl_pct = 0 

        # 1. Get base data
        row = self.data[self.current_index]
        base_price = row["price"]
        market_type = self.task_config["market_type"] 

        # 2. VOLATILITY LOGIC
        vol_scale = {"low": 0.5, "medium": 1.2, "high": 3.0}
        vol_val = vol_scale[self.task_config["volatility"]]

        price = base_price
        if market_type == "trending":
            price += (0.1 * vol_val) + random.uniform(-0.5 * vol_val, 0.5 * vol_val)
        elif market_type == "sideways":
            price += random.uniform(-1.5 * vol_val, 1.5 * vol_val)
        elif market_type == "volatile":
            price += random.uniform(-4.0 * vol_val, 4.0 * vol_val)

        self.current_price = price

        # 3. ACTION INTERCEPTOR
        if action in ["BUY_CALL", "BUY_PUT"]:
            if self.position != "none": action = "HOLD" 
        elif action == "EXIT":
            if self.position == "none": action = "HOLD" 

        # 4. STOP-LOSS
        if self.position != "none":
            if self.position == "call":
                unrealized_pnl_pct = (self.current_price - self.entry_price) / self.entry_price
            else:
                unrealized_pnl_pct = (self.entry_price - self.current_price) / self.entry_price
            
            if unrealized_pnl_pct <= -self.stop_loss_limit:
                action = "EXIT" 

        # 5. FIXED BALANCE CHECK (Returns Pydantic Models now)
        if self.balance < 1000:
            info = {"reason": "insufficient_balance"}
            obs_model = Observation(**self._get_observation())
            return obs_model, -1.0, True, info

        # 6. EXECUTE ACTIONS
        if action == "BUY_CALL" and self.position == "none":
            self.position = "call"
            self.position_size = 0.1 * self.balance
            self.entry_price = self.current_price * (1 + self.slippage_percent)
            self.trade_count += 1
            self.balance -= (self.position_size + self.brokerage)

        elif action == "BUY_PUT" and self.position == "none":
            self.position = "put"
            self.position_size = 0.1 * self.balance
            self.entry_price = self.current_price * (1 - self.slippage_percent)
            self.trade_count += 1
            self.balance -= (self.position_size + self.brokerage)

        elif action == "EXIT" and self.position != "none":
            if self.position == "call":
                exit_price = self.current_price * (1 - self.slippage_percent)
                profit = (exit_price - self.entry_price) / self.entry_price
            else:
                exit_price = self.current_price * (1 + self.slippage_percent)
                profit = (self.entry_price - exit_price) / self.entry_price

            pnl = self.position_size * profit
            self.balance += (self.position_size + pnl - self.brokerage)
            self.position = "none"
            self.entry_price = None
            self.position_size = 0

        # 7. MOVE FORWARD
        self.current_index += 1
        if (self.current_index - self.start_index) >= self.episode_length:
            done = True 

        # 8. EQUITY UPDATE
        if self.position == "none":
            self.equity = self.balance
        else:
            if self.position == "call":
                unrealized = (self.current_price - self.entry_price) / self.entry_price
            else:
                unrealized = (self.entry_price - self.current_price) / self.entry_price
            self.equity = self.balance + self.position_size + (self.position_size * unrealized)
        #  Update Drawdown Tracking
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # 9. REWARD
        raw_reward = (self.equity - prev_equity) / max(self.initial_balance, 1)
        if self.position == "none" and action == "HOLD":
            raw_reward -= 0.001 
        reward = math.tanh(raw_reward) 

        # 10. PACKAGING
        info = {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "trade_count": int(self.trade_count),
            "max_drawdown": float(self.max_drawdown)
        }
        
        obs_dict = self._get_observation()
        observation_model = Observation(**obs_dict) if not done else None
        
        return observation_model, float(reward), done, info

    def state(self):
        return {
            "balance": self.balance,
            "equity": self.equity,
            "position": self.position,
            "entry_price": self.entry_price,
            "trade_count": self.trade_count,
            "max_drawdown": self.max_drawdown,
            "step": self.current_index
        }