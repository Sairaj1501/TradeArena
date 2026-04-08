import numpy as np
import pandas as pd
from core.data_processing import load_data
from server.environment import TradingEnvironment
from tasks.tasks import get_task_config

class QLearningAgent:
    def __init__(self, actions):
        self.q_table = {}
        self.actions = actions
        self.learning_rate = 0.1
        self.discount_factor = 0.9

    def get_state_bucket(self, obs):
        
        rsi_bucket = "high" if obs["rsi"] > 70 else "low" if obs["rsi"] < 30 else "mid"
        trend = obs["trend"]
        pos = obs["position"]
        return f"{rsi_bucket}_{trend}_{pos}"

    def choose_action(self, obs, epsilon=0.1):
        state = self.get_state_bucket(obs)
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.actions)
        
        # Get Q-values for the state, default to zeros if state is new
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        
        return self.actions[np.argmax(self.q_table[state])]

# --- THE MAIN LOOP (The part that makes it RUN) [cite: 203] ---
if __name__ == "__main__":
    # 1. Load Data
    print("Loading data...")
    data = load_data("data/NIFTY 50_minute.csv")

    # 2. Setup Task
    task_name = "easy"  # You can change this to medium or hard [cite: 17]
    config = get_task_config(task_name)
    env = TradingEnv(data, config)

    # 3. Initialize Agent
    actions = ["BUY_CALL", "BUY_PUT", "HOLD", "EXIT"]
    agent = QLearningAgent(actions)

    print(f"Starting training on {task_name} task...")

    # 4. Run Episodes [cite: 204]
    results = []
    for episode in range(10): # Start with 10 to test
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        final_state = env.state()
        print(f"Episode {episode+1}: Equity: {final_state['equity']:.2f}, Trades: {final_state['trade_count']}")
        
        results.append(final_state)

    # 5. Save Results [cite: 204]
    df = pd.DataFrame(results)
    df.to_csv("trading_results.csv", index=False)
    print("Training complete. Results saved to trading_results.csv")