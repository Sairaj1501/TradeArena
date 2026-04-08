import numpy as np
from core.data_processing import load_data
from env.trading_env import TradingEnv
from tasks.tasks import get_task_config
from grader.grader import grade_agent


def get_rule_action(obs):
    if obs.position == "none":
        if obs.rsi < 35 and obs.trend == "bullish":
            return "BUY_CALL"
        elif obs.rsi > 65 and obs.trend == "bearish":
            return "BUY_PUT"
        else:
            return "HOLD"
    else:
        if obs.position == "call" and obs.rsi > 60:
            return "EXIT"
        elif obs.position == "put" and obs.rsi < 40:
            return "EXIT"

        if obs.time_to_expiry < 5:
            return "EXIT"

        return "HOLD"


if __name__ == "__main__":
    print("Running Rule-Based Baseline (Multi-Episode)...")

    data = load_data("data/NIFTY 50_minute.csv")
    task = "easy"
    config = get_task_config(task)

    num_episodes = 5  # 🔥 change this (5–10 is ideal)

    scores = []
    equities = []
    trades = []

    for episode in range(num_episodes):
        env = TradingEnv(data, config)

        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = get_rule_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        final_state = env.state()
        score = grade_agent(task, final_state)

        scores.append(score)
        equities.append(final_state["equity"])
        trades.append(final_state["trade_count"])

        print(f"Episode {episode+1}: Score={score}, Equity={final_state['equity']:.2f}, Trades={final_state['trade_count']}")

    # ✅ FINAL AVERAGE RESULTS
    print("\n" + "="*40)
    print("AVERAGE RESULTS")
    print("="*40)
    print(f"Avg Score: {np.mean(scores):.2f}")
    print(f"Avg Equity: {np.mean(equities):.2f}")
    print(f"Avg Trades: {np.mean(trades):.2f}")
    print("="*40)