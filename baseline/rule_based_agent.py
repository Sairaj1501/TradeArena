import random
import numpy as np
from core.data_processing import load_data
from server.environment import TradingEnvironment
from tasks.tasks import get_task_config
from grader.grader import grade_agent


def get_rule_action(obs):
    if obs.position == "none":
        if obs.rsi < 45:
            return "BUY_CALL"
        elif obs.rsi > 55:
            return "BUY_PUT"
        
        # fallback exploration
        if random.random() < 0.1:
            return random.choice(["BUY_CALL", "BUY_PUT"])
        
        return "HOLD"

    else:
        if obs.position == "call" and obs.rsi > 55:
            return "EXIT"
        elif obs.position == "put" and obs.rsi < 45:
            return "EXIT"

        if obs.time_to_expiry < 10:
            return "EXIT"

        return "HOLD"
    

if __name__ == "__main__":
    print("🚀 Running Rule-Based Agent (All Tasks | Multi-Episode)\n")

    data = load_data("data/NIFTY 50_minute.csv")

    tasks = ["easy", "medium", "hard"]
    num_episodes = 5  # 🔥 You can increase to 10

    final_results = {}

    # 🔁 Loop through all tasks
    for task in tasks:
        print("\n" + "="*60)
        print(f"📊 TASK: {task.upper()}")
        print("="*60)

        config = get_task_config(task)

        scores = []
        equities = []
        trades = []

        # 🔁 Run multiple episodes
        for episode in range(num_episodes):
            env = TradingEnv(data, config)

            obs = env.reset()
            done = False

            while not done:
                action = get_rule_action(obs)
                obs, reward, done, info = env.step(action)

            final_state = env.state()
            score = grade_agent(task, final_state)

            scores.append(score)
            equities.append(final_state["equity"])
            trades.append(final_state["trade_count"])

            print(f"Episode {episode+1} → Score: {score}, Equity: {final_state['equity']:.2f}, Trades: {final_state['trade_count']}")

        # 📊 Compute averages
        avg_score = np.mean(scores)
        avg_equity = np.mean(equities)
        avg_trades = np.mean(trades)

        final_results[task] = avg_score

        print("\n--- Task Summary ---")
        print(f"Avg Score   : {avg_score:.2f}")
        print(f"Avg Equity  : {avg_equity:.2f}")
        print(f"Avg Trades  : {avg_trades:.2f}")

    # 🏆 FINAL SUMMARY
    print("\n" + "="*60)
    print("🏆 FINAL PERFORMANCE SUMMARY")
    print("="*60)

    for task in tasks:
        print(f"{task.upper():<10} → Avg Score: {final_results[task]:.2f}")

    overall_avg = np.mean(list(final_results.values()))

    print("="*60)