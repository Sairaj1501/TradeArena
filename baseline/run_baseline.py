import os
import json
import random
import numpy as np
from openai import OpenAI

from core.data_processing import load_data
from server.environment import TradingEnvironment
from tasks.tasks import get_task_config
from grader.grader import grade_agent


# ===============================
# 🔒 Reproducibility (IMPORTANT)
# ===============================
random.seed(42)
np.random.seed(42)


# ===============================
# 🤖 OpenAI Client (ENV CONFIG)
# ===============================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("API_BASE_URL")  # optional but required by checklist
)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")


# ===============================
# 🧠 LLM Action Generator
# ===============================
def get_llm_action(obs):
    """Generate action using LLM with strict validation."""

    prompt = f"""
    You are an expert options trader.

    Based on the following market data, choose ONE action:
    BUY_CALL, BUY_PUT, HOLD, EXIT

    Observation:
    - Price: {obs.price}
    - RSI: {obs.rsi}
    - Trend: {obs.trend}
    - Position: {obs.position}
    - Equity: {obs.equity}

    Respond with ONLY the action name.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0  # 🔥 makes it more deterministic
        )

        action = response.choices[0].message.content.strip().upper()

    except Exception as e:
        print(f"STEP: error=LLM_failure fallback=HOLD")
        return "HOLD"

    # ✅ Strict validation
    if action not in ["BUY_CALL", "BUY_PUT", "HOLD", "EXIT"]:
        print(f"STEP: invalid_action={action} fallback=HOLD")
        return "HOLD"

    return action


# ===============================
# 🚀 MAIN EXECUTION
# ===============================
if __name__ == "__main__":

    print("START: baseline_run")

    # Load data
    data = load_data("data/NIFTY 50_minute.csv")

    # Select task
    task = "easy"  # change to medium/hard if needed
    task_config = get_task_config(task)

    # Initialize environment
    env = TradingEnvironment(data, task_config)

    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    print(f"STEP: init task={task}")

    # ===============================
    # 🔁 EPISODE LOOP
    # ===============================
    while not done:

        action = get_llm_action(obs)

        print(f"STEP: step={step_count} action={action}")

        obs, reward, done, info = env.step(action)

        total_reward += reward
        step_count += 1

    # ===============================
    # 📊 FINAL EVALUATION
    # ===============================
    final_state = env.state()
    score = grade_agent(task, final_state)

    print(f"END: baseline_run score={score}")

    # Optional readable summary (not required but helpful)
    print("-" * 40)
    print(f"Final Equity : {final_state['equity']:.2f}")
    print(f"Trade Count  : {final_state['trade_count']}")
    print(f"Total Reward : {total_reward:.4f}")
    print(f"Hackathon Score : {score}")
    print("-" * 40)
