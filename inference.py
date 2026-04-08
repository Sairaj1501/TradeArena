import os
import sys
import json
import random
import numpy as np
from openai import OpenAI

from core.data_processing import load_data
from server.environment import TradingEnvironment
from tasks.tasks import get_task_config
from grader.grader import grade_agent

# ===============================
# Reproducibility (IMPORTANT)
# ===============================
random.seed(42)
np.random.seed(42)

# ===============================
# OpenAI Client (ENV CONFIG)
# ===============================
def get_openai_client():
    # Checklist: "Must use OpenAI Client for all LLM calls using above variables"
    return OpenAI(
        api_key=os.getenv("HF_TOKEN", "dummy_key_if_missing"), 
        base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1") 
    )

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# ===============================
# LLM Action Generator
# ===============================
def get_llm_action(obs):
    """Generate action using LLM with strict validation."""

    client = get_openai_client()
    
    # Check if obs is a dictionary, if not it might be a Pydantic object
    obs_dict = obs if isinstance(obs, dict) else obs.dict()

    prompt = f"""
    You are an expert options trader.

    Based on the following market data, choose ONE action:
    BUY_CALL, BUY_PUT, HOLD, EXIT

    Observation:
    - Price: {obs_dict['price']}
    - RSI: {obs_dict['rsi']}
    - Trend: {obs_dict['trend']}
    - Position: {obs_dict['position']}
    - Equity: {obs_dict['equity']}

    Respond with ONLY the action name.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )

        action = response.choices[0].message.content.strip().upper()

    except Exception as e:
        print(f"[STEP] error=LLM_failure fallback=HOLD")
        return "HOLD"

    if action not in ["BUY_CALL", "BUY_PUT", "HOLD", "EXIT"]:
        print(f"[STEP] invalid_action={action} fallback=HOLD")
        return "HOLD"

    return action


# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    
    # "3+ tasks with graders"
    tasks = ["easy", "medium", "hard"]

    try:
        data = load_data("data/NIFTY 50_minute.csv")
    except Exception as e:
        print("[END] score=0.0")
        sys.exit(0)

    for task in tasks:
        # Checklist format constraints: [START]
        print(f"[START] {task}")
        
        task_config = get_task_config(task)
        env = TradingEnvironment(data, task_config)

        obs = env.reset()
        done = False
        step_count = 0

        while not done:
            action = get_llm_action(obs)

            # Checklist format constraints: [STEP]
            print(f"[STEP] step={step_count} action={action}")

            obs, reward_obj = env.step(action)
            done = reward_obj.done
            step_count += 1
            
            # Infra Restriction: Prevent endless loop hitting the 20 minute limit
            if step_count > 500:
                print("[STEP] forced_break=true")
                break

        final_state = env.state()
        try:
            score = grade_agent(task, final_state)
        except Exception:
            score = 0.0

        # Checklist format constraints: [END]
        print(f"[END] {task} score={score}")
