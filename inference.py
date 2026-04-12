import os
import sys
import random
import numpy as np
from pathlib import Path
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
def get_api_key():
    return os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")


def get_openai_client():
    return OpenAI(
        api_key=get_api_key(),
        base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    )

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# ===============================
# LLM Action Generator
# ===============================
def get_llm_action(obs):
    api_key = get_api_key()

    # If no API key → skip LLM completely
    if not api_key:
        raise Exception("No API key - using fallback")

    client = get_openai_client()
    
    if isinstance(obs, dict):
        obs_dict = obs
    elif hasattr(obs, "model_dump"):
        obs_dict = obs.model_dump()
    else:
        obs_dict = obs.dict()

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
            temperature=0,
            timeout=5
        )

        action = response.choices[0].message.content.strip().upper()

        if action not in ["BUY_CALL", "BUY_PUT", "HOLD", "EXIT"]:
            return "HOLD", f"invalid_action:{action}"

        return action, "llm"

    except Exception:
        price = obs_dict["price"]
        rsi = obs_dict["rsi"]
        position = obs_dict["position"]

        if rsi < 30 and position == "NONE":
            action = "BUY_CALL"
        elif rsi > 70 and position == "NONE":
            action = "BUY_PUT"
        elif position != "NONE":
            action = "EXIT"
        else:
            action = "HOLD"

        return action, "rule_based"


# ===============================
# MAIN EXECUTION (FAIL-SAFE WRAPPER)
# ===============================
if __name__ == "__main__":
    print("SCRIPT_STARTED", flush=True)

    tasks = ["easy", "medium", "hard"]

    try:
        data_path = Path(__file__).resolve().parent / "data" / "NIFTY 50_minute.csv"

        data = load_data(str(data_path))

        for task in tasks:
            print(f"[START] task={task}", flush=True)

            task_config = get_task_config(task)
            env = TradingEnvironment(data, task_config)

            obs = env.reset()
            done = False
            step_count = 0
            total_reward = 0.0

            while not done:
                action, fallback_reason = get_llm_action(obs)

                obs, reward_obj = env.step(action)
                reward = float(reward_obj.value)
                done = reward_obj.done
                total_reward += reward

                print(
                    f"[STEP] task={task} step={step_count} action={action} reward={reward:.6f} done={str(done).lower()} fallback={fallback_reason}",
                    flush=True,
                )

                step_count += 1

                if step_count > 500:
                    print(
                        f"[STEP] task={task} step={step_count} action=HOLD reward=0.000000 done=true fallback=forced_break",
                        flush=True,
                    )
                    break

            final_state = env.state()
            try:
                score = grade_agent(task, final_state)
            except Exception:
                score = 0.0

            print(f"[END] task={task} score={float(score):.6f} steps={step_count}", flush=True)

    except Exception as e:
        # ===============================
        # GLOBAL FAIL-SAFE (CRITICAL)
        # ===============================
        print("FATAL_ERROR:", str(e), flush=True)

        for task in tasks:
            print(f"[START] task={task}", flush=True)
            print(
                f"[STEP] task={task} step=0 action=HOLD reward=0.000000 done=true fallback=fatal_error",
                flush=True,
            )
            print(f"[END] task={task} score=0.000000 steps=0", flush=True)

        sys.exit(0)
