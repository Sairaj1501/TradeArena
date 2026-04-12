import os
import sys
import random
import numpy as np
from pathlib import Path
from openai import OpenAI

from core.data_processing import load_data
from server.environment import TradingEnvironment
from tasks.tasks import get_task_config

# ===============================
# Reproducibility (IMPORTANT)
# ===============================
random.seed(42)
np.random.seed(42)

# ===============================
# OpenAI Client (STRICT PROXY CONFIG)
# ===============================
def get_openai_client():
    # Scaler/Meta specifically inject these variables for the LiteLLM Proxy
    api_key = os.environ.get("API_KEY") 
    api_base = os.environ.get("API_BASE_URL")

    if not api_key or not api_base:
        # Fallback for local testing only
        api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")

    return OpenAI(
        api_key=api_key,
        base_url=api_base,
    )

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# ===============================
# LLM Action Generator
# ===============================
def get_llm_action(obs):
    try:
        client = get_openai_client()
        
        # Format observation for the LLM
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs

        prompt = f"""
        Expert Options Trader. Data: {obs_dict}.
        Respond with ONLY one word: BUY_CALL, EXIT, or HOLD.
        """

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
            timeout=10
        )

        action = response.choices[0].message.content.strip().upper()
        
        # Strict validation of action string
        if action not in ["BUY_CALL", "EXIT", "HOLD"]:
            return "HOLD", "invalid_llm_response"

        return action, "llm"

    except Exception as e:
        return "HOLD", f"fallback_error:{str(e)[:15]}"


# ===============================
# MAIN EXECUTION (VALIDATOR COMPLIANT)
# ===============================
if __name__ == "__main__":
    print("SCRIPT_STARTED", flush=True)

    # Use UPPERCASE to match typical task config keys
    tasks = ["EASY", "MEDIUM", "HARD"]

    try:
        data_path = Path(__file__).resolve().parent / "data" / "NIFTY 50_minute.csv"
        data = load_data(str(data_path))

        for task_id in tasks:
            # Correct Start Tag
            print(f"[START] task={task_id}", flush=True)

            task_config = get_task_config(task_id)
            env = TradingEnvironment(data, task_config)

            obs = env.reset()
            done = False
            step_count = 0

            # Dynamic length from config or default to 200
            max_steps = task_config.get("episode_length", 200)

            while not done and step_count < max_steps:
                action, source = get_llm_action(obs)

                obs, reward_obj = env.step(action)
                
                # Correct Step Tag
                # Note: reward must be printed exactly as float
                print(
                    f"[STEP] step={step_count} reward={float(reward_obj.value):.6f} action={action}",
                    flush=True,
                )

                done = reward_obj.done
                step_count += 1

            final_state = env.state()
            # Final score based on equity/profit
            score = float(final_state.get('equity', 100000.0))

            # Correct End Tag
            print(f"[END] task={task_id} score={score:.6f} steps={step_count}", flush=True)

    except Exception as e:
        print(f"FATAL_ERROR: {str(e)}", flush=True)
        sys.exit(1)