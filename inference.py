import os
import sys
import random
import numpy as np
from pathlib import Path
from openai import OpenAI

# Required imports
from core.data_processing import load_data
from server.environment import TradingEnvironment
from tasks.tasks import get_task_config

# Reproducibility
random.seed(42)
np.random.seed(42)

def get_openai_client():
    # STRICT: Use exactly what the validator injects.
    # Do NOT use default URLs or fallbacks like HF_TOKEN here.
    api_key = os.environ.get("API_KEY")
    api_base = os.environ.get("API_BASE_URL")
    
    if not api_key or not api_base:
        print(f"FATAL: Missing proxy variables. API_KEY: {bool(api_key)}, BASE: {bool(api_base)}", flush=True)
        return None
    
    return OpenAI(api_key=api_key, base_url=api_base)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

def get_llm_action(obs):
    client = get_openai_client()
    if not client:
        # If no client, we must FAIL, not fallback to rules.
        raise RuntimeError("No API Client available for proxy hit.")

    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
    prompt = f"Trader. Data: {obs_dict}. Action: BUY_CALL, EXIT, HOLD. ONE WORD ONLY."

    # This MUST hit the proxy.
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,
        timeout=15 # Increased timeout for proxy stability
    )
    action = response.choices[0].message.content.strip().upper()
    return (action if action in ["BUY_CALL", "EXIT", "HOLD"] else "HOLD")

if __name__ == "__main__":
    print("SCRIPT_STARTED", flush=True)
    tasks = ["EASY", "MEDIUM", "HARD"]

    try:
        base_dir = Path(__file__).resolve().parent
        data_path = base_dir / "data" / "NIFTY 50_minute.csv"
        data = load_data(str(data_path))

        for task_id in tasks:
            # Tags only print if execution is actually happening.
            print(f"[START] task={task_id}", flush=True)
            
            config = get_task_config(task_id)
            env = TradingEnvironment(data, config)
            obs = env.reset()
            done, steps = False, 0
            max_steps = config.get("episode_length", 200)

            while not done and steps < max_steps:
                # If this fails, the script CRASHES, which is better than faking success.
                action = get_llm_action(obs)
                obs, reward_obj = env.step(action)
                
                print(f"[STEP] step={steps} reward={float(reward_obj.value):.6f} action={action}", flush=True)
                
                done = reward_obj.done
                steps += 1

            final_state = env.state()
            score = float(final_state.get('equity', 100000.0))
            print(f"[END] task={task_id} score={score:.6f} steps={steps}", flush=True)

    except Exception as e:
        print(f"UNHANDLED_EXCEPTION: {e}", flush=True)
        # We do NOT print fake [END] tags here anymore.
        sys.exit(1) # Tell the validator something went wrong

    sys.exit(0)