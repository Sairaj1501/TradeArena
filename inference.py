import os
import sys
import random
import numpy as np
from pathlib import Path
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
)

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
def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def get_openai_client():
    # Checklist: "Must use OpenAI Client for all LLM calls using above variables"
    # api_key = _env_or_default("HF_TOKEN", "")
    api_key = _env_or_default("HF_TOKEN", "")

    if not _env_or_default("HF_TOKEN", ""):
        # Fallback logic (NO API DEPENDENCY)

        close_price = obs.get("close", 0)
        sma = obs.get("sma_20", close_price)
        rsi = obs.get("rsi", 50)

        if rsi < 30 and close_price > sma:
            return "BUY", "fallback_rsi_oversold"
        elif rsi > 70 and close_price < sma:
            return "SELL", "fallback_rsi_overbought"
        else:
            return "HOLD", "fallback_no_signal"
    base_url = _env_or_default("API_BASE_URL", "https://api.openai.com/v1")

    return OpenAI(
        api_key=api_key if api_key else "dummy_key_if_missing",
        base_url=base_url,
        timeout=10.0,
        max_retries=1,
    )

MODEL_NAME = _env_or_default("MODEL_NAME", "gpt-4o-mini")

# ===============================
# LLM Action Generator
# ===============================
def get_llm_action(obs):
    """Generate action using LLM with strict validation.

    Returns:
        tuple[str, str]: (action, fallback_reason)
    """

    if not _env_or_default("HF_TOKEN", ""):
        return "HOLD", "missing_hf_token"

    client = get_openai_client()
    
    # Support both dict observations and Pydantic models (v1/v2)
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
            temperature=0
        )

        action = response.choices[0].message.content.strip().upper()

    except AuthenticationError:
        return "HOLD", "auth_error"
    except RateLimitError:
        return "HOLD", "rate_limited"
    except APITimeoutError:
        return "HOLD", "timeout"
    except APIConnectionError:
        return "HOLD", "connection_error"
    except NotFoundError:
        return "HOLD", "model_not_found"
    except BadRequestError:
        return "HOLD", "bad_request"
    except Exception:
        return "HOLD", "llm_failure"

    if action not in ["BUY_CALL", "BUY_PUT", "HOLD", "EXIT"]:
        return "HOLD", f"invalid_action:{action}"

    return action, "none"


# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    data_path = Path(__file__).resolve().parent / "data" / "NIFTY 50_minute.csv"

    try:
        data = load_data(str(data_path))
    except Exception as e:
        # Emit structured blocks even on fatal data load failures.
        for task in tasks:
            print(f"[START] task={task}", flush=True)
            print(
                f"[STEP] task={task} step=0 action=HOLD reward=0.000000 done=true fallback=data_load_failed",
                flush=True,
            )
            print(f"[END] task={task} score=0.000000 steps=0", flush=True)
        sys.exit(0)

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
