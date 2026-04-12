"""
inference.py — TradeArena OpenEnv Hackathon Submission
=====================================================
Mandatory environment variables (injected by validator):
    API_BASE_URL  : LLM proxy endpoint
    API_KEY       : Proxy API key
    MODEL_NAME    : Model identifier

Stdout format (strictly followed):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import sys
import random
import math
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from core.data_processing import load_data
from server.environment import TradingEnvironment
from server.models import Action
from tasks.tasks import get_task_config
from grader.grader import grade_agent

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────
# Use the env vars EXACTLY as injected by the validator.
# The sample reference uses HF router as the default — we mirror that.
API_KEY: str = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "tradearena"
MAX_STEPS = 400          # upper bound — tasks terminate earlier via episode_length
TEMPERATURE = 0.0
MAX_TOKENS = 15
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = (
    "You are an expert NIFTY 50 options trader. "
    "Given market observations, choose ONE action from: BUY_CALL, BUY_PUT, HOLD, EXIT. "
    "Reply with EXACTLY one word — the action name only."
)


# ── Stdout helpers ────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM client ───────────────────────────────────────────────────────────────
def _make_client() -> OpenAI:
    """Always initialise OpenAI client with proxy vars from environment."""
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def get_llm_action(client: OpenAI, obs_dict: dict) -> str:
    """Call the LLM proxy and return a valid action string."""
    user_msg = (
        f"Market data: price={obs_dict.get('price', 0):.2f}, "
        f"rsi={obs_dict.get('rsi', 50):.1f}, "
        f"trend={obs_dict.get('trend', 'neutral')}, "
        f"position={obs_dict.get('position', 'none')}, "
        f"equity={obs_dict.get('equity', 100000):.2f}, "
        f"time_to_expiry={obs_dict.get('time_to_expiry', 0)}. "
        "Choose action:"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip().upper()
        # Accept only valid actions
        for valid in ["BUY_CALL", "BUY_PUT", "EXIT", "HOLD"]:
            if valid in raw:
                return valid
        return "HOLD"
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "HOLD"


# ── Single task runner ────────────────────────────────────────────────────────
def run_task(client: OpenAI, data: list, task_name: str) -> None:
    """Run one full episode for a given task and emit structured logs."""
    config = get_task_config(task_name)
    env = TradingEnvironment(data, config)
    episode_max = config.get("episode_length", 200)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        obs_dict = obs if isinstance(obs, dict) else (obs.model_dump() if hasattr(obs, "model_dump") else dict(obs))

        done = False
        for step in range(1, episode_max + 1):
            if done:
                break

            action_str = get_llm_action(client, obs_dict)

            # Step the environment
            action_obj = Action(action=action_str)
            result_obs, reward_obj = env.step(action_obj)

            reward = float(reward_obj.value)
            done = reward_obj.done
            last_error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=last_error)

            if done or result_obs is None:
                break

            obs_dict = result_obs.model_dump() if hasattr(result_obs, "model_dump") else dict(result_obs)

        # Grade with normalized [0, 1] score
        final_state = env.state()
        score = float(grade_agent(task_name, final_state))
        score = max(0.0, min(score, 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        last_error = str(exc)
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[DEBUG] inference.py started", flush=True)

    # Validate proxy vars are present (fail loudly so validator can diagnose)
    if not API_KEY:
        print("[FATAL] API_KEY / HF_TOKEN not set — cannot call LLM proxy.", flush=True)
        sys.exit(1)

    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)

    # Build single OpenAI client reused across all tasks
    client = _make_client()

    # Load market data once
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "NIFTY 50_minute.csv"
    try:
        data = load_data(str(data_path))
        print(f"[DEBUG] Loaded {len(data)} data rows", flush=True)
    except Exception as exc:
        print(f"[FATAL] Could not load data: {exc}", flush=True)
        sys.exit(1)

    # Run all 3 tasks sequentially
    for task_name in TASKS:
        run_task(client, data, task_name)

    sys.exit(0)