"""
inference.py — TradeArena OpenEnv Hackathon Submission
=====================================================
Mandatory environment variables (injected by validator):
    API_BASE_URL  : LLM proxy endpoint
    API_KEY       : Proxy API key
    MODEL_NAME    : Model identifier

Optional:
    ENV_BASE_URL  : Base URL of the running environment server
                    (defaults to the deployed HF Space)

Stdout format (strictly followed):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

NOTE: This script talks to the running environment server via HTTP.
      It does NOT import local server modules or load local data files,
      so it works correctly whether executed inside or outside the container.
"""

import os
import sys
import math
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Environment configuration ─────────────────────────────────────────────────
# Points to the running HF Space (or localhost if the server is local).
ENV_BASE_URL: str = os.environ.get(
    "ENV_BASE_URL", "https://sairaj1501-tradearena1.hf.space"
).rstrip("/")

# ── LLM proxy configuration — exactly as injected by the validator ─────────────
API_KEY: str = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "tradearena"
SUCCESS_SCORE_THRESHOLD = 0.1
TASKS = ["easy", "medium", "hard"]
TASK_MAX_STEPS = {"easy": 200, "medium": 300, "hard": 400}

SYSTEM_PROMPT = (
    "You are an expert NIFTY 50 options trader. "
    "Given market observations, choose ONE action from: BUY_CALL, BUY_PUT, HOLD, EXIT. "
    "Reply with EXACTLY one word — the action name only."
)

VALID_ACTIONS = {"BUY_CALL", "BUY_PUT", "HOLD", "EXIT"}

# ── Scoring (inline, no local imports needed) ─────────────────────────────────
def _grade(task_name: str, final_state: Dict[str, Any]) -> float:
    """Normalize final state into a [0, 1] score. Mirrors grader/grader.py."""
    initial_balance = 100_000.0
    equity = float(final_state.get("equity", initial_balance))
    profit_pct = (equity - initial_balance) / initial_balance
    trade_count = int(final_state.get("trade_count", 0))
    mdd = float(final_state.get("max_drawdown", 0.0))

    def smooth(p, shift=0.02, scale=0.05):
        return 0.5 * (math.tanh((p + shift) / scale) + 1.0)

    if task_name == "easy":
        if trade_count == 0:
            return 0.0
        score = smooth(profit_pct, shift=0.02, scale=0.04)

    elif task_name == "medium":
        if 2 <= trade_count <= 20:
            trade_ok = 1.0
        elif trade_count == 0:
            trade_ok = 0.0
        else:
            trade_ok = 0.6
        score = smooth(profit_pct, shift=0.01, scale=0.04) * trade_ok * 0.85

    elif task_name == "hard":
        if mdd <= 0.15:
            mdd_penalty = 1.0
        else:
            mdd_penalty = max(0.0, 1 - (mdd - 0.15) / 0.15)
        score = smooth(profit_pct, shift=0.0, scale=0.05) * mdd_penalty * 0.7

    else:
        score = smooth(profit_pct)

    return round(float(max(0.0, min(score, 1.0))), 4)


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
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── HTTP helpers ──────────────────────────────────────────────────────────────
def _http_reset(task: str, timeout: int = 30) -> Optional[Dict]:
    """Call POST /reset?task=<task> and return the observation dict."""
    try:
        r = requests.post(
            f"{ENV_BASE_URL}/reset",
            params={"task": task},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"[DEBUG] /reset failed: {exc}", flush=True)
        return None


def _http_step(action: str, timeout: int = 30) -> Optional[Dict]:
    """Call POST /step and return the step result dict."""
    try:
        r = requests.post(
            f"{ENV_BASE_URL}/step",
            json={"action": action},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"[DEBUG] /step failed: {exc}", flush=True)
        return None


def _http_state(timeout: int = 15) -> Optional[Dict]:
    """Call GET /state and return the state dict."""
    try:
        r = requests.get(f"{ENV_BASE_URL}/state", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"[DEBUG] /state failed: {exc}", flush=True)
        return None


# ── LLM action ────────────────────────────────────────────────────────────────
def get_llm_action(client: OpenAI, obs: Dict) -> str:
    """Call the LLM proxy and return a valid action string."""
    user_msg = (
        f"Market data: price={obs.get('price', 0):.2f}, "
        f"rsi={obs.get('rsi', 50):.1f}, "
        f"trend={obs.get('trend', 'neutral')}, "
        f"position={obs.get('position', 'none')}, "
        f"equity={obs.get('equity', 100000):.2f}, "
        f"time_to_expiry={obs.get('time_to_expiry', 0)}. "
        "Choose action:"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=15,
        )
        raw = (completion.choices[0].message.content or "").strip().upper()
        for valid in ["BUY_CALL", "BUY_PUT", "EXIT", "HOLD"]:
            if valid in raw:
                return valid
        return "HOLD"
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "HOLD"


# ── Single-task runner ────────────────────────────────────────────────────────
def run_task(client: OpenAI, task_name: str) -> None:
    """Run one full episode for a task via HTTP and emit structured logs."""
    max_steps = TASK_MAX_STEPS.get(task_name, 200)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = _http_reset(task_name)
        if obs is None:
            raise RuntimeError(f"Environment /reset failed for task={task_name}")

        done = False
        for step in range(1, max_steps + 1):
            if done:
                break

            action = get_llm_action(client, obs)
            result = _http_step(action)

            if result is None:
                log_step(step=step, action=action, reward=0.0, done=True, error="step_failed")
                break

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            error_str: Optional[str] = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done, error=error_str)

            if done:
                break

            next_obs = result.get("observation")
            if next_obs is None:
                break
            obs = next_obs

        # Grade
        final_state = _http_state() or {}
        score = _grade(task_name, final_state)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[DEBUG] inference.py started", flush=True)
    print(f"[DEBUG] ENV_BASE_URL={ENV_BASE_URL}", flush=True)
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)

    if not API_KEY:
        print("[FATAL] API_KEY / HF_TOKEN not set — cannot call LLM proxy.", flush=True)
        # Still emit END lines for all tasks so the validator sees output
        for t in TASKS:
            log_start(task=t, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])
        sys.exit(1)

    # Wait for the environment server to be ready (relevant when server
    # starts in the same container orchestration run).
    for attempt in range(6):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
            if r.status_code == 200:
                print(f"[DEBUG] Environment server ready at {ENV_BASE_URL}", flush=True)
                break
        except Exception:
            pass
        print(f"[DEBUG] Waiting for env server... attempt {attempt + 1}/6", flush=True)
        time.sleep(5)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    for task_name in TASKS:
        run_task(client, task_name)

    sys.exit(0)