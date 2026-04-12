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

Timing budget (validator limit = 20 min):
    Health check : up to  90 s  (6 × 15 s)
    Per task     : 15 steps × (8 s LLM + 5 s HTTP) = ~3 min
    3 tasks      : ~9 min
    Total worst  : ~10.5 min  ← well within 20 min budget
"""

import os
import sys
import math
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Timing constants ──────────────────────────────────────────────────────────
LLM_TIMEOUT_S   = 8    # per LLM call (seconds)
HTTP_TIMEOUT_S  = 8    # per /reset or /step call (seconds)
HEALTH_TIMEOUT_S = 10  # per /health poll
MAX_HEALTH_ATTEMPTS = 6
HEALTH_SLEEP_S  = 15   # sleep between health retries
# Global wall-clock guard — bail-out this many seconds before validator's limit
GLOBAL_LIMIT_S  = 17 * 60   # 17 minutes
MAX_STEPS_PER_TASK = 15     # 15 × 3 tasks = 45 total LLM calls

# ── Environment / LLM configuration ──────────────────────────────────────────
ENV_BASE_URL: str = os.environ.get(
    "ENV_BASE_URL", "https://sairaj1501-tradearena1.hf.space"
).rstrip("/")

API_KEY: str      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK              = "tradearena"
SUCCESS_SCORE_THRESHOLD = 0.1
TASKS                  = ["easy", "medium", "hard"]

SYSTEM_PROMPT = (
    "You are a NIFTY 50 options trader. "
    "Reply with ONE word only: BUY_CALL, BUY_PUT, HOLD, or EXIT."
)

# ── Wall-clock start time (set in __main__) ───────────────────────────────────
_t0: float = 0.0


def _elapsed() -> float:
    return time.time() - _t0


def _time_remaining() -> float:
    return GLOBAL_LIMIT_S - _elapsed()


def _over_budget() -> bool:
    return _elapsed() >= GLOBAL_LIMIT_S


# ── Inline scoring (no local imports) ─────────────────────────────────────────
def _grade(task_name: str, final_state: Dict[str, Any]) -> float:
    """Return a normalized [0, 1] score from final episode state."""
    initial_balance = 100_000.0
    equity      = float(final_state.get("equity", initial_balance))
    profit_pct  = (equity - initial_balance) / initial_balance
    trade_count = int(final_state.get("trade_count", 0))
    mdd         = float(final_state.get("max_drawdown", 0.0))

    def _smooth(p: float, shift: float = 0.02, scale: float = 0.05) -> float:
        return 0.5 * (math.tanh((p + shift) / scale) + 1.0)

    if task_name == "easy":
        score = 0.0 if trade_count == 0 else _smooth(profit_pct, 0.02, 0.04)
    elif task_name == "medium":
        trade_ok = 1.0 if 2 <= trade_count <= 20 else (0.0 if trade_count == 0 else 0.6)
        score = _smooth(profit_pct, 0.01, 0.04) * trade_ok * 0.85
    elif task_name == "hard":
        mdd_penalty = 1.0 if mdd <= 0.15 else max(0.0, 1 - (mdd - 0.15) / 0.15)
        score = _smooth(profit_pct, 0.0, 0.05) * mdd_penalty * 0.7
    else:
        score = _smooth(profit_pct)

    return round(float(max(0.0, min(score, 1.0))), 4)


# ── Stdout helpers ─────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── HTTP helpers (tight timeouts) ─────────────────────────────────────────────
def _http_reset(task: str) -> Optional[Dict]:
    try:
        r = requests.post(
            f"{ENV_BASE_URL}/reset",
            params={"task": task},
            timeout=HTTP_TIMEOUT_S,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"[DEBUG] /reset failed: {exc}", flush=True)
        return None


def _http_step(action: str) -> Optional[Dict]:
    try:
        r = requests.post(
            f"{ENV_BASE_URL}/step",
            json={"action": action},
            timeout=HTTP_TIMEOUT_S,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"[DEBUG] /step failed: {exc}", flush=True)
        return None


def _http_state() -> Optional[Dict]:
    try:
        r = requests.get(f"{ENV_BASE_URL}/state", timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"[DEBUG] /state failed: {exc}", flush=True)
        return None


# ── LLM action (with hard timeout) ────────────────────────────────────────────
def get_llm_action(client: OpenAI, obs: Dict) -> str:
    """Call LLM proxy with a strict timeout. Falls back to HOLD on any error."""
    user_msg = (
        f"price={obs.get('price', 0):.1f} rsi={obs.get('rsi', 50):.0f} "
        f"trend={obs.get('trend', 'neutral')} position={obs.get('position', 'none')} "
        f"equity={obs.get('equity', 100000):.0f}. Action?"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=10,
            timeout=LLM_TIMEOUT_S,
        )
        raw = (completion.choices[0].message.content or "").strip().upper()
        for valid in ["BUY_CALL", "BUY_PUT", "EXIT", "HOLD"]:
            if valid in raw:
                return valid
    except Exception as exc:
        print(f"[DEBUG] LLM timeout/error: {exc}", flush=True)
    return "HOLD"


# ── Single-task runner ─────────────────────────────────────────────────────────
def run_task(client: OpenAI, task_name: str) -> None:
    """Run one episode (≤ MAX_STEPS_PER_TASK steps) and emit structured logs."""
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Check global budget before starting a new task
        if _over_budget():
            raise RuntimeError("Global time budget exhausted before task start")

        obs = _http_reset(task_name)
        if obs is None:
            raise RuntimeError(f"/reset failed for task={task_name}")

        done = False
        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if done or _over_budget():
                break

            action = get_llm_action(client, obs)
            result = _http_step(action)

            if result is None:
                log_step(step=step, action=action, reward=0.0, done=True, error="step_failed")
                break

            reward = float(result.get("reward", 0.0))
            done   = bool(result.get("done", False))
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done, error=None)

            if done:
                break
            obs = result.get("observation") or obs  # keep last good obs if None

        final_state = _http_state() or {}
        score   = _grade(task_name, final_state)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        print(f"[DEBUG] elapsed={_elapsed():.1f}s remaining={_time_remaining():.1f}s", flush=True)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _t0 = time.time()

    print("[DEBUG] inference.py started", flush=True)
    print(f"[DEBUG] ENV_BASE_URL={ENV_BASE_URL}", flush=True)
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] MAX_STEPS_PER_TASK={MAX_STEPS_PER_TASK} | GLOBAL_LIMIT={GLOBAL_LIMIT_S}s", flush=True)

    # ── Validate API credentials ───────────────────────────────────────────────
    if not API_KEY:
        print("[FATAL] API_KEY / HF_TOKEN not set — cannot call LLM proxy.", flush=True)
        for t in TASKS:
            log_start(task=t, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])
        sys.exit(1)

    # ── Wait for environment server (HF Space may need a cold-start wake-up) ──
    server_ready = False
    for attempt in range(MAX_HEALTH_ATTEMPTS):
        if _over_budget():
            break
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=HEALTH_TIMEOUT_S)
            if r.status_code == 200:
                print(f"[DEBUG] Env server ready (attempt {attempt + 1})", flush=True)
                server_ready = True
                break
        except Exception as exc:
            print(f"[DEBUG] Health check attempt {attempt + 1}/{MAX_HEALTH_ATTEMPTS}: {exc}", flush=True)

        if attempt < MAX_HEALTH_ATTEMPTS - 1:
            # Only sleep if there's time left and more retries remain
            sleep_time = min(HEALTH_SLEEP_S, max(0, _time_remaining() - 60))
            if sleep_time > 0:
                time.sleep(sleep_time)

    if not server_ready:
        print("[WARN] Env server did not respond to health check — proceeding anyway.", flush=True)

    # ── Build LLM client ───────────────────────────────────────────────────────
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    # ── Run all tasks (stops early if global budget is exceeded) ──────────────
    for task_name in TASKS:
        if _over_budget():
            print(f"[WARN] Skipping task {task_name} — global time limit reached.", flush=True)
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            continue
        run_task(client, task_name)

    print(f"[DEBUG] All tasks done. Total elapsed: {_elapsed():.1f}s", flush=True)
    sys.exit(0)