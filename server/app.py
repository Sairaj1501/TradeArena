"""
server/app.py — FastAPI server for TradeArena OpenEnv environment
Provides: GET /, POST /reset, POST /step, GET /state, GET /health
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from server.models import Action, Observation, Reward
from server.environment import TradingEnvironment
from typing import Optional
from core.data_processing import load_data
from tasks.tasks import get_task_config
from grader.grader import grade_agent
from pydantic import BaseModel
import os
import io
import threading
import contextlib

# ── NOTE: NO import from inference.py — that would be a circular import ──────
# The web UI runs its own inline LLM call.

app = FastAPI(title="TradeArena OpenEnv")
RUN_LOCK = threading.Lock()

# ── Initialise environment (globally shared for HTTP endpoints) ───────────────
try:
    _data = load_data("data/NIFTY 50_minute.csv")
    _task_config = get_task_config("easy")
    env = TradingEnvironment(_data, _task_config)
    print("[INFO] Environment initialised successfully.", flush=True)
except Exception as _e:
    print(f"[ERROR] Failed to initialise environment: {_e}", flush=True)
    env = None
    _data = None


# ── Inline LLM helper for web UI (avoids circular import) ────────────────────
def _web_ui_llm_action(obs_dict: dict, api_key: str, api_base: str, model: str) -> tuple[str, str]:
    """Call LLM proxy from web UI context. Returns (action, reason)."""
    from openai import OpenAI  # local import is fine here — no circularity
    try:
        client = OpenAI(api_key=api_key, base_url=api_base)
        user_msg = (
            f"Market: price={obs_dict.get('price', 0):.2f}, "
            f"rsi={obs_dict.get('rsi', 50):.1f}, "
            f"trend={obs_dict.get('trend', 'neutral')}, "
            f"position={obs_dict.get('position', 'none')}, "
            f"equity={obs_dict.get('equity', 100000):.2f}. "
            "Action (BUY_CALL/BUY_PUT/HOLD/EXIT):"
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a NIFTY 50 options trader. Reply with ONE word: BUY_CALL, BUY_PUT, HOLD, or EXIT."},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=15,
        )
        raw = (completion.choices[0].message.content or "").strip().upper()
        for valid in ["BUY_CALL", "BUY_PUT", "EXIT", "HOLD"]:
            if valid in raw:
                return valid, "llm"
        return "HOLD", "parse_fallback"
    except Exception as exc:
        return "HOLD", f"error: {exc}"


# ── Root Web Interface ────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def read_root():
    default_api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    default_model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TradeArena Submission</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; padding: 20px; }
            .container { max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            input[type=password], input[type=text] { width: 100%; padding: 12px; margin: 8px 0 15px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
            label { font-weight: bold; }
            button { background-color: #3498db; color: white; padding: 14px 20px; border: none; border-radius: 4px; cursor: pointer; width: 100%; font-size: 16px; font-weight: bold; }
            button:hover { background-color: #2980b9; }
            pre { background-color: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; min-height: 200px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 TradeArena Submission Viewer</h1>
            <p>Enter your proxy credentials to run a live evaluation. Output logs will appear below.</p>

            <label for="apikey">HF_TOKEN / API Key</label>
            <input type="password" id="apikey" placeholder="hf_..." required>

            <label for="apiBase">API_BASE_URL</label>
            <input type="text" id="apiBase" value="%%DEFAULT_API_BASE%%">

            <label for="modelName">MODEL_NAME</label>
            <input type="text" id="modelName" value="%%DEFAULT_MODEL%%">

            <button id="runBtn" onclick="runBaseline()">▶ Run Evaluation</button>

            <h3 style="margin-top: 30px;">Terminal Output:</h3>
            <pre id="output">Ready to execute.</pre>
        </div>

        <script>
            async function runBaseline() {
                const apiKey = document.getElementById("apikey").value;
                const apiBase = document.getElementById("apiBase").value;
                const modelName = document.getElementById("modelName").value;
                const output = document.getElementById("output");
                const runBtn = document.getElementById("runBtn");

                if (!apiKey) { alert("Please provide an API Key."); return; }

                output.innerText = "Running...";
                runBtn.disabled = true;

                try {
                    const response = await fetch('/run-interactive-baseline', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ api_key: apiKey, api_base_url: apiBase, model_name: modelName })
                    });
                    const data = await response.json();
                    output.innerText = data.error ? "Error: " + data.error : data.logs;
                } catch (err) {
                    output.innerText = "Fetch Error: " + err;
                } finally {
                    runBtn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """
    return (
        html_content
        .replace("%%DEFAULT_API_BASE%%", default_api_base)
        .replace("%%DEFAULT_MODEL%%", default_model)
    )


class LLMRequest(BaseModel):
    api_key: str
    api_base_url: Optional[str] = None
    model_name: Optional[str] = None


# ── Interactive LLM Baseline Runner (web UI endpoint) ────────────────────────
@app.post("/run-interactive-baseline")
def run_interactive_baseline(req: LLMRequest):
    if not env or _data is None:
        return {"error": "Environment not initialized"}

    if not RUN_LOCK.acquire(blocking=False):
        return {"error": "Another run is already in progress. Please wait and retry."}

    api_base = (req.api_base_url or "").strip() or os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model = (req.model_name or "").strip() or os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    new_stdout = io.StringIO()

    try:
        with contextlib.redirect_stdout(new_stdout):
            print(f"[START] task=easy env=tradearena model={model}")
            config = get_task_config("easy")
            episode_env = TradingEnvironment(_data, config)
            obs = episode_env.reset()
            obs_dict = obs if isinstance(obs, dict) else (obs.model_dump() if hasattr(obs, "model_dump") else dict(obs))

            done = False
            step_count = 0
            total_reward = 0.0
            rewards = []

            while not done and step_count < config.get("episode_length", 200):
                action_str, reason = _web_ui_llm_action(obs_dict, req.api_key, api_base, model)
                action_obj = Action(action=action_str)
                result_obs, reward_obj = episode_env.step(action_obj)

                reward = float(reward_obj.value)
                done = reward_obj.done
                total_reward += reward
                step_count += 1
                rewards.append(reward)

                print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")

                if done or result_obs is None:
                    break
                obs_dict = result_obs.model_dump() if hasattr(result_obs, "model_dump") else dict(result_obs)

            final_state = episode_env.state()
            score = float(grade_agent("easy", final_state))
            score = max(0.0, min(score, 1.0))
            success = score >= 0.1
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(f"[END] success={str(success).lower()} steps={step_count} score={score:.3f} rewards={rewards_str}")
            print("-" * 40)
            print(f"Final Equity : {final_state.get('equity', 0):.2f}")
            print(f"Trade Count  : {final_state.get('trade_count')}")
            print(f"Total Reward : {total_reward:.4f}")
            print(f"Hackathon Score: {score}")

    except Exception as e:
        with contextlib.redirect_stdout(new_stdout):
            print(f"CRITICAL ERROR: {str(e)}")
    finally:
        RUN_LOCK.release()

    return {"logs": new_stdout.getvalue()}


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "env_ready": env is not None}


# ── Reset ─────────────────────────────────────────────────────────────────────
@app.post("/reset", response_model=Observation)
def reset():
    if not env:
        return Observation(
            price=0.0, rsi=50.0, trend="neutral",
            time_to_expiry=0, position="none",
            entry_price=None, balance=0.0, equity=0.0
        )
    obs = env.reset()
    # env.reset() returns a raw dict — convert to Observation model
    if isinstance(obs, dict):
        return Observation(**obs)
    return obs


# ── Step ──────────────────────────────────────────────────────────────────────
@app.post("/step")
def step(action: Action):
    if not env:
        return {"error": "Environment not initialized"}

    result_obs, reward_obj = env.step(action)

    return {
        "observation": result_obs.model_dump() if result_obs and hasattr(result_obs, "model_dump") else None,
        "reward": reward_obj.value,
        "done": reward_obj.done,
        "info": reward_obj.info,
    }


# ── State ─────────────────────────────────────────────────────────────────────
@app.get("/state")
def state():
    if not env:
        return {"error": "Environment not initialized"}
    return env.state()


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main():
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
