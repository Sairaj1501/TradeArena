from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from server.models import Action
from server.environment import TradingEnvironment
from typing import Optional
from server.models import Observation, Reward
from core.data_processing import load_data
from tasks.tasks import get_task_config
from pydantic import BaseModel
import os
import sys
import io

from inference import get_llm_action
from grader.grader import grade_agent

app = FastAPI()

# ===============================
# 🔁 INIT ENV
# ===============================
try:
    data = load_data("data/NIFTY 50_minute.csv")
    task_config = get_task_config("easy")
    env = TradingEnvironment(data, task_config)
except Exception as e:
    print(f"Failed to initialize environment: {e}")
    env = None

# ===============================
# 🏠 ROOT WEB INTERFACE
# ===============================
@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TradeArena Submission</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; padding: 20px; }
            .container { max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            input[type=password] { width: 100%; padding: 12px; margin: 15px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
            button { background-color: #3498db; color: white; padding: 14px 20px; border: none; border-radius: 4px; cursor: pointer; width: 100%; font-size: 16px; font-weight: bold; }
            button:hover { background-color: #2980b9; }
            pre { background-color: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; min-height: 200px; }
            .loader { border: 4px solid #f3f3f3; border-top: 4px solid #fff; border-radius: 50%; width: 20px; height: 20px; animation: spin 1s linear infinite; display: inline-block; vertical-align: middle; margin-left: 10px; display: none; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 TradeArena Submission Viewer</h1>
            <p>Welcome Judges! Enter your API key (HF_TOKEN) below to execute the <code>inference.py</code> simulation live. The output logs will appear directly on this page.</p>
            
            <label for="apikey"><b>HF_TOKEN / OpenAI API Key</b></label>
            <input type="password" id="apikey" placeholder="sk-proj-..." required>
            
            <button onclick="runBaseline()">Run Evaluation <div class="loader" id="loader"></div></button>
            
            <h3 style="margin-top: 30px;">Terminal Output:</h3>
            <pre id="output">Ready to execute.</pre>
        </div>

        <script>
            async function runBaseline() {
                const apiKey = document.getElementById("apikey").value;
                const output = document.getElementById("output");
                const loader = document.getElementById("loader");
                
                if (!apiKey) {
                    alert("Please provide an OpenAI API Key.");
                    return;
                }

                output.innerText = "Initializing Trading Environment...\nExecuting 100 RL steps against the market...\n(This might take up to 60 seconds)";
                loader.style.display = "inline-block";

                try {
                    
                    const response = await fetch('./run-interactive-baseline', { 
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ api_key: apiKey })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        output.innerText = "Error: " + data.error;
                    } else {
                        output.innerText = data.logs;
                    }
                } catch (err) {
                    output.innerText = "Fetch Error: " + err;
                } finally {
                    loader.style.display = "none";
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content

class LLMRequest(BaseModel):
    api_key: str

# ===============================
# 🤖 INTERACTIVE LLM BASELINE RUNNER
# ===============================
@app.post("/run-interactive-baseline")
def run_interactive_baseline(req: LLMRequest):
    if not env:
        return {"error": "Environment not initialized"}

    if req.api_key == "test":
        return {"logs": "Mock run successful"}

    # Temporarily set the key for this execution using HF_TOKEN
    os.environ["HF_TOKEN"] = req.api_key

    # We redirect standard output (print statements) to a string buffer to send back to the UI
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        print("START: baseline_run")
        task = "easy"
        task_config = get_task_config(task)
        
        # We need to run baseline logic here locally
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        # Note: Depending on logic, this runs get_action from inference
        # If the user switched get_action to RSI rule-based, this will run the rule-based logic!
        # If they need the LLM, they can change inference back to get_llm_action.
        while not done:
            action = get_llm_action(obs)
            print(f"STEP: step={step_count} action={action}")
            
            obs, reward_obj = env.step(action)
            done = reward_obj.done
            reward = reward_obj.value
            
            total_reward += reward
            step_count += 1
            
        final_state = env.state()
        score = grade_agent(task, final_state)
        
        print(f"END: baseline_run score={score}")
        print("-" * 40)
        print(f"Final Equity : {final_state.get('equity', 0):.2f}")
        print(f"Trade Count  : {final_state.get('trade_count')}")
        print(f"Total Reward : {total_reward:.4f}")
        print(f"Hackathon Score: {score}")

    except Exception as e:
        print(f"CRITICAL SIMULATION ERROR: {str(e)}")
    finally:
        # Restore normal printing
        sys.stdout = old_stdout

    return {"logs": new_stdout.getvalue()}

# ===============================
# ❤️ HEALTH CHECK
# ===============================
@app.get("/health")
def health():
    return {"status": "healthy"}

# ===============================
# 🔁 RESET
# ===============================
@app.post("/reset", response_model=Observation)
def reset():
    if not env:
        return {"error": "Environment not initialized"}
    obs = env.reset()
    return obs

# ===============================
# ⚡ STEP
# ===============================
@app.post("/step")
def step(action: Action):
    if not env:
        return {"error": "Environment not initialized"}
        
    obs, reward_obj = env.step(action)

    return {
        "observation": obs.model_dump() if hasattr(obs, 'model_dump') else obs,
        "reward": reward_obj.value,
        "done": reward_obj.done,
        "info": reward_obj.info
    }

# ===============================
# 📊 STATE
# ===============================
@app.get("/state")
def state():
    if not env:
        return {"error": "Environment not initialized"}
    return env.state()
