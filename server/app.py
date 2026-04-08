from fastapi import FastAPI, BackgroundTasks
from server.models import Action
from server.environment import TradingEnvironment
from typing import Optional
from server.models import Observation, Reward
from core.data_processing import load_data
from tasks.tasks import get_task_config

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
# 🏠 ROOT
# ===============================
@app.get("/")
def read_root():
    return {"message": "Welcome to the TradeArena Agent API", "docs": "/docs"}

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
# 🤖 RULE-BASED ACTION GENERATOR
# ===============================
def run_rule_based_simulation():
    if not env:
        print("Simulation aborted: Environment not initialized")
        return
        
    print("Simulation started")
    obs = env.reset()
    done = False
    
    while not done:
        # Pydantic or dict check
        rsi = obs.rsi if hasattr(obs, 'rsi') else obs.get('rsi', 50)
        
        # Rule-based logic
        if rsi < 30:
            action = "BUY_CALL"
        elif rsi > 70:
            action = "BUY_PUT"
        else:
            action = "HOLD"
            
        obs, reward_obj = env.step(action)
        done = reward_obj.done

    print("Simulation finished")

# ===============================
# 🚀 RUN BASELINE BACKGROUND
# ===============================
@app.post("/run-baseline")
def run_baseline_simulation(background_tasks: BackgroundTasks):
    """
    Triggers the baseline logic in the background so the HTTP request doesn't timeout.
    """
    background_tasks.add_task(run_rule_based_simulation)
    return {"message": "Baseline simulation started in the background."}

# ===============================
# ⚡ STEP
# ===============================
@app.post("/step")
def step(action: Action):
    if not env:
        return {"error": "Environment not initialized"}
        
    obs, reward_obj = env.step(action)

    return {
        "observation": obs.dict() if hasattr(obs, 'dict') else obs,
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
