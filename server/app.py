from fastapi import FastAPI
from server.models import Action
from server.environment import TradingEnvironment
from typing import Optional
from server.models import Observation, Reward
from core.data_processing import load_data
from tasks.tasks import get_task_config
from baseline.run_baseline import get_llm_action

app = FastAPI()

# ===============================
# 🔁 INIT ENV
# ===============================
data = load_data("data/NIFTY 50_minute.csv")
task_config = get_task_config("easy")

env = TradingEnvironment(data, task_config)


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
    obs = env.reset()
    return obs


# ===============================
# 🤖 GET LLM ACTION
# ===============================
@app.post("/act")
def act(obs: Observation):
    # This endpoint lets the judges ask your baseline agent what to do!
    action = get_llm_action(obs)
    return {"action": action}

# ===============================
# ⚡ STEP
# ===============================
@app.post("/step")
def step(action: Action):
    obs, reward = env.step(action)

    return {
        "observation": obs.dict() if obs else None,
        "reward": reward.value,
        "done": reward.done,
        "info": reward.info
    }


# ===============================
# 📊 STATE
# ===============================
@app.get("/state")
def state():
    return env.state()
