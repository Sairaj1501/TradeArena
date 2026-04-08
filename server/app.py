from fastapi import FastAPI
from server.models import Action
# from server.environment import TradingEnvironment
from server.environment import TradingEnvironment
from typing import Optional
from server.models import Observation, Reward
from core.data_processing import load_data
from tasks.tasks import get_task_config

app = FastAPI()

# ===============================
# 🔁 INIT ENV
# ===============================
data = load_data("data/NIFTY 50_minute.csv")
task_config = get_task_config("easy")

env = TradingEnvironment(data, task_config)


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
