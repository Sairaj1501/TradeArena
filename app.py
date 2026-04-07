from core.data_processing import load_data
from env.trading_env import TradingEnv
from tasks.tasks import get_task_config

data = load_data("data/NIFTY 50_minute.csv")
task = "easy"  # or medium / hard
task_config = get_task_config(task)

env = TradingEnv(data, task_config)

obs = env.reset()

print("Initial Observation:", obs)