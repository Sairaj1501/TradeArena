def get_task_config(task_name):
    if task_name == "easy":
        return {
            "market_type": "trending",
            "volatility": "low", # 0.5 scale
            "episode_length": 100, # Shorter is easier to learn
            "stop_loss_threshold": 0.10 # Very forgiving
        }
    elif task_name == "medium":
        return {
            "market_type": "sideways",
            "volatility": "medium", # 1.2 scale
            "episode_length": 200,
            "stop_loss_threshold": 0.05
        }
    elif task_name == "hard":
        return {
            "market_type": "volatile",
            "volatility": "high", # 3.0 scale
            "episode_length": 400, # Longer requires more stamina
            "stop_loss_threshold": 0.02 # Very strict
        }
    raise ValueError("Invalid task")