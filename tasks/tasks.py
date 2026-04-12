def get_task_config(task_name):
    # Convert to lowercase to avoid "EASY" vs "easy" crashes
    task_name = task_name.lower()

    if task_name == "easy":
        return {
            "market_type": "trending",
            "volatility": "low", 
            "episode_length": 200, # FIXED: Minimum 200 required
            "stop_loss_threshold": 0.10 
        }
    elif task_name == "medium":
        return {
            "market_type": "sideways",
            "volatility": "medium",
            "episode_length": 300, # FIXED: Minimum 300 required
            "stop_loss_threshold": 0.05
        }
    elif task_name == "hard":
        return {
            "market_type": "volatile",
            "volatility": "high",
            "episode_length": 400, # Standard requirement
            "stop_loss_threshold": 0.02 
        }
    
    # Fallback to EASY instead of crashing
    return {
        "market_type": "trending",
        "volatility": "low",
        "episode_length": 200,
        "stop_loss_threshold": 0.10
    }