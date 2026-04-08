from fastapi import BackgroundTasks
import asyncio
from baseline.run_baseline import get_llm_action

@app.post("/run-baseline")
def run_baseline_simulation(background_tasks: BackgroundTasks):
    """
    Triggers the baseline logic in the background so the HTTP request doesn't timeout.
    """
    def simulation_loop():
        # Replicate your run_baseline.py episode loop logic here
        obs = env.reset()
        done = False
        while not done:
            action = get_llm_action(obs)
            obs, reward = env.step(action)
            if reward.done:
                done = True
                print("Simulation finished")
                
    background_tasks.add_task(simulation_loop)
    return {"message": "Baseline simulation started in the background."}