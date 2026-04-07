import os
import json
from openai import OpenAI
from core.data_processing import load_data
from env.trading_env import TradingEnv
from tasks.tasks import get_task_config
from grader.grader import grade_agent  # Ensure this matches your filename

# 1. Setup OpenAI Client (Requirement: Reads from environment variables)
# Set your key in terminal: export 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_llm_action(obs):
    """Sends the market observation to GPT to decide the next move."""
    
    prompt = f"""
    You are an expert options trader. Based on the following market data, 
    choose the best action: BUY_CALL, BUY_PUT, HOLD, or EXIT.
    
    Observation:
    - Current Price: {obs.price}
    - RSI: {obs.rsi}
    - Trend: {obs.trend}
    - Position: {obs.position}
    - Equity: {obs.equity}
    
    Respond with ONLY the action name in uppercase.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10
    )
    
    action = response.choices[0].message.content.strip().upper()

    if action not in ["BUY_CALL", "BUY_PUT", "HOLD", "EXIT"]:
        return "HOLD"

    return action

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading data and initializing environment...")
    data = load_data("data/NIFTY 50_minute.csv")
    
    # Task selection: easy, medium, or hard
    task = "easy" 
    task_config = get_task_config(task)
    env = TradingEnv(data, task_config)

    obs = env.reset()
    done = False
    total_reward = 0

    print(f"Starting Baseline Inference on Task: {task}")

    while not done:
        # 2. Get action from LLM instead of RSI rules
        action = get_llm_action(obs)
        
        # 3. Step the environment
        obs, reward, done, info = env.step(action)
        total_reward += reward

    # 4. Final Grading
    final_state = env.state()
    # This now passes the WHOLE state dict to your grader as requested
    score = grade_agent(task, final_state)

    print("-" * 30)
    print(f"Final Equity: {final_state['equity']:.2f}")
    print(f"Trade Count: {final_state['trade_count']}")
    print(f"Hackathon Score: {score}")
    print("-" * 30)