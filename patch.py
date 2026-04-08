import sys
with open("inference.py", "r", encoding="utf-8") as f:
    content = f.read()

idx = content.find("if __name__ == \`"__main__\`":")
if idx != -1:
    new_content = content[:idx] + """if __name__ == "__main__":
    from core.data_processing import load_data
    from server.environment import TradingEnvironment
    from tasks.tasks import get_task_config
    from grader.grader import grade_agent

    try:
        data = load_data("data/NIFTY 50_minute.csv")
    except:
        print("[END] score=0.0")
        sys.exit(0)
    
    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        print(f"[START] {task}")
        try:
            task_config = get_task_config(task)
            env = TradingEnvironment(data, task_config)

            obs = env.reset()
            done = False
            step_count = 0

            while not done:
                action = get_llm_action(obs)
                print(f"[STEP] action={action}")

                obs, reward_obj = env.step(action)
                done = reward_obj.done
                step_count += 1
                if step_count > 300: break

            final_state = env.state()
            try:
                score = grade_agent(task, final_state)
            except Exception:
                score = 0.0

            print(f"[END] score={score}")
        except Exception as e:
            print(f"[END] score=0.0")
"""
    with open("inference.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Patched inference.py")
