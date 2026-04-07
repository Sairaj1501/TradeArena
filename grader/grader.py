def grade_agent(task_name, final_stats):
    """
    Produces a score between 0.0 and 1.0 based on task-specific KPIs.
    """
    initial_balance = 100000
    net_profit = final_stats['equity'] - initial_balance
    profit_pct = net_profit / initial_balance
    
    if task_name == "easy":
        # Objective: Just don't lose money and execute at least one trade.
        if final_stats['trade_count'] == 0: return 0.0
        score = min(max(profit_pct + 0.05, 0) / 0.1, 1.0)
        
    elif task_name == "medium":
        # Objective: Profitability with moderate trade frequency.
        # Penalty if trades > 20 or trades < 2.
        trade_count = final_stats['trade_count']
        trade_ok = 1.0 if 2 <= trade_count <= 20 else 0.5
        profit_score = min(max(profit_pct, 0) / 0.15, 1.0)
        score = profit_score * trade_ok
        
    elif task_name == "hard":
        # Objective: Survival + Profit. Max Drawdown must stay below 15%.
        mdd = final_stats.get('max_drawdown', 0)
        mdd_penalty = 1.0 if mdd < 0.15 else max(0, 1 - (mdd - 0.15) / 0.10)
        profit_score = min(max(profit_pct, 0) / 0.20, 1.0)
        score = profit_score * mdd_penalty
        
    return round(float(max(0, min(score, 1.0))), 2)