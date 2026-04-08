import math

def _smooth_score(profit_pct, shift=0.02, scale=0.05):
    return 0.5 * (math.tanh((profit_pct + shift) / scale) + 1.0)


def grade_agent(task_name, final_stats):
    initial_balance = 100000
    net_profit = final_stats['equity'] - initial_balance
    profit_pct = net_profit / initial_balance

    trade_count = final_stats.get('trade_count', 0)
    mdd = final_stats.get('max_drawdown', 0)

    if task_name == "easy":
        if trade_count == 0:
            return 0.0

        # ✅ generous scoring
        profit_score = _smooth_score(profit_pct, shift=0.02, scale=0.04)
        score = profit_score

    elif task_name == "medium":
        # ✅ stricter than easy
        if 2 <= trade_count <= 20:
            trade_ok = 1.0
        elif trade_count == 0:
            trade_ok = 0.0
        else:
            trade_ok = 0.6

        profit_score = _smooth_score(profit_pct, shift=0.01, scale=0.04)

        # 🔥 difficulty penalty
        score = profit_score * trade_ok * 0.85

    elif task_name == "hard":
        # ✅ strict drawdown penalty
        if mdd <= 0.15:
            mdd_penalty = 1.0
        else:
            mdd_penalty = max(0.0, 1 - (mdd - 0.15) / 0.15)

        profit_score = _smooth_score(profit_pct, shift=0.0, scale=0.05)

        # 🔥 stronger difficulty penalty
        score = profit_score * mdd_penalty * 0.7

    else:
        raise ValueError("Invalid task")

    return round(float(max(0, min(score, 1.0))), 2)