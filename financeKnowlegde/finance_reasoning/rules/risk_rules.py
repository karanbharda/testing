def evaluate_risk(context):
    triggers = []

    if context.get("stop_loss_hit"):
        triggers.append({
            "rule_id": "RISK_STOPLOSS_001",
            "domain": "RISK",
            "severity": "HIGH"
        })

    if context.get("volatility_spike"):
        triggers.append({
            "rule_id": "RISK_VOLATILITY_002",
            "domain": "RISK",
            "severity": "MEDIUM"
        })

    if context.get("drawdown_limit_hit"):
        triggers.append({
            "rule_id": "RISK_DRAWDOWN_003",
            "domain": "RISK",
            "severity": "CRITICAL"
        })

    return triggers
