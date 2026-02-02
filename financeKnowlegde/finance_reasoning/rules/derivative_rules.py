def evaluate_derivatives(context):
    triggers = []

    if context.get("fno_ban"):
        triggers.append({
            "rule_id": "DERIV_FNO_BAN_001",
            "domain": "DERIVATIVES",
            "severity": "MEDIUM"
        })

    if context.get("margin_shortfall"):
        triggers.append({
            "rule_id": "DERIV_MARGIN_002",
            "domain": "DERIVATIVES",
            "severity": "HIGH"
        })

    if context.get("near_expiry"):
        triggers.append({
            "rule_id": "DERIV_EXPIRY_003",
            "domain": "DERIVATIVES",
            "severity": "LOW"
        })

    return triggers
