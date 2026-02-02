from finance_reasoning.rules.risk_rules import evaluate_risk
from finance_reasoning.rules.derivative_rules import evaluate_derivatives
from finance_reasoning.rules.priority import DOMAIN_PRIORITY, SEVERITY_PRIORITY

def evaluate_all_rules(context):
    triggers = []

    triggers.extend(evaluate_risk(context))
    triggers.extend(evaluate_derivatives(context))

    return triggers


def apply_priority(triggers):
    """
    Sort and suppress lower-priority explanations
    """

    if not triggers:
        return []

    # Sort by domain priority first, then severity
    triggers_sorted = sorted(
        triggers,
        key=lambda x: (
            DOMAIN_PRIORITY.get(x["domain"], 99),
            SEVERITY_PRIORITY.get(x["severity"], 99)
        )
    )

    highest_domain = triggers_sorted[0]["domain"]

    # suppress lower domain triggers
    final = [
        t for t in triggers_sorted
        if t["domain"] == highest_domain
    ]

    return final
