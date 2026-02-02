def build_explanation(triggers, contract_map):
    explanation = []

    for t in triggers:
        contract = contract_map.get(t["rule_id"])
        if contract:
            explanation.extend(
                contract["explanation_blocks"]["primary"]
            )

    return explanation
def compose_explanation(blocks):
    """
    Converts explanation blocks into clean finance explanation.
    """

    seen = set()
    ordered = []

    for block in blocks:
        if block not in seen:
            ordered.append(block)
            seen.add(block)

    explanation = ". ".join(ordered)

    if not explanation.endswith("."):
        explanation += "."

    return explanation
