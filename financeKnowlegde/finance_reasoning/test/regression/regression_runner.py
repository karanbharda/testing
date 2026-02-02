def run_test(scenario, engine):
    result = engine.process(scenario["context"])

    for rule in scenario["expected_rules"]:
        assert rule in result["rule_ids"]

    for word in scenario["expected_keywords"]:
        assert word.lower() in result["final_answer"].lower()
