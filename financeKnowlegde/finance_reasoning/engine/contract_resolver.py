def resolve_contracts(trigger_ids, contract_map):
    return [contract_map[t] for t in trigger_ids if t in contract_map]
