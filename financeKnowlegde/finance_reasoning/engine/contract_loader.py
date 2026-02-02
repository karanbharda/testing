#!/usr/bin/env python3
"""
Contract Loader
===============

Loads trading contracts from JSON files for the Finance Reasoning Engine.
"""

import os
import json
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

def load_contracts() -> Dict[str, Dict[str, Any]]:
    """
    Load all contracts from the contracts directory

    Returns:
        Dictionary mapping contract IDs to contract data
    """
    contracts = {}

    # Get the contracts directory path
    current_dir = Path(__file__).parent.parent
    contracts_dir = current_dir / "contracts"

    if not contracts_dir.exists():
        logger.warning(f"Contracts directory not found: {contracts_dir}")
        return contracts

    # Load contracts from all subdirectories
    for domain_dir in contracts_dir.iterdir():
        if domain_dir.is_dir():
            domain_name = domain_dir.name

            for contract_file in domain_dir.glob("*.json"):
                try:
                    with open(contract_file, 'r', encoding='utf-8') as f:
                        contract_data = json.load(f)

                    # Validate contract has required fields
                    required_fields = ["id", "domain", "explanation_blocks"]
                    if not all(field in contract_data for field in required_fields):
                        logger.warning(f"Contract {contract_file} missing required fields")
                        continue

                    # Add to contracts map
                    contract_id = contract_data["id"]
                    contracts[contract_id] = contract_data

                    logger.debug(f"Loaded contract: {contract_id}")

                except Exception as e:
                    logger.error(f"Failed to load contract {contract_file}: {e}")

    logger.info(f"Loaded {len(contracts)} contracts")
    return contracts


def get_contract_by_id(contract_id: str) -> Dict[str, Any]:
    """
    Get a specific contract by ID

    Args:
        contract_id: Contract identifier

    Returns:
        Contract data or empty dict if not found
    """
    contracts = load_contracts()
    return contracts.get(contract_id, {})


def get_contracts_by_domain(domain: str) -> Dict[str, Dict[str, Any]]:
    """
    Get all contracts for a specific domain

    Args:
        domain: Domain name (e.g., "RISK", "DERIVATIVES")

    Returns:
        Dictionary of contracts for the domain
    """
    contracts = load_contracts()
    domain_contracts = {}

    for contract_id, contract_data in contracts.items():
        if contract_data.get("domain") == domain:
            domain_contracts[contract_id] = contract_data

    return domain_contracts
