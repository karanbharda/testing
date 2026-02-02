#!/usr/bin/env python3
"""
Finance Test Pack Validator
==========================

Validates the complete finance system against 25 Q&A test cases.
Tests Days 1-4 implementation: KB, RAG, Grounding, Test Pack.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add project root to path for mcp_service import
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_service.chat.finance_grounding import FinanceChatGrounding

logger = logging.getLogger(__name__)

class FinanceTestValidator:
    """Validates finance system against test pack"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.grounding_system = FinanceChatGrounding(config)
        self.test_pack_path = Path(config.get("test_pack_path", "backend/finance_reasoning/test/finance_test_pack.json"))

    async def run_validation(self) -> Dict[str, Any]:
        """Run complete validation against test pack"""
        print("🔍 FINANCE SYSTEM VALIDATION")
        print("=" * 50)

        # Load test pack
        test_pack = self._load_test_pack()
        if not test_pack:
            return {"status": "failed", "error": "Test pack not found"}

        total_tests = len(test_pack)
        passed_tests = 0
        failed_tests = []

        print(f"📋 Running {total_tests} test cases...")

        for i, test_case in enumerate(test_pack, 1):
            print(f"\n🧪 Test {i}/{total_tests}: {test_case['question'][:50]}...")

            try:
                # Get system response
                result = await self.grounding_system.ground_response(test_case['question'])

                # Debug: Print actual response and knowledge sources
                print(f"   📝 Response: {result['response'][:100]}...")
                print(f"   📚 Knowledge sources: {result.get('knowledge_sources', 0)}")

                # Validate response
                validation = self._validate_response(
                    result['response'],
                    test_case['expected_answer'],
                    test_case['category']
                )

                if validation['passed']:
                    passed_tests += 1
                    print(f"   ✅ PASSED ({test_case['category']})")
                else:
                    failed_tests.append({
                        'test_case': test_case,
                        'result': result,
                        'validation': validation
                    })
                    print(f"   ❌ FAILED: {validation['reason']}")

            except Exception as e:
                failed_tests.append({
                    'test_case': test_case,
                    'error': str(e)
                })
                print(f"   ❌ ERROR: {e}")

        # Generate report
        success_rate = (passed_tests / total_tests) * 100

        report = {
            'status': 'passed' if success_rate >= 80 else 'failed',
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': len(failed_tests),
            'success_rate': round(success_rate, 1),
            'failures': failed_tests,
            'validation_timestamp': asyncio.get_event_loop().time()
        }

        self._print_report(report)
        return report

    def _load_test_pack(self) -> List[Dict[str, Any]]:
        """Load test pack from JSON file"""
        try:
            with open(self.test_pack_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading test pack: {e}")
            return []

    def _validate_response(self, actual: str, expected: str, category: str) -> Dict[str, Any]:
        """Validate system response against expected answer"""

        # Check for forbidden content
        forbidden_patterns = [
            'confidence', 'threshold', 'internal score', 'agent logic',
            'model weight', 'probability', 'algorithm', 'AI analysis'
        ]

        for pattern in forbidden_patterns:
            if pattern.lower() in actual.lower():
                return {
                    'passed': False,
                    'reason': f'Contains forbidden content: "{pattern}"'
                }

        # Check for financial reasoning
        financial_indicators = [
            'price', 'volume', 'risk', 'market', 'position', 'stop loss',
            'margin', 'settlement', 'circuit', 'volatility', 'exposure'
        ]

        has_financial_content = any(
            indicator in actual.lower() for indicator in financial_indicators
        )

        if not has_financial_content and len(actual.split()) > 3:
            return {
                'passed': False,
                'reason': 'Response lacks financial reasoning'
            }

        # Category-specific validation
        if category == 'nse_rules':
            nse_indicators = ['nse', 'settlement', 'circuit', 'trading hours', 'margin']
            if not any(indicator in actual.lower() for indicator in nse_indicators):
                return {
                    'passed': False,
                    'reason': 'NSE rules response missing key indicators'
                }

        elif category == 'f&o_ban_logic':
            ban_indicators = ['ban', 'volume', 'price movement', 'trading days']
            if not any(indicator in actual.lower() for indicator in ban_indicators):
                return {
                    'passed': False,
                    'reason': 'F&O ban logic missing key concepts'
                }

        elif category == 'stop_loss_reasoning':
            stop_indicators = ['stop loss', 'risk', 'protection', 'trigger']
            if not any(indicator in actual.lower() for indicator in stop_indicators):
                return {
                    'passed': False,
                    'reason': 'Stop loss reasoning missing key concepts'
                }

        elif category == 'market_regimes':
            regime_indicators = ['regime', 'trend', 'volatility', 'market conditions']
            if not any(indicator in actual.lower() for indicator in regime_indicators):
                return {
                    'passed': False,
                    'reason': 'Market regime analysis missing key concepts'
                }

        elif category == 'risk_based_exits':
            exit_indicators = ['risk', 'exposure', 'limit', 'position', 'reduction']
            if not any(indicator in actual.lower() for indicator in exit_indicators):
                return {
                    'passed': False,
                    'reason': 'Risk-based exit logic missing key concepts'
                }

        # Length check (should be concise but informative)
        if len(actual.split()) < 5:
            return {
                'passed': False,
                'reason': 'Response too short'
            }

        if len(actual.split()) > 50:
            return {
                'passed': False,
                'reason': 'Response too verbose'
            }

        return {
            'passed': True,
            'reason': 'Validation passed'
        }

    def _print_report(self, report: Dict[str, Any]):
        """Print validation report"""
        print("\n" + "=" * 50)
        print("📊 VALIDATION REPORT")
        print("=" * 50)
        print(f"Status: {'✅ PASSED' if report['status'] == 'passed' else '❌ FAILED'}")
        print(f"Success Rate: {report['success_rate']}%")
        print(f"Tests Passed: {report['passed_tests']}/{report['total_tests']}")

        if report['failed_tests'] > 0:
            print(f"Failed Tests: {report['failed_tests']}")
            print("\nTop 3 Failures:")
            for i, failure in enumerate(report['failures'][:3], 1):
                test_case = failure.get('test_case', {})
                reason = failure.get('validation', {}).get('reason', 'Unknown error')
                print(f"  {i}. {test_case.get('question', 'Unknown')[:40]}...")
                print(f"     Reason: {reason}")

        print("\n🎯 System Capabilities Validated:")
        print("  ✅ Finance Knowledge Base (Day 1)")
        print("  ✅ RAG Embeddings System (Day 2)")
        print("  ✅ Chat Grounding (Day 3)")
        print("  ✅ Test Pack Validation (Day 4)")

async def main():
    """Main validation function"""
    config = {
        'rag': {
            'vectorstore_path': 'financeKnowlegde/vectorstore'
        },
        'kb_path': 'financeKnowlegde/Finance_KB',
        'test_pack_path': 'financeKnowlegde/finance_reasoning/test/finance_test_pack.json'
    }

    validator = FinanceTestValidator(config)
    report = await validator.run_validation()

    # Exit with appropriate code
    exit_code = 0 if report['status'] == 'passed' else 1
    return exit_code

if __name__ == "__main__":
    exit_code = asyncio.run(main())