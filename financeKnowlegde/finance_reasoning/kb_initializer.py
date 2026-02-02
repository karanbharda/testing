#!/usr/bin/env python3
"""
Finance KB Complete Initialization & Setup System
==================================================

Comprehensive setup and initialization for production-ready Finance KB.
Includes data embedding, validation, and system integration.
"""

import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from financeKnowlegde.vectorstore.rag_loader import FinanceRAGLoader
from financeKnowlegde.finance_reasoning.data_integration_manager import DataIntegrationManager
from financeKnowlegde.finance_reasoning.kb_health_checker import KBHealthChecker


class FinanceKBInitializer:
    """Complete Finance KB initialization and setup"""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration"""
        self.config = config or self._get_default_config()
        self.initialized = False
        self.start_time = datetime.now()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'kb_path': 'financeKnowlegde/Finance_KB',
            'vectorstore_path': 'financeKnowlegde/vectorstore',
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'chunk_size': 512,
            'chunk_overlap': 50,
            'kb_version': '2.0.0',
            'health_report_path': 'data/kb_health_report.json',
            'integration_log_path': 'data/kb_integration.json'
        }

    def run_full_initialization(self) -> Dict[str, Any]:
        """Run complete initialization pipeline"""
        print("\n" + "=" * 70)
        print("🚀 FINANCE KB COMPLETE INITIALIZATION & SETUP")
        print("=" * 70)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'steps': {},
            'overall_status': 'pending'
        }

        try:
            # Step 1: Health check (pre-initialization)
            results['steps']['pre_health_check'] = self._run_pre_health_check()
            
            # Step 2: Build vectorstore
            results['steps']['vectorstore_build'] = self._build_vectorstore()
            
            # Step 3: Verify integration
            results['steps']['integration_verification'] = self._verify_integration()
            
            # Step 4: Run full health check (post-initialization)
            results['steps']['post_health_check'] = self._run_post_health_check()
            
            # Step 5: Generate reports
            results['steps']['report_generation'] = self._generate_reports()
            
            # Determine overall status
            if any(s.get('status') == 'fail' for s in results['steps'].values()):
                results['overall_status'] = 'failed'
            elif any(s.get('status') == 'warning' for s in results['steps'].values()):
                results['overall_status'] = 'warning'
            else:
                results['overall_status'] = 'success'
            
            self.initialized = (results['overall_status'] == 'success')
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            results['overall_status'] = 'failed'
            results['error'] = str(e)
        
        # Print summary
        self._print_initialization_summary(results)
        
        return results

    def _run_pre_health_check(self) -> Dict[str, Any]:
        """Run pre-initialization health check"""
        print("\n📋 Step 1: Pre-Initialization Health Check")
        print("-" * 50)
        
        try:
            checker = KBHealthChecker(self.config)
            
            # Run quick checks only
            structure_check = checker._check_structure()
            coverage_check = checker._check_coverage()
            
            status = 'pass' if structure_check['status'] == 'pass' else 'warning'
            
            return {
                'status': status,
                'structure_check': structure_check,
                'coverage_check': coverage_check,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Pre-health check failed: {e}")
            return {'status': 'fail', 'error': str(e)}

    def _build_vectorstore(self) -> Dict[str, Any]:
        """Build and initialize vectorstore"""
        print("\n🔧 Step 2: Building Vectorstore")
        print("-" * 50)
        
        try:
            print("   Initializing RAG loader...")
            rag_loader = FinanceRAGLoader(self.config)
            
            kb_path = self.config.get('kb_path', 'Finance_KB')
            print(f"   Building vectorstore from {kb_path}...")
            
            rag_loader.build_vectorstore(kb_path)
            
            stats = rag_loader.get_stats()
            
            print(f"   ✅ Vectorstore built successfully!")
            print(f"      - Chunks: {stats.get('total_chunks', 0)}")
            print(f"      - Embedding dimension: {stats.get('embedding_dim', 0)}")
            print(f"      - Sources: {len(stats.get('sources', []))}")
            
            return {
                'status': 'pass',
                'vectorstore_stats': stats,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Vectorstore build failed: {e}")
            return {'status': 'fail', 'error': str(e)}

    def _verify_integration(self) -> Dict[str, Any]:
        """Verify KB integration with systems"""
        print("\n🔗 Step 3: Verifying Integration")
        print("-" * 50)
        
        try:
            # Check data integration manager
            print("   Verifying data integration manager...")
            dim = DataIntegrationManager(self.config)
            kb_stats = dim.get_kb_statistics()
            
            print(f"   KB Statistics:")
            print(f"      - Total files: {kb_stats['total_files']}")
            print(f"      - Total size: {kb_stats['total_size_kb']:.2f} KB")
            
            for category, info in kb_stats['categories'].items():
                print(f"      - {category}: {info['files']} files ({info['size_kb']:.2f} KB)")
            
            # Validate integrity
            print("\n   Running integrity validation...")
            validation = dim.validate_kb_integrity()
            
            status = 'pass' if validation['status'] == 'valid' else 'warning'
            
            if validation.get('issues'):
                print(f"   ⚠️  Issues found: {len(validation['issues'])}")
                for issue in validation['issues'][:3]:
                    print(f"       - {issue}")
            
            return {
                'status': status,
                'kb_statistics': kb_stats,
                'validation': validation,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Integration verification failed: {e}")
            return {'status': 'fail', 'error': str(e)}

    def _run_post_health_check(self) -> Dict[str, Any]:
        """Run complete post-initialization health check"""
        print("\n🏥 Step 4: Post-Initialization Health Check")
        print("-" * 50)
        
        try:
            checker = KBHealthChecker(self.config)
            
            # Run content quality check
            quality = checker._check_content_quality()
            
            # Run consistency check
            consistency = checker._check_consistency()
            
            # Run integration check
            integration = checker._check_integration()
            
            # Run performance check
            performance = checker._check_performance()
            
            # Determine status
            statuses = [quality['status'], consistency['status'], integration['status'], performance['status']]
            overall = 'fail' if 'fail' in statuses else 'warning' if 'warning' in statuses else 'pass'
            
            return {
                'status': overall,
                'quality_check': quality,
                'consistency_check': consistency,
                'integration_check': integration,
                'performance_check': performance,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Post-health check failed: {e}")
            return {'status': 'fail', 'error': str(e)}

    def _generate_reports(self) -> Dict[str, Any]:
        """Generate final reports"""
        print("\n📊 Step 5: Generating Reports")
        print("-" * 50)
        
        try:
            # Full health check report
            checker = KBHealthChecker(self.config)
            health_report = checker.run_full_check()
            
            print("   ✅ Health report generated")
            
            return {
                'status': 'pass',
                'health_report_path': str(checker.health_report_path),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {'status': 'fail', 'error': str(e)}

    def _print_initialization_summary(self, results: Dict[str, Any]):
        """Print initialization summary"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("📈 INITIALIZATION SUMMARY")
        print("=" * 70)
        
        print(f"\nOverall Status: {results['overall_status'].upper()}")
        print(f"Duration: {duration:.2f} seconds")
        
        print("\nStep Results:")
        for step_name, result in results['steps'].items():
            status_icon = "✅" if result.get('status') == 'pass' else "⚠️" if result.get('status') == 'warning' else "❌"
            print(f"  {status_icon} {step_name}: {result.get('status', 'unknown').upper()}")
        
        if results['overall_status'] == 'success':
            print("\n🎉 Finance KB successfully initialized and ready for production!")
            print("\nNext steps:")
            print("  1. Run validation tests: python -m pytest tests/")
            print("  2. Start MCP server: python backend/start_mcp_server.py")
            print("  3. Monitor KB health: python financeKnowlegde/finance_reasoning/kb_health_checker.py")
        elif results['overall_status'] == 'warning':
            print("\n⚠️  Finance KB initialized with warnings.")
            print("   Review and address warnings for production readiness.")
        else:
            print("\n❌ Finance KB initialization failed.")
            print("   Review errors and retry initialization.")
        
        print("\n" + "=" * 70)

    def quick_test(self) -> bool:
        """Run quick functionality test"""
        print("\n🧪 Running Quick Functionality Test")
        print("-" * 50)
        
        try:
            # Test RAG loader
            print("Testing RAG loader...", end=" ")
            rag_loader = FinanceRAGLoader(self.config)
            results = rag_loader.retrieve("What are NSE trading hours?", top_k=3)
            print(f"✅ ({len(results)} results)")
            
            # Test data integration
            print("Testing data integration manager...", end=" ")
            dim = DataIntegrationManager(self.config)
            stats = dim.get_kb_statistics()
            print(f"✅ ({stats['total_files']} files)")
            
            # Test health checker
            print("Testing health checker...", end=" ")
            checker = KBHealthChecker(self.config)
            health = checker._check_structure()
            print("✅")
            
            print("\n✅ All functionality tests passed!")
            return True
        
        except Exception as e:
            print(f"\n❌ Functionality test failed: {e}")
            return False


def main():
    """Main initialization entry point"""
    
    # Create initializer
    config = {
        'kb_path': str(Path(__file__).parent / 'Finance_KB'),
        'vectorstore_path': str(Path(__file__).parent / 'vectorstore'),
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'chunk_size': 512,
        'chunk_overlap': 50,
        'kb_version': '2.0.0',
        'health_report_path': str(Path(__file__).parent.parent / 'data' / 'kb_health_report.json'),
        'integration_log_path': str(Path(__file__).parent.parent / 'data' / 'kb_integration.json')
    }
    
    initializer = FinanceKBInitializer(config)
    
    # Run full initialization
    results = initializer.run_full_initialization()
    
    # Run quick test if initialization successful
    if initializer.initialized:
        initializer.quick_test()
    
    # Return exit code
    return 0 if results['overall_status'] == 'success' else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
