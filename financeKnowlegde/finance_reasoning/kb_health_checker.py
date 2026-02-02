#!/usr/bin/env python3
"""
Finance KB Health Check & Validation System
==============================================

Monitors KB health, validates completeness, and ensures quality standards.
Implements industry-level checks for production readiness.
"""

import json
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime
import hashlib
import re

logger = logging.getLogger(__name__)

class KBHealthChecker:
    """Comprehensive health checker for Finance KB"""

    # Quality thresholds
    THRESHOLDS = {
        'min_chunk_size': 50,  # Minimum words per chunk
        'max_chunk_size': 2000,  # Maximum words per chunk
        'min_file_size': 500,  # Minimum bytes per file
        'min_content_quality': 0.7,  # 70% quality threshold
        'required_sections': {
            'equities': ['nse_advanced_rules', 'sebi_listings_compliance'],
            'derivatives': ['options_greeks_advanced', 'futures_contracts_advanced'],
            'ta_indicators': ['advanced_indicators'],
            'fa_basics': ['financial_statements_analysis'],
            'risk_models': ['comprehensive_risk_framework'],
            'strategies': ['trading_systems_detailed']
        }
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kb_path = Path(config.get("kb_path", "Finance_KB"))
        self.health_report_path = Path(config.get("health_report_path", "data/kb_health_report.json"))
        
        self.issues = []
        self.warnings = []
        self.statistics = {}

    def run_full_check(self) -> Dict[str, Any]:
        """Run complete health check"""
        print("🏥 FINANCE KB HEALTH CHECK")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Run all checks
        checks = [
            ("Structure Check", self._check_structure),
            ("Content Quality", self._check_content_quality),
            ("Coverage Check", self._check_coverage),
            ("Consistency Check", self._check_consistency),
            ("Integration Check", self._check_integration),
            ("Performance Check", self._check_performance),
        ]
        
        check_results = {}
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}...", end=" ")
            try:
                result = check_func()
                check_results[check_name] = result
                status = "✅ PASS" if result['status'] == 'pass' else "⚠️  WARNING" if result['status'] == 'warning' else "❌ FAIL"
                print(status)
            except Exception as e:
                print(f"❌ ERROR: {e}")
                self.issues.append(f"{check_name} failed: {str(e)}")
                check_results[check_name] = {'status': 'fail', 'error': str(e)}
        
        # Compile report
        duration = (datetime.now() - start_time).total_seconds()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': round(duration, 2),
            'overall_status': self._determine_overall_status(),
            'checks': check_results,
            'statistics': self.statistics,
            'issues': self.issues,
            'warnings': self.warnings,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        self._save_report(report)
        
        # Print summary
        self._print_summary(report)
        
        return report

    def _check_structure(self) -> Dict[str, Any]:
        """Validate KB directory structure"""
        result = {'status': 'pass', 'details': {}}
        
        required_dirs = list(self.THRESHOLDS['required_sections'].keys())
        missing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = self.kb_path / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
                self.issues.append(f"Missing directory: {dir_name}")
                result['status'] = 'fail'
            else:
                md_files = list(dir_path.glob('*.md'))
                result['details'][dir_name] = {
                    'exists': True,
                    'file_count': len(md_files),
                    'files': [f.name for f in md_files]
                }
        
        if missing_dirs:
            result['missing_directories'] = missing_dirs
        
        return result

    def _check_content_quality(self) -> Dict[str, Any]:
        """Check content quality metrics"""
        result = {'status': 'pass', 'quality_scores': {}}
        
        for category_dir in self.kb_path.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith('_'):
                continue
            
            category_quality = []
            
            for md_file in category_dir.glob('*.md'):
                quality_score = self._assess_file_quality(md_file)
                category_quality.append(quality_score)
                
                if quality_score < self.THRESHOLDS['min_content_quality']:
                    self.warnings.append(
                        f"Low quality content: {md_file.relative_to(self.kb_path)} "
                        f"(score: {quality_score:.2f})"
                    )
            
            if category_quality:
                avg_quality = sum(category_quality) / len(category_quality)
                result['quality_scores'][category_dir.name] = round(avg_quality, 3)
                
                if avg_quality < self.THRESHOLDS['min_content_quality']:
                    result['status'] = 'warning'
        
        return result

    def _assess_file_quality(self, file_path: Path) -> float:
        """Assess individual file quality (0.0 to 1.0)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            score = 0.0
            checks = 0
            
            # Check 1: File size (0.1 points)
            if len(content) >= self.THRESHOLDS['min_file_size']:
                score += 0.1
            checks += 1
            
            # Check 2: Has headings (0.15 points)
            if len(re.findall(r'^#+\s', content, re.MULTILINE)) > 0:
                score += 0.15
            checks += 1
            
            # Check 3: Has code blocks or structure (0.15 points)
            if '```' in content or '- ' in content or '| ' in content:
                score += 0.15
            checks += 1
            
            # Check 4: Has multiple paragraphs (0.15 points)
            paragraphs = len([p for p in content.split('\n\n') if len(p.strip()) > 50])
            if paragraphs >= 3:
                score += 0.15
            checks += 1
            
            # Check 5: No excessive whitespace (0.1 points)
            if content.count('\n\n\n') == 0:
                score += 0.1
            checks += 1
            
            # Check 6: Contains relevant keywords (0.15 points)
            keywords = ['trading', 'price', 'market', 'risk', 'return', 'stock', 'index',
                       'financial', 'portfolio', 'analysis', 'strategy', 'rule', 'requirement']
            keyword_count = sum(1 for kw in keywords if kw in content.lower())
            if keyword_count >= 3:
                score += 0.15
            checks += 1
            
            # Check 7: Proper formatting (0.15 points)
            has_bold = '**' in content or '__' in content
            has_lists = re.search(r'^[\s]*[-*+]\s', content, re.MULTILINE)
            if has_bold or has_lists:
                score += 0.15
            checks += 1
            
            return min(score, 1.0)
        
        except Exception as e:
            logger.error(f"Error assessing {file_path}: {e}")
            return 0.0

    def _check_coverage(self) -> Dict[str, Any]:
        """Check KB coverage against required sections"""
        result = {'status': 'pass', 'coverage': {}}
        
        for category, required_sections in self.THRESHOLDS['required_sections'].items():
            category_path = self.kb_path / category
            
            if not category_path.exists():
                result['coverage'][category] = {'status': 'missing'}
                result['status'] = 'fail'
                continue
            
            found_sections = []
            missing_sections = []
            
            for required in required_sections:
                # Check for file containing required section
                found = False
                for md_file in category_path.glob('*.md'):
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if required.replace('_', ' ').lower() in content.lower():
                            found = True
                            break
                
                if found:
                    found_sections.append(required)
                else:
                    missing_sections.append(required)
                    self.warnings.append(f"Missing section: {category}/{required}")
            
            coverage_pct = (len(found_sections) / len(required_sections)) * 100
            result['coverage'][category] = {
                'coverage_percent': round(coverage_pct, 1),
                'found': found_sections,
                'missing': missing_sections
            }
            
            if coverage_pct < 100:
                result['status'] = 'warning'
        
        return result

    def _check_consistency(self) -> Dict[str, Any]:
        """Check consistency across KB"""
        result = {'status': 'pass', 'checks': {}}
        
        # Check 1: File naming consistency
        naming_issues = []
        for md_file in self.kb_path.rglob('*.md'):
            if not re.match(r'^[a-z0-9_]+\.md$', md_file.name):
                naming_issues.append(md_file.name)
        
        if naming_issues:
            result['checks']['naming'] = {
                'status': 'fail',
                'issues': naming_issues
            }
            result['status'] = 'warning'
        else:
            result['checks']['naming'] = {'status': 'pass'}
        
        # Check 2: Content consistency (same topic, similar depth)
        structure_issues = []
        for md_file in self.kb_path.rglob('*.md'):
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                heading_count = len(re.findall(r'^#+\s', content, re.MULTILINE))
                if heading_count < 2:
                    structure_issues.append(md_file.name)
        
        if structure_issues:
            self.warnings.append(f"Files with insufficient headings: {len(structure_issues)}")
        
        result['checks']['structure'] = {
            'status': 'warning' if structure_issues else 'pass',
            'issues_count': len(structure_issues)
        }
        
        return result

    def _check_integration(self) -> Dict[str, Any]:
        """Check KB integration with RAG system"""
        result = {'status': 'pass', 'details': {}}
        
        try:
            # Check vectorstore exists
            vectorstore_path = Path(self.config.get("vectorstore_path", "vectorstore"))
            embeddings_file = vectorstore_path / "embeddings.pkl"
            chunks_file = vectorstore_path / "chunks.pkl"
            
            result['details']['vectorstore_exists'] = embeddings_file.exists() and chunks_file.exists()
            
            if not result['details']['vectorstore_exists']:
                self.warnings.append("Vectorstore not found. Run initialization to build vectorstore.")
                result['status'] = 'warning'
            else:
                result['details']['vectorstore_ready'] = True
            
            # Check integration metadata
            metadata_file = self.kb_path / "integration_metadata.json"
            result['details']['metadata_exists'] = metadata_file.exists()
            
        except Exception as e:
            result['status'] = 'fail'
            result['error'] = str(e)
        
        return result

    def _check_performance(self) -> Dict[str, Any]:
        """Check KB performance metrics"""
        result = {'status': 'pass', 'metrics': {}}
        
        try:
            total_files = 0
            total_size = 0
            avg_file_size = 0
            
            for md_file in self.kb_path.rglob('*.md'):
                total_files += 1
                file_size = md_file.stat().st_size
                total_size += file_size
            
            if total_files > 0:
                avg_file_size = total_size / total_files
            
            result['metrics']['total_files'] = total_files
            result['metrics']['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            result['metrics']['avg_file_size_kb'] = round(avg_file_size / 1024, 2)
            
            # Store for report
            self.statistics['kb_files'] = total_files
            self.statistics['kb_size_mb'] = result['metrics']['total_size_mb']
            
        except Exception as e:
            result['status'] = 'fail'
            result['error'] = str(e)
        
        return result

    def _determine_overall_status(self) -> str:
        """Determine overall health status"""
        if self.issues:
            return 'unhealthy'
        elif self.warnings:
            return 'warning'
        else:
            return 'healthy'

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on checks"""
        recommendations = []
        
        if not (self.kb_path / "integration_metadata.json").exists():
            recommendations.append("Initialize KB with data_integration_manager to track sources")
        
        if len(self.warnings) > 0:
            recommendations.append(f"Address {len(self.warnings)} warnings for production readiness")
        
        if not any((self.kb_path / 'equities').glob('*advanced*.md')):
            recommendations.append("Add advanced equities content for comprehensive coverage")
        
        if not any((self.kb_path / 'risk_models').glob('*.md')):
            recommendations.append("Expand risk management section with more detailed content")
        
        recommendations.append("Run vectorstore initialization to enable semantic search")
        recommendations.append("Set up automated health checks in CI/CD pipeline")
        
        return recommendations

    def _save_report(self, report: Dict[str, Any]):
        """Save health report to file"""
        try:
            self.health_report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.health_report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Health report saved to {self.health_report_path}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

    def _print_summary(self, report: Dict[str, Any]):
        """Print health check summary"""
        print("\n" + "=" * 60)
        print("📊 HEALTH CHECK SUMMARY")
        print("=" * 60)
        print(f"\nOverall Status: {report['overall_status'].upper()}")
        print(f"Duration: {report['duration_seconds']}s")
        print(f"KB Files: {self.statistics.get('kb_files', 'N/A')}")
        print(f"KB Size: {self.statistics.get('kb_size_mb', 'N/A')} MB")
        
        if report['issues']:
            print(f"\n❌ Issues ({len(report['issues'])}):")
            for issue in report['issues'][:5]:
                print(f"   - {issue}")
            if len(report['issues']) > 5:
                print(f"   ... and {len(report['issues']) - 5} more")
        
        if report['warnings']:
            print(f"\n⚠️  Warnings ({len(report['warnings'])}):")
            for warning in report['warnings'][:5]:
                print(f"   - {warning}")
            if len(report['warnings']) > 5:
                print(f"   ... and {len(report['warnings']) - 5} more")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"   {i}. {rec}")
        
        print("\n✅ Health check complete!")


if __name__ == "__main__":
    config = {
        'kb_path': 'Finance_KB',
        'health_report_path': 'data/kb_health_report.json',
        'vectorstore_path': 'vectorstore'
    }
    
    checker = KBHealthChecker(config)
    report = checker.run_full_check()
