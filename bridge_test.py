#!/usr/bin/env python3
"""
Bridge Test: Subprocess-based before/after verification
Treats embedding_comparison_experiment.py as black box CLI tool
EXPERIMENT: Process isolation for true behavioral comparison
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any
import logging
import argparse
import time
from datetime import datetime

# Configure diagnostic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BridgeTest:
    def __init__(self):
        logger.info("Initializing BridgeTest - SUBPROCESS MODE")
        self.results_dir = Path("bridge_test_results")
        self.results_dir.mkdir(exist_ok=True)
        self.experiment_script = "embedding_comparison_experiment.py"
        
        # TODO: ASSUMPTION - Virtual environment is already activated
        # CONTEXT: subprocess_execution=requires_active_venv, script_path=embedding_comparison_experiment.py
        self.python_cmd = "python"  # Assumes venv is active
        
    def run_experiment_subprocess(self) -> Dict[str, Any]:
        """Execute experiment script via subprocess and capture results"""
        logger.info("Running experiment via subprocess")
        
        # Build command
        cmd = [
            self.python_cmd,
            self.experiment_script,
            "--output-dir", str(self.results_dir)
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Execute subprocess with comprehensive logging
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                timeout=300  # 5 minute timeout for safety
            )
            
            execution_time = time.time() - start_time
            logger.info(f"Subprocess completed in {execution_time:.2f}s with exit code {result.returncode}")
            
            # Log subprocess output for diagnostics
            if result.stdout:
                logger.debug(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"STDERR:\n{result.stderr}")
                
            # Save execution log
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "command": cmd,
                "exit_code": result.returncode,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            with open(self.results_dir / "execution_log.json", "w") as f:
                json.dump(log_data, f, indent=2)
                
            # TODO: ASSUMPTION - Exit code 0 means success, 1 means failure
            # CONTEXT: result.returncode={result.returncode}, expected=0_for_success
            assert result.returncode == 0, f"Experiment failed with exit code {result.returncode}: {result.stderr}"
            
            # Load and return the results
            output_file = self.results_dir / "experiment_output.json"
            assert output_file.exists(), f"Expected output file missing: {output_file}"
            
            with open(output_file, 'r') as f:
                results = json.load(f)
                
            logger.info("✓ Experiment subprocess completed successfully")
            return results
            
        except subprocess.TimeoutExpired:
            logger.error("Subprocess timed out after 5 minutes")
            assert False, "Experiment subprocess timed out"
            
        except Exception as e:
            logger.error(f"Subprocess execution failed: {e}")
            # TODO: ASSUMPTION - Any subprocess failure should be fatal for bridge test
            # CONTEXT: exception={type(e).__name__}, cmd={cmd}
            assert False, f"Experiment subprocess failed: {e}"
            
    def save_baseline(self, results: Dict) -> str:
        """Save baseline results with metadata"""
        logger.info("Saving baseline results")
        
        baseline_data = {
            "captured_at": datetime.now().isoformat(),
            "experiment_results": results,
            "metadata": {
                "script_version": self.experiment_script,
                "results_structure_version": "1.0"
            }
        }
        
        baseline_file = self.results_dir / "baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"✓ Baseline saved to {baseline_file}")
        
        # Generate baseline summary for user
        exp_info = results["experiment_info"]
        precision = results["precision_metrics"]
        
        baseline_summary = f"""
BASELINE CAPTURED:
- Models: {exp_info['models']['minilm']} vs {exp_info['models']['qwen']}
- Data: {exp_info['data_stats']['text_chunks']} chunks, {exp_info['data_stats']['facts']} facts
- Execution: {exp_info['execution_time']:.2f}s
- P@1: MiniLM={precision['minilm']['p@1']:.6f}, Qwen={precision['qwen']['p@1']:.6f}
"""
        
        return baseline_summary
        
    def load_baseline(self) -> Dict:
        """Load existing baseline results"""
        baseline_file = self.results_dir / "baseline.json"
        
        # TODO: ASSUMPTION - Baseline must exist for comparison
        # CONTEXT: baseline_file={baseline_file}, operation=compare_baseline
        assert baseline_file.exists(), f"No baseline found at {baseline_file}. Run --capture-baseline first."
        
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
            
        logger.info(f"✓ Baseline loaded from {baseline_file}")
        return baseline_data["experiment_results"]
        
    def compare_results(self, baseline: Dict, current: Dict) -> Dict:
        """Compare current results with baseline - exact matching"""
        logger.info("Comparing results with baseline (EXACT MATCHING)")
        
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "comparison_type": "exact_match",
            "matches": {},
            "differences": {},
            "overall_match": True
        }
        
        # Key sections to compare
        compare_sections = [
            "precision_metrics",
            "similarity_distributions"
        ]
        
        for section in compare_sections:
            logger.debug(f"Comparing section: {section}")
            
            baseline_section = baseline.get(section, {})
            current_section = current.get(section, {})
            
            section_match = self._deep_compare(baseline_section, current_section, f"{section}")
            comparison["matches"][section] = section_match
            
            if not section_match:
                comparison["overall_match"] = False
                comparison["differences"][section] = {
                    "baseline": baseline_section,
                    "current": current_section
                }
                
        # Data stats comparison (should be identical)
        baseline_stats = baseline["experiment_info"]["data_stats"]
        current_stats = current["experiment_info"]["data_stats"]
        
        stats_match = baseline_stats == current_stats
        comparison["matches"]["data_stats"] = stats_match
        
        if not stats_match:
            comparison["overall_match"] = False
            comparison["differences"]["data_stats"] = {
                "baseline": baseline_stats,
                "current": current_stats
            }
            
        # Save comparison report
        report_file = self.results_dir / "comparison_report.json"
        with open(report_file, 'w') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
            
        # Save current run results
        current_file = self.results_dir / "latest_run.json"
        with open(current_file, 'w') as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
            
        logger.info(f"✓ Comparison report saved to {report_file}")
        return comparison
        
    def _deep_compare(self, baseline: Any, current: Any, path: str) -> bool:
        """Deep comparison of nested structures"""
        if type(baseline) != type(current):
            logger.debug(f"Type mismatch at {path}: {type(baseline)} vs {type(current)}")
            return False
            
        if isinstance(baseline, dict):
            if set(baseline.keys()) != set(current.keys()):
                logger.debug(f"Key mismatch at {path}: {set(baseline.keys())} vs {set(current.keys())}")
                return False
                
            for key in baseline.keys():
                if not self._deep_compare(baseline[key], current[key], f"{path}.{key}"):
                    return False
            return True
            
        elif isinstance(baseline, (list, tuple)):
            if len(baseline) != len(current):
                logger.debug(f"Length mismatch at {path}: {len(baseline)} vs {len(current)}")
                return False
                
            for i, (b_item, c_item) in enumerate(zip(baseline, current)):
                if not self._deep_compare(b_item, c_item, f"{path}[{i}]"):
                    return False
            return True
            
        elif isinstance(baseline, float):
            # Exact float comparison for now - can be made configurable later
            match = baseline == current
            if not match:
                logger.debug(f"Float mismatch at {path}: {baseline} vs {current} (diff: {abs(baseline - current)})")
            return match
            
        else:
            match = baseline == current
            if not match:
                logger.debug(f"Value mismatch at {path}: {baseline} vs {current}")
            return match
            
    def capture_baseline(self) -> None:
        """Capture baseline by running experiment subprocess"""
        logger.info("🔍 CAPTURING BASELINE")
        print("=" * 80)
        print("BRIDGE TEST: CAPTURING BASELINE")
        print("Executing experiment as subprocess...")
        print("=" * 80)
        
        try:
            # Run experiment via subprocess
            results = self.run_experiment_subprocess()
            
            # Save baseline
            summary = self.save_baseline(results)
            
            print(f"\n✅ BASELINE CAPTURED SUCCESSFULLY")
            print(summary)
            print("=" * 80)
            
        except Exception as e:
            logger.error(f"BASELINE CAPTURE FAILED: {e}")
            print(f"\n❌ BASELINE CAPTURE FAILED: {e}")
            raise
            
    def compare_baseline(self) -> bool:
        """Compare current run with baseline"""
        logger.info("🔍 COMPARING WITH BASELINE")
        print("=" * 80)
        print("BRIDGE TEST: COMPARING WITH BASELINE")
        print("Executing experiment as subprocess...")
        print("=" * 80)
        
        try:
            # Load baseline
            baseline = self.load_baseline()
            
            # Run current experiment
            current = self.run_experiment_subprocess()
            
            # Compare results
            comparison = self.compare_results(baseline, current)
            
            # Report results
            print(f"\n🔍 COMPARISON RESULTS:")
            print(f"Overall Match: {comparison['overall_match']}")
            
            if comparison['overall_match']:
                print("\n✅ BRIDGE TEST PASSED - Results are identical!")
                print("All metrics match baseline exactly.")
            else:
                print("\n❌ BRIDGE TEST FAILED - Results differ from baseline!")
                print("\nDifferences found in:")
                for section, diffs in comparison['differences'].items():
                    print(f"  • {section}")
                    
                # TODO: ASSUMPTION - Any difference is a failure in exact mode
                # CONTEXT: comparison_result=failed, differences={list(comparison['differences'].keys())}
                print(f"\nSee detailed comparison in: {self.results_dir / 'comparison_report.json'}")
                
            print("=" * 80)
            return comparison['overall_match']
            
        except Exception as e:
            logger.error(f"BASELINE COMPARISON FAILED: {e}")
            print(f"\n❌ BASELINE COMPARISON FAILED: {e}")
            raise


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Bridge Test - Subprocess-based before/after verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bridge_test.py --capture-baseline
  python bridge_test.py --compare-baseline
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--capture-baseline",
        action="store_true",
        help="Capture baseline by running experiment subprocess"
    )
    
    group.add_argument(
        "--compare-baseline", 
        action="store_true",
        help="Compare current run with existing baseline"
    )
    
    return parser.parse_args()


def main():
    """Main bridge test execution with CLI support"""
    print("🌉 BRIDGE TEST - Subprocess-based Before/After Verification")
    print("PROCESS ISOLATION | BLACK BOX TESTING | EXACT COMPARISON")
    print("-" * 80)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Initialize bridge test
        bridge = BridgeTest()
        
        if args.capture_baseline:
            bridge.capture_baseline()
            sys.exit(0)
            
        elif args.compare_baseline:
            success = bridge.compare_baseline()
            sys.exit(0 if success else 1)
            
    except Exception as e:
        logger.error(f"BRIDGE TEST FAILED: {e}")
        print(f"\n❌ BRIDGE TEST FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
