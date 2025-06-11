# Bridge Test Design

## Purpose
The bridge test enables safe refactoring by capturing behavioral baselines before changes and verifying identical results after changes. It treats `embedding_comparison_experiment.py` as a black box CLI tool.

## Core Principle
**Process Isolation**: The bridge test executes the script via subprocess, never imports it. This ensures true before/after comparison regardless of internal implementation changes.

## Design

### CLI Interface
The `embedding_comparison_experiment.py` script operates as a standalone tool:
- Executable via `python embedding_comparison_experiment.py`
- Outputs results to `bridge_test_results/` directory  
- Currently uses hardcoded models (all-MiniLM-L6-v2 vs Qwen/Qwen3-Embedding-0.6B)
- Future: Will accept model names as CLI arguments while maintaining backward compatibility

### Bridge Test Operations

#### 1. Capture Baseline
```bash
python bridge_test.py --capture-baseline
```
- Executes the experiment script via subprocess
- Captures the complete JSON output
- Stores as baseline in `bridge_test_results/baseline.json`
- Records execution metadata (timestamp, git commit, etc.)

#### 2. Compare to Baseline  
```bash
python bridge_test.py --compare-baseline
```
- Executes the experiment script via subprocess
- Captures the complete JSON output
- Performs deep comparison with stored baseline
- Reports any differences with detailed diagnostic output

### Comparison Strategy

**Exact Comparison First**: The initial implementation compares results exactly - any difference is a failure. This strict approach ensures we catch all behavioral changes.

**Future Flexibility**: The comparison logic can be enhanced later to handle acceptable variations (floating point tolerance, timing differences, etc.) while maintaining the core framework.

### File Structure
```
bridge_test_results/
├── baseline.json              # Captured baseline results
├── latest_run.json           # Most recent comparison run
├── comparison_report.json    # Detailed diff report
└── execution_log.txt         # Subprocess output capture
```

### Key Benefits

1. **Implementation Agnostic**: Works regardless of internal refactoring
2. **Version Control Friendly**: Can compare across git commits  
3. **True Behavioral Testing**: Only cares about final outputs
4. **Extensible**: Framework supports future enhancements
5. **No Hash Complexity**: Direct result comparison enables nuanced analysis

### Exit Codes
- `0`: Success (baseline captured or comparison passed)
- `1`: Failure (script execution failed or results differ)

### Verbose Output
All operations provide detailed logging:
- Subprocess execution details
- Result comparison diagnostics  
- Timing and performance metrics
- Full error context on failures

## Usage Workflow

1. **Before Refactoring**: `python bridge_test.py --capture-baseline`
2. **Make Changes**: Refactor the implementation
3. **After Refactoring**: `python bridge_test.py --compare-baseline`
4. **Verify**: Ensure comparison passes (identical results)

## Future Enhancements

- **Model Parameter Support**: CLI args for custom model selection
- **Tolerance Configuration**: Configurable comparison sensitivity
- **Multiple Baselines**: Support for different test scenarios
- **Git Integration**: Automatic commit tracking and comparison

The bridge test provides a safety net for experimentation while maintaining the ability to verify that core functionality remains unchanged.

## Execution Plan

### Phase 1: CLI Interface (Step 1)
- Modify `embedding_comparison_experiment.py` to accept CLI args
- Add `--output-dir` parameter (default: `bridge_test_results/`)
- Maintain backward compatibility with current hardcoded behavior
- Ensure script exits with proper codes (0=success, 1=failure)

### Phase 2: Bridge Test Implementation (Step 2) 
- Rewrite `bridge_test.py` to use subprocess instead of imports
- Implement `--capture-baseline` mode
- Implement `--compare-baseline` mode
- Add result comparison logic (exact matching initially)
- Create structured output in `bridge_test_results/`

### Phase 3: Validation (Step 3)
- Run baseline capture with current implementation
- Verify baseline contains expected data structure
- Test comparison mode (should pass with identical runs)
- Document any edge cases or limitations

### Phase 4: Integration (Step 4)
- Update usage documentation
- Create helper scripts if needed
- Validate the complete workflow end-to-end

**Estimated Implementation: 4 steps, can be completed incrementally**
