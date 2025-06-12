# Scale Comparison Experiments

## Overview

This document tracks scale comparison experiments using the configurable evaluation system to understand how model size affects embedding performance within the same architecture families.

## Experiment 1: Qwen3-Embedding Scale Analysis
**Configuration**: `config_qwen3_4b_vs_8b.json`  
**Output**: `qwen3_4b_vs_8b_scale_comparison.json`

### Models Compared
- **Qwen3-Embedding-4B**: 4 billion parameter model
- **Qwen3-Embedding-8B**: 8 billion parameter model (2x larger)

### Hypothesis
Larger Qwen3-8B model should show better semantic understanding and fact retrieval precision due to increased parameter count.

### Configuration Features
- **Dynamic Model Names**: `qwen3_4b` and `qwen3_8b`
- **Auto-Detected Dimensions**: Both models use `expected_dimensions: null`
- **Trust Remote Code**: Both require `trust_remote_code: true`
- **Extended K Values**: [1, 3, 5, 10] for comprehensive evaluation

---

## Experiment 2: Instructor Scale Analysis  
**Configuration**: `config_instructor_large_vs_xl.json`  
**Output**: `instructor_large_vs_xl_comparison.json`

### Models Compared
- **Instructor-Large**: Standard large-scale instruction-following embedding model
- **Instructor-XL**: Extra-large version with enhanced capabilities

### Hypothesis
Instructor-XL should demonstrate superior performance due to larger capacity for following embedding instructions and understanding context.

### Configuration Features
- **Dynamic Model Names**: `instructor_large` and `instructor_xl`
- **Fixed Dimensions**: Both expected to produce 768-dimensional embeddings
- **Standard Trust**: Both use `trust_remote_code: false`
- **Extended K Values**: [1, 3, 5, 10] for comprehensive evaluation

---

## Comparative Analysis Framework

### Scale Effect Metrics
1. **Precision@K Performance**: How does model size affect fact retrieval accuracy?
2. **Similarity Distributions**: Do larger models show different similarity patterns?
3. **Processing Time**: What's the computational cost of increased scale?
4. **Discrimination Power**: Do larger models better distinguish relevant/irrelevant content?

### Cross-Architecture Insights
- **Qwen3 Family**: Same architecture, different scales (4B vs 8B)
- **Instructor Family**: Instruction-tuned models, different scales (Large vs XL)
- **Performance Patterns**: Is "bigger always better" for embedding tasks?

### Configuration System Validation
These experiments further validate the configurable evaluation system's ability to:
- ✅ Handle different model scales seamlessly
- ✅ Process models with different parameter counts
- ✅ Generate dynamic model names for any architecture
- ✅ Support both auto-detected and fixed embedding dimensions
- ✅ Work with varying trust_remote_code requirements

---

## Expected Research Insights

### Scale vs Performance
- Do 8B models consistently outperform 4B models?
- What's the performance gain per additional parameter?
- Are there diminishing returns at larger scales?

### Architecture-Specific Patterns
- **Qwen3**: How does scale affect Chinese-influenced embedding models?
- **Instructor**: How does scale impact instruction-following capability?

### Task-Specific Optimization
- Which model size is optimal for fact retrieval tasks?
- Do larger models show better semantic understanding?
- Is there a sweet spot for computational efficiency vs performance?

---

## Running Experiments

```bash
# Qwen3 Scale Comparison
cd /home/steve/EmbeddingCompare
source .venv/bin/activate
python3 embedding_comparison_experiment.py --config config_qwen3_4b_vs_8b.json

# Instructor Scale Comparison  
python3 embedding_comparison_experiment.py --config config_instructor_large_vs_xl.json
```

Both experiments are currently running and will provide valuable insights into how model scale affects embedding performance for fact retrieval tasks.
