# Prompt-Directed Embeddings: When Foundation Models Shift Weekly

> *"Building on a foundation that shifts weekly"* — A real-time case study in embedding model selection during the rapid evolution of open-weight AI.

## The Problem: Choosing Embeddings in a Moving Landscape

In May 2025, only **three open-weight models** offered prompt-directed embeddings—the ability to bend similarity space with natural-language hints at query time. Two were eliminated by licensing constraints or technical limitations. **E5-Mistral-7B** won by elimination, not excellence.

Seven days later, **Qwen3-Embedding** landed and completely reshuffled the performance rankings.

**This repository documents that transition in real-time.**

## What Are Prompt-Directed Embeddings?

Think: `query: "playwright style"` → vectors automatically shift toward theatrical passages **without re-embedding your entire corpus**. It's semantic search with runtime steering.

## The Constraint Landscape (May 2025)

| Model Family           | Prompt Direction | License       | Viable? | Notes                    |
|------------------------|------------------|---------------|---------|--------------------------|
| INSTRUCTOR             | ✓                | Apache-2.0    | ✓       | Lower MTEB scores        |
| NV-Embed-e5-v5         | ✓                | CC-BY-NC-4.0  | ✗       | Non-commercial only      |
| **e5-mistral-7B**      | ✓                | MIT           | ✓       | **Won by elimination**   |
| **Qwen3-Embedding**    | ✓                | Apache-2.0    | ✓       | **Arrived one week later** |

## The Validation Results

**Test Dataset**: 819 factual extractions from Jane Eyre Chapter 1  
**Task**: Identify which of the 6 chapters originated the fact.

| Model         | Size  | Dimensions | P@1     | P@3     | P@5     | Winner |
|---------------|-------|------------|---------|---------|---------|--------|
| E5-Mistral    | 7B    | 4096       | 0.349   | 0.661   | 0.878   |        |
| **Qwen3-0.6B** | **0.6B** | **1024** | **0.452*** | **0.750** | **0.957*** | **🏆** |
| Qwen3-4B      | 4B    | 2048       | 0.403   | 0.803*  | 0.943   |        |
| Qwen3-8B      | 8B    | 4096       | 0.398   | 0.774   | 0.929   |        |

**\* Best in category** | P@N = answer appears in top N results

### Key Finding: Smaller Won
The **0.6B Qwen3 model beats the 7B incumbent by 29% relative improvement** on P@1 and outperforms every larger sibling. Possible explanations:
- **Sample noise** in evaluation
- **Dimensional efficiency**: 1024-dim space forces crisper features for narrative text
- **Architecture advantage**: Qwen3's training approach

## Repository Structure

```
📁 EmbeddingCompare/
├── 📄 LANDING_README.md              ← You are here
├── 📄 README.md                      ← Original Qwen3 vs MiniLM comparison
├── 📄 SCALE_COMPARISON_EXPERIMENTS.md ← Model scale analysis (this document)
├── 📄 E5_MISTRAL_VS_QWEN3_CONFIG_README.md ← E5-Mistral vs Qwen3 setup
│
├── 🔧 Configuration Files
│   ├── config_e5_mistral_vs_qwen3.json      ← E5-Mistral vs Qwen3-0.6B
│   ├── config_qwen3_4b_vs_8b.json          ← Qwen3 scale comparison
│   └── config_instructor_large_vs_xl.json   ← Instructor scale comparison
│
├── 📊 Results & Analysis
│   ├── e5_mistral_vs_qwen3_comparison.json
│   ├── qwen3_4b_vs_8b_scale_comparison.json
│   └── embedding_comparison_results.json
│
├── 🧪 Core Experiment Engine
│   ├── embedding_comparison_experiment.py   ← Main evaluation engine
│   ├── comparison_config.py                 ← Configuration system
│   └── bridge_test.py                      ← Validation framework
│
└── 📁 data/
    ├── source/     ← Jane Eyre scene files
    └── facts/      ← Extracted factual statements
```

## Quick Start

```bash
# Setup environment
uv sync
source .venv/bin/activate

# Run the flagship comparison: E5-Mistral vs Qwen3
python embedding_comparison_experiment.py --config config_e5_mistral_vs_qwen3.json

# Explore scale effects within Qwen3 family
python embedding_comparison_experiment.py --config config_qwen3_4b_vs_8b.json
```

## Key Experiments

### 1. **E5-Mistral vs Qwen3 Head-to-Head** 
- **File**: [`E5_MISTRAL_VS_QWEN3_CONFIG_README.md`](E5_MISTRAL_VS_QWEN3_CONFIG_README.md)
- **Config**: `config_e5_mistral_vs_qwen3.json`
- **Question**: Does the week-later arrival beat the safe-bet incumbent?

### 2. **Scale Analysis Within Families**
- **File**: [`SCALE_COMPARISON_EXPERIMENTS.md`](SCALE_COMPARISON_EXPERIMENTS.md)
- **Configs**: `config_qwen3_4b_vs_8b.json`, `config_instructor_large_vs_xl.json`
- **Question**: Is bigger always better for embedding tasks?

### 3. **Original Baseline Study**
- **File**: [`README.md`](README.md)
- **Focus**: Qwen3-0.6B vs all-MiniLM-L6-v2 comparison
- **Historical Context**: The initial validation that started this investigation

## Why This Matters

### The Control Problem
Closed models bind your data schema to a vendor's timeline. You upgrade on **their** schedule, not **yours**. Embeddings are as fundamental as a database schema—you wouldn't let a vendor dictate when you can query your own data.

### Small Data, Long Half-Lives
Even 70k Gutenberg texts weigh just 57GB. The challenge isn't scale—it's **dataset hygiene**. Mix retention rules, licenses, or PII and you invite compliance nightmares. That's why we start with carefully audited public-domain works.

### Building on Shifting Foundations
This repository captures a specific moment: **May-June 2025, when open embedding models were evolving weekly**. Today's winner may be tomorrow's legacy choice, but the evaluation methodology remains valuable.

## Next Phase: Prompt Direction Testing

The validation above tested **raw semantic understanding** without prompt hints. The real test comes next:

**Same evaluation + hinted queries**. Which model still clears the bar when we activate prompt-directed similarity?

## Research Contributions

1. **Real-time model selection methodology** during rapid AI evolution
2. **Configurable evaluation framework** for embedding model comparison
3. **Open dataset and reproducible benchmarks** for narrative text retrieval
4. **Scale analysis** within model families (is bigger always better?)
5. **Licensing constraint mapping** for production deployment decisions

---

**License**: MIT | **Dataset**: Public domain (Jane Eyre) | **Models**: Various open-weight licenses

*This evaluation captures the state of prompt-directed embeddings as of June 2025. The landscape continues to evolve rapidly.*
