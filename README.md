# Embedding Model Comparison Experiment

Direct comparison of `all-MiniLM-L6-v2` vs `Qwen/Qwen3-Embedding-0.6B` using Jane Eyre text chunks and extracted facts.

## Purpose

Evaluate retrieval accuracy for extracted facts against narrative text segments using cosine similarity and Precision@K metrics.

## Installation & Usage

```bash
# Install dependencies with UV
uv sync
source .venv/bin/activate

# Run experiment
python embedding_comparison_experiment.py
```

Results saved to `embedding_comparison_results.json` with comprehensive diagnostics.

## Data Structure

```
data/
├── source/           # Jane Eyre scene files (JaneEyre-scene-001.txt, etc.)
└── facts/           # Extracted facts JSON (all_chapters_combined_mistral.json)
```

## Key Findings

**Dataset**: 6 text chunks, 819 extracted facts | **Execution Time**: 24.71 seconds

### Model Performance Comparison

| Model | P@1 | P@3 | P@5 |
|-------|-----|-----|-----|
| **Qwen/Qwen3-Embedding-0.6B** | **0.452** | **0.750** | **0.957** |
| all-MiniLM-L6-v2 | 0.227 | 0.620 | 0.873 |

**Winner**: Qwen significantly outperforms MiniLM across all precision metrics.

### Similarity Score Distributions

**Qwen (Superior)**:
- Mean similarity: 0.512 ± 0.096
- Range: [0.248, 0.861] 
- Quartiles: Q25=0.443, Q50=0.506, Q75=0.574
- Higher baseline similarity scores indicate better semantic understanding

**MiniLM (Baseline)**:
- Mean similarity: 0.229 ± 0.112
- Range: [-0.157, 0.607]
- Quartiles: Q25=0.152, Q50=0.228, Q75=0.304
- Lower scores with negative values suggest weaker fact-to-text alignment

### Key Insights

1. **Qwen demonstrates 99% superior performance**: Nearly 2x better P@1 (0.452 vs 0.227)
2. **Consistent advantage**: Qwen leads across all precision levels (P@1, P@3, P@5)
3. **Semantic quality**: Qwen's higher mean similarity (0.512 vs 0.229) indicates better semantic understanding
4. **Stability**: Qwen's tighter distribution (σ=0.096 vs 0.112) shows more consistent performance

## Dependencies

- `sentence-transformers>=2.2.0` - For embedding models
- `torch>=1.12.0` - PyTorch backend  
- `numpy>=1.21.0` - Numerical computations
- `scikit-learn>=1.0.0` - Similarity calculations

## Methodology

1. Load Jane Eyre text chunks and extracted facts
2. Generate embeddings using both models
3. Calculate cosine similarities between facts and text chunks
4. Evaluate retrieval accuracy using Precision@K
5. Analyze similarity score distributions

## License

Jane Eyre text is public domain. Code for research/educational use.

