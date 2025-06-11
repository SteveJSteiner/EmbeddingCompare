# E5-Mistral-7B vs Qwen3-Embedding Configuration

## Configuration Overview

This configuration compares two state-of-the-art embedding models:

### Model A: E5-Mistral-7B-Instruct
- **HuggingFace ID**: `intfloat/e5-mistral-7b-instruct`
- **Architecture**: Based on Mistral-7B with instruction tuning
- **Expected Dimensions**: 4096 (high-dimensional embeddings)
- **Trust Remote Code**: False (standard model)
- **Size**: ~14GB (7 billion parameters)

### Model B: Qwen3-Embedding-0.6B  
- **HuggingFace ID**: `Qwen/Qwen3-Embedding-0.6B`
- **Architecture**: Qwen3-based embedding model
- **Expected Dimensions**: Auto-detected (null in config)
- **Trust Remote Code**: True (required for Qwen models)
- **Size**: ~600M parameters

## Experiment Configuration

### Evaluation Settings
- **Precision@K Values**: [1, 3, 5, 10] - Extended to include P@10
- **Similarity Metric**: Cosine similarity
- **Include Diagnostics**: True (full similarity distributions)

### Expected Output
- **Results File**: `e5_mistral_vs_qwen3_comparison.json`
- **Location**: Root directory (no output-dir specified)

## Model Comparison Significance

This comparison is particularly interesting because:

1. **Scale Difference**: 7B vs 0.6B parameters (10x+ difference)
2. **Architecture Variety**: Mistral-based vs Qwen-based models
3. **Embedding Dimensions**: High-dimensional (4096) vs standard dimensions
4. **Training Approach**: Instruction-tuned vs embedding-specialized

## Configuration Features Demonstrated

✅ **Dynamic Model Names**: Uses `e5_mistral_7b` and `qwen3_embedding` internally  
✅ **Configurable K Values**: Extended to [1,3,5,10] for more comprehensive evaluation  
✅ **Large Model Support**: Handles multi-GB model downloads automatically  
✅ **Mixed Trust Settings**: Different trust_remote_code settings per model  
✅ **Dimension Flexibility**: One model with expected dimensions, one auto-detected  

## Expected Performance Insights

The E5-Mistral-7B model, being much larger and instruction-tuned, may show:
- Higher precision scores on fact retrieval tasks
- Different similarity distribution patterns due to higher dimensionality
- Potentially better semantic understanding due to larger parameter count

This comparison will validate the configurable evaluation system with models of significantly different scales and architectures.

## Status

✅ **Configuration Created**: `config_e5_mistral_vs_qwen3.json`  
🔄 **Currently Running**: Model download and comparison in progress  
⏳ **ETA**: ~10-15 minutes (depending on download speed and GPU processing)  

The experiment demonstrates Milestone 3's configurable evaluation system working with arbitrary model pairs, including large-scale models requiring significant downloads.
