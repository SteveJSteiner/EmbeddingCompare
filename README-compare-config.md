# Comparison Configuration Design

## Purpose
Enable `embedding_comparison_experiment.py` to compare arbitrary embedding models through a flexible configuration system while maintaining strict experimental validation and backward compatibility.

## Design Principles

### 1. Configuration-Driven Flexibility
- **JSON-based configuration**: Human-readable, version-controllable, and easily extensible
- **Model-agnostic design**: Support any embedding model compatible with SentenceTransformer
- **Backward compatibility**: Existing hardcoded behavior remains the default

### 2. Experimental Rigor
- **Strict validation**: All configuration parameters must be explicitly validated
- **No silent fallbacks**: Invalid configurations fail immediately with diagnostic output
- **Reproducible results**: Configuration captures all parameters needed for exact reproduction
- **Explicit configuration required**: No default/implicit behavior beyond bridge test validation

### 3. Process Safety
- **Bridge test compatibility**: Temporary legacy mode for baseline validation only
- **Atomic execution**: Either complete success or complete failure - no partial states
- **Full diagnostic output**: Every configuration decision is logged and traceable
- **Migration path**: Clear transition from hardcoded to configuration-driven

## Configuration Schema

### Base Configuration Structure
```json
{
  "experiment_config": {
    "name": "Custom Model Comparison",
    "description": "Comparing two arbitrary embedding models",
    "version": "1.0"
  },
  "models": {
    "model_a": {
      "name": "first-model-identifier",
      "huggingface_id": "sentence-transformers/all-MiniLM-L6-v2",
      "display_name": "MiniLM-L6-v2",
      "expected_dimensions": 384,
      "trust_remote_code": false
    },
    "model_b": {
      "name": "second-model-identifier", 
      "huggingface_id": "Qwen/Qwen3-Embedding-0.6B",
      "display_name": "Qwen3-Embedding-0.6B",
      "expected_dimensions": null,
      "trust_remote_code": true
    }
  },
  "data_config": {
    "base_path": "./data",
    "source_pattern": "JaneEyre-scene-*.txt",
    "facts_file": "facts/all_chapters_combined_mistral.json",
    "validation": {
      "min_chunks": 1,
      "min_facts": 1,
      "require_chapter_mapping": true
    }
  },
  "evaluation_config": {
    "precision_k_values": [1, 3, 5],
    "similarity_metric": "cosine",
    "include_diagnostics": true
  },
  "output_config": {
    "results_file": "comparison_results.json",
    "diagnostic_summary": true,
    "save_embeddings": false
  }
}
```

### Model Configuration Fields

#### Required Fields
- **`name`**: Internal identifier used in results and logging
- **`huggingface_id`**: Model identifier for SentenceTransformer loading
- **`display_name`**: Human-readable name for output

#### Optional Fields  
- **`expected_dimensions`**: Expected embedding dimensionality (null = auto-detect)
- **`trust_remote_code`**: Whether to trust remote code execution (default: false)
- **`model_kwargs`**: Additional parameters passed to SentenceTransformer constructor
- **`encode_kwargs`**: Additional parameters passed to model.encode()

### Data Configuration Options

#### Path Configuration
- **`base_path`**: Root directory for data files
- **`source_pattern`**: Glob pattern for text chunk files
- **`facts_file`**: Path to facts JSON file (relative to base_path)

#### Validation Rules
- **`min_chunks`**: Minimum required text chunks
- **`min_facts`**: Minimum required facts
- **`require_chapter_mapping`**: Whether facts must have chapter information

### Evaluation Configuration

#### Metrics Configuration
- **`precision_k_values`**: List of K values for Precision@K calculation
- **`similarity_metric`**: Similarity function ("cosine", "dot", "euclidean")
- **`include_diagnostics`**: Whether to compute similarity distributions

#### Advanced Options (Future)
- **`cross_validation`**: Enable k-fold cross-validation
- **`statistical_tests`**: Enable significance testing
- **`bootstrap_samples`**: Number of bootstrap samples for confidence intervals

## CLI Interface Evolution

### Current State (Hardcoded)
```bash
python embedding_comparison_experiment.py
# Uses hardcoded all-MiniLM-L6-v2 vs Qwen/Qwen3-Embedding-0.6B
# THIS WILL BE REMOVED AFTER BRIDGE TEST VALIDATION
```

### Proposed Enhanced Interface
```bash
# Primary interface: Configuration file required
python embedding_comparison_experiment.py --config custom_comparison.json

# Override output directory
python embedding_comparison_experiment.py --config custom_comparison.json --output-dir ./custom_results/

# Quick model comparison (auto-generate config)
python embedding_comparison_experiment.py --model-a "all-MiniLM-L6-v2" --model-b "sentence-transformers/all-mpnet-base-v2"

# BRIDGE TEST ONLY: Temporary backward compatibility for validation
python embedding_comparison_experiment.py --legacy-mode
```

### CLI Arguments Priority
1. **Explicit CLI args** (highest priority)
2. **Configuration file** (required for normal operation)  
3. **Legacy mode flag** (bridge test validation ONLY - will be removed)

## Implementation Strategy

### Phase 1: Configuration Infrastructure
```python
class ComparisonConfig:
    """Configuration management with strict validation"""
    
    def __init__(self, config_path: str = None):
        # TODO: Load and validate configuration
        # TODO: Merge with CLI overrides
        # TODO: Assert all required fields present
        pass
        
    def validate_model_config(self, model_config: Dict) -> None:
        """Strict model configuration validation"""
        # TODO: Check required fields exist
        # TODO: Validate HuggingFace model ID exists
        # TODO: Test model loading without full initialization
        pass
        
    def get_model_specs(self) -> Tuple[Dict, Dict]:
        """Return validated model specifications"""
        # TODO: Return model_a and model_b configurations
        pass
```

### Phase 2: Dynamic Model Loading
```python
class EmbeddingComparison:
    def __init__(self, config: ComparisonConfig, output_dir: str = None):
        # TODO: Accept configuration instead of hardcoded paths
        # TODO: Initialize dynamic model containers
        pass
        
    def load_models_from_config(self):
        """Load models dynamically based on configuration"""
        # TODO: Load model_a using config.models.model_a
        # TODO: Load model_b using config.models.model_b  
        # TODO: Validate embedding dimensions match expectations
        # TODO: Store models in generic containers (self.models['model_a'])
        pass
```

### Phase 3: Generalized Evaluation
```python
def evaluate_fact_retrieval_generic(self) -> Dict:
    """Model-agnostic fact retrieval evaluation"""
    # TODO: Replace hardcoded "minilm"/"qwen" with config.models keys
    # TODO: Use self.models[model_name] for embedding computation
    # TODO: Maintain identical evaluation logic with configurable models
    pass
```

### Phase 4: Legacy Mode for Bridge Test ONLY
```python
def get_legacy_config() -> ComparisonConfig:
    """TEMPORARY: Provide legacy configuration for bridge test validation only"""
    # TODO: Return configuration that exactly replicates current behavior
    # TODO: Ensure bridge test passes with --legacy-mode
    # TODO: Add deprecation warning when used
    # TODO: Remove this function after bridge test validation complete
    pass

def handle_legacy_mode():
    """TEMPORARY: Handle --legacy-mode flag for bridge testing"""
    # TODO: Print deprecation warning
    # TODO: Log that legacy mode is for bridge testing only
    # TODO: Return legacy configuration
    # TODO: Exit with error if used outside bridge test context
    pass
```

## Validation Strategy

### Configuration Validation Levels

#### 1. Schema Validation
- **JSON structure**: Validate against schema  
- **Required fields**: Ensure all mandatory fields present
- **Type checking**: Validate field types and constraints
- **Cross-field validation**: Check field relationships (e.g., model_a != model_b)

#### 2. Model Validation  
- **Model existence**: Verify HuggingFace model IDs are valid
- **Loading test**: Attempt model loading without full initialization
- **Dimension verification**: Check embedding dimensions if specified
- **Compatibility test**: Ensure models work with SentenceTransformer

#### 3. Data Validation
- **Path existence**: Verify all data paths exist
- **Content validation**: Check data format matches expectations  
- **Size constraints**: Validate minimum data requirements
- **Structure validation**: Ensure facts have required fields

### Error Handling Philosophy

Following the experimental rigor principle:
```python
# GOOD: Explicit failure with full diagnostic context
assert model_config.get('huggingface_id'), (
    f"Model configuration missing required 'huggingface_id' field. "
    f"Model config: {model_config}. "
    f"Configuration file: {config_path}. "
    f"Available models: {list(self.models.keys())}"
)

# BAD: Silent fallback that masks configuration issues
model_id = model_config.get('huggingface_id', 'all-MiniLM-L6-v2')
```

## Bridge Test Compatibility

### Legacy Mode for Validation ONLY
- **Temporary legacy flag**: `--legacy-mode` replicates current hardcoded behavior
- **Bridge test validation**: Ensure new configuration system produces identical results
- **Deprecation timeline**: Remove legacy mode after successful bridge test validation
- **No permanent backward compatibility**: Force migration to configuration-driven approach

### Testing New Configurations
```bash
# Capture baseline with current hardcoded implementation
python bridge_test.py --capture-baseline

# Test with equivalent legacy mode (should be identical)
python embedding_comparison_experiment.py --legacy-mode
python bridge_test.py --compare-baseline

# Should pass: legacy mode produces identical results to hardcoded version
```

### Migration Validation
```bash
# After implementing configuration system:
# 1. Test legacy mode matches original baseline
python embedding_comparison_experiment.py --legacy-mode
python bridge_test.py --compare-baseline

# 2. Test equivalent configuration produces same results
python embedding_comparison_experiment.py --config legacy_equivalent.json  
python bridge_test.py --compare-baseline

# 3. Remove legacy mode after validation passes
# 4. Update bridge test to use configuration as new baseline
```

## Configuration Examples

### Example 1: Different Model Architectures
```json
{
  "experiment_config": {
    "name": "Sentence-BERT vs MPNet Comparison",
    "description": "Comparing SBERT with MPNet architecture"
  },
  "models": {
    "model_a": {
      "name": "sbert",
      "huggingface_id": "sentence-transformers/all-MiniLM-L6-v2", 
      "display_name": "Sentence-BERT MiniLM",
      "expected_dimensions": 384
    },
    "model_b": {
      "name": "mpnet",
      "huggingface_id": "sentence-transformers/all-mpnet-base-v2",
      "display_name": "MPNet Base",
      "expected_dimensions": 768
    }
  }
}
```

### Example 2: Multilingual Models
```json
{
  "experiment_config": {
    "name": "Multilingual Embedding Comparison",
    "description": "English vs Multilingual model comparison"
  },
  "models": {
    "model_a": {
      "name": "english",
      "huggingface_id": "sentence-transformers/all-MiniLM-L6-v2",
      "display_name": "English MiniLM"
    },
    "model_b": {
      "name": "multilingual", 
      "huggingface_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "display_name": "Multilingual MiniLM"
    }
  }
}
```

### Example 3: Custom Data Configuration
```json
{
  "experiment_config": {
    "name": "Custom Dataset Comparison"
  },
  "models": {
    "model_a": {"name": "baseline", "huggingface_id": "all-MiniLM-L6-v2"},
    "model_b": {"name": "candidate", "huggingface_id": "Qwen/Qwen3-Embedding-0.6B"}
  },
  "data_config": {
    "base_path": "./custom_data",
    "source_pattern": "document-*.txt",
    "facts_file": "extracted_facts.json",
    "validation": {
      "min_chunks": 5,
      "min_facts": 10,
      "require_chapter_mapping": false
    }
  },
  "evaluation_config": {
    "precision_k_values": [1, 5, 10],
    "include_diagnostics": true
  }
}
```

## Implementation Roadmap

### Milestone 1: Core Configuration System
- [ ] Create `ComparisonConfig` class with validation
- [ ] Implement JSON schema validation  
- [ ] Add CLI argument parsing for config file (required)
- [ ] Add temporary `--legacy-mode` flag for bridge test validation ONLY

### Milestone 2: Dynamic Model Loading
- [ ] Refactor `EmbeddingComparison` to require configuration
- [ ] Implement generic model loading from config
- [ ] Update embedding computation to be model-agnostic
- [ ] Validate embedding dimensions and compatibility

### Milestone 3: Configurable Evaluation
- [ ] Generalize evaluation methods to work with any model pair
- [ ] Update result formatting to use configured model names
- [ ] Implement configurable precision@K values
- [ ] Add configurable similarity metrics

### Milestone 4: Enhanced CLI Interface
- [ ] Add `--model-a` and `--model-b` quick comparison flags (auto-generate config)
- [ ] Implement `--output-dir` override
- [ ] Add configuration validation command
- [ ] Create configuration file templates

### Milestone 5: Validation & Legacy Removal
- [ ] Comprehensive configuration validation tests
- [ ] Bridge test verification with legacy mode
- [ ] Verify equivalent configuration produces identical results
- [ ] **REMOVE legacy mode** after bridge test validation passes
- [ ] Update documentation to show configuration-only usage

## Benefits of This Design

### 1. Experimental Flexibility
- **Model exploration**: Easy comparison of any embedding models
- **Hypothesis testing**: Quick validation of model performance claims
- **Ablation studies**: Systematic evaluation of model variations

### 2. Reproducibility
- **Configuration versioning**: Exact experiment reproduction from config files
- **Parameter tracking**: All experiment parameters explicitly documented
- **Result correlation**: Link results to specific configurations

### 3. Maintainability  
- **Separation of concerns**: Configuration separate from implementation
- **Extensibility**: Add new models/metrics without code changes
- **Testing**: Configuration validation independent of experiment logic

### 4. Research Workflow Integration
- **Version control**: JSON configurations track experiment evolution
- **Collaboration**: Share configurations instead of code modifications
- **Automation**: Batch experiments with different configurations
- **Forced migration**: No implicit defaults - explicit configuration required

**IMPORTANT**: The configuration system eliminates implicit behavior. Users must explicitly specify what they want to compare, making experiments more reproducible and reducing hidden assumptions.
