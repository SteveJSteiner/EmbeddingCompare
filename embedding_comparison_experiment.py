#!/usr/bin/env python3
"""
Embedding Model Comparison: all-MiniLM-L6-v2 vs Qwen/Qwen3-Embedding-0.6B
Direct comparison using Jane Eyre text chunks and extracted facts.
EXPERIMENT: Fragile implementation with zero fallbacks for precise testing.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import time
import argparse
import sys
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Import configuration system
from comparison_config import ComparisonConfig

# Configure diagnostic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingComparison:
    def __init__(self, config: ComparisonConfig, output_dir: str = None):
        logger.info("Initializing EmbeddingComparison - CONFIGURATION-DRIVEN MODE")
        
        # TODO: ASSUMPTION - Configuration is valid and contains all required model specs
        # CONTEXT: config_source={config.config_source}
        assert config is not None, (
            "Configuration is required for EmbeddingComparison. "
            "Configuration must be loaded with ComparisonConfig before initialization."
        )
        
        self.config = config
        
        # Get data configuration from config
        data_config = config.get_data_config()
        self.base_path = Path(data_config["base_path"])
        self.source_path = self.base_path / "source" 
        self.facts_path = self.base_path / data_config["facts_file"]
        
        # Output directory configuration  
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
        else:
            self.output_dir = Path(".")
        
        # NO FALLBACKS - Paths must exist exactly
        self._verify_paths_strict()
        
        # Dynamic model configuration from config
        model_a, model_b = config.get_model_specs()
        self.model_specs = {
            model_a["name"]: model_a,
            model_b["name"]: model_b
        }
        self.model_names = list(self.model_specs.keys())  # Dynamic model names
        
        logger.info(f"Dynamic model configuration loaded: {self.model_names}")
        logger.debug(f"Model A: {model_a['display_name']} ({model_a['huggingface_id']})")
        logger.debug(f"Model B: {model_b['display_name']} ({model_b['huggingface_id']})")
        
        # Storage containers - now dynamic based on config
        self.text_chunks = {}
        self.facts = []
        self.embeddings = {name: {} for name in self.model_names}
        self.models = {}  # Will store loaded SentenceTransformer models
        self.results = {}
        
    def _verify_paths_strict(self):
        """Strict path verification with no fallbacks"""
        logger.debug(f"Verifying base path: {self.base_path}")
        assert self.base_path.exists(), f"Base path missing: {self.base_path}"
        
        logger.debug(f"Verifying source path: {self.source_path}")
        assert self.source_path.exists(), f"Source path missing: {self.source_path}"
        
        logger.debug(f"Verifying facts file: {self.facts_path}")
        assert self.facts_path.exists(), f"Facts file missing: {self.facts_path}"
        
        scene_files = list(self.source_path.glob("JaneEyre-scene-*.txt"))
        assert len(scene_files) > 0, f"No scene files found in {self.source_path}"
        
        logger.info(f"Path verification complete: {len(scene_files)} scene files found")
                    
    def load_models_from_config(self):
        """Load models dynamically based on configuration with strict validation"""
        logger.info("Loading models from configuration in STRICT mode")
        
        for model_name, model_spec in self.model_specs.items():
            logger.debug(f"Loading model: {model_name} ({model_spec['huggingface_id']})")
            
            # TODO: ASSUMPTION - HuggingFace model ID exists and is accessible
            # CONTEXT: model_name={model_name}, huggingface_id={model_spec['huggingface_id']}
            try:
                # Load with trust_remote_code setting from configuration
                trust_code = model_spec.get("trust_remote_code", False)
                logger.debug(f"Loading {model_spec['huggingface_id']} with trust_remote_code={trust_code}")
                
                model = SentenceTransformer(
                    model_spec["huggingface_id"],
                    trust_remote_code=trust_code
                )
                
                self.models[model_name] = model
                logger.info(f"✓ {model_spec['display_name']} loaded successfully")
                
                # Validate embedding dimensions if specified
                if model_spec.get("expected_dimensions") is not None:
                    self._validate_embedding_dimensions(model_name, model_spec)
                    
            except Exception as e:
                assert False, (
                    f"Failed to load model '{model_name}' ({model_spec['huggingface_id']}). "
                    f"Error: {type(e).__name__}: {e}. "
                    f"Model spec: {model_spec}. "
                    f"Please verify the model ID exists on HuggingFace and is accessible."
                )
        
        logger.info(f"✓ All models loaded successfully: {list(self.models.keys())}")
        
    def _validate_embedding_dimensions(self, model_name: str, model_spec: Dict):
        """Validate that model produces expected embedding dimensions"""
        expected_dims = model_spec["expected_dimensions"]
        logger.debug(f"Validating embedding dimensions for {model_name} (expected: {expected_dims})")
        
        # Test with a simple sentence
        test_text = "This is a test sentence for dimension validation."
        test_embedding = self.models[model_name].encode([test_text])[0]
        actual_dims = test_embedding.shape[0]
        
        # TODO: ASSUMPTION - Model produces embeddings with expected dimensions
        # CONTEXT: model_name={model_name}, expected_dims={expected_dims}, actual_dims={actual_dims}
        assert actual_dims == expected_dims, (
            f"Model '{model_name}' dimension mismatch. "
            f"Expected: {expected_dims}, Got: {actual_dims}. "
            f"Model: {model_spec['huggingface_id']}. "
            f"Test embedding shape: {test_embedding.shape}. "
            f"Please verify the expected_dimensions in configuration."
        )
        
        logger.info(f"✓ {model_name} embedding dimensions validated: {actual_dims}")
    
    def get_model_embedding(self, model_name: str, text: str) -> np.ndarray:
        """Get embedding from specified model with validation"""
        logger.debug(f"Computing {model_name} embedding for text length: {len(text)}")
        
        # TODO: ASSUMPTION - Model name exists in loaded models
        # CONTEXT: model_name={model_name}, available_models={list(self.models.keys())}
        assert model_name in self.models, (
            f"Model '{model_name}' not found in loaded models. "
            f"Available models: {list(self.models.keys())}. "
            f"Please ensure model is loaded before computing embeddings."
        )
        
        try:
            embedding = self.models[model_name].encode([text])[0]
            logger.debug(f"{model_name} embedding shape: {embedding.shape}")
            return embedding
        except Exception as e:
            assert False, (
                f"Failed to compute embedding with model '{model_name}'. "
                f"Error: {type(e).__name__}: {e}. "
                f"Text length: {len(text)}. "
                f"Model spec: {self.model_specs[model_name]}. "
                f"Please verify model compatibility and text format."
            )
    
    def load_data_strict(self):
        """Load data with strict validation using configuration"""
        logger.info("Loading data in STRICT mode from configuration")
        
        # Get data configuration
        data_config = self.config.get_data_config()
        source_pattern = data_config["source_pattern"]
        
        # Load scene files using configured pattern
        scene_files = sorted(self.source_path.glob(source_pattern))
        assert len(scene_files) > 0, (
            f"No files found matching pattern '{source_pattern}' in {self.source_path}. "
            f"Available files: {list(self.source_path.glob('*'))}. "
            f"Please verify the source_pattern in configuration."
        )
        
        for scene_file in scene_files:
            scene_id = scene_file.stem
            logger.debug(f"Loading {scene_id}")
            with open(scene_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            assert len(content) > 0, f"Empty scene file: {scene_file}"
            self.text_chunks[scene_id] = content
            
        logger.info(f"✓ Loaded {len(self.text_chunks)} scene chunks using pattern '{source_pattern}'")
        
        # Load facts file
        logger.debug("Loading facts file")
        with open(self.facts_path, 'r', encoding='utf-8') as f:
            facts_data = json.load(f)
        
        # Validate facts structure
        assert isinstance(facts_data, list), "Facts must be a list"
        assert len(facts_data) > 0, "Facts list cannot be empty"
        
        for fact in facts_data:
            assert 'description' in fact, "Each fact must have 'description'"
            assert 'chapter' in fact, "Each fact must have 'chapter'"
            
        self.facts = facts_data
        logger.info(f"✓ Loaded {len(self.facts)} facts")
        
    def compute_embeddings(self):
        """Compute embeddings for all text chunks and facts using configured models"""
        logger.info("Computing embeddings for all data with configured models")
        
        # Embed text chunks with all models
        for chunk_id, text in self.text_chunks.items():
            logger.debug(f"Computing embeddings for {chunk_id}")
            for model_name in self.model_names:
                self.embeddings[model_name][chunk_id] = self.get_model_embedding(model_name, text)
            
        # Embed facts with all models
        for i, fact in enumerate(self.facts):
            fact_id = f"fact_{i}"
            logger.debug(f"Computing embeddings for {fact_id}")
            fact_text = fact['description']
            for model_name in self.model_names:
                self.embeddings[model_name][fact_id] = self.get_model_embedding(model_name, fact_text)
            
        logger.info("✓ All embeddings computed for all configured models")
        
    def evaluate_fact_retrieval(self) -> Dict:
        """Evaluate fact-to-chapter retrieval accuracy with configured models"""
        logger.info("Evaluating fact retrieval accuracy with configured models")
        
        results = {model_name: [] for model_name in self.model_names}
        
        for fact_idx, fact in enumerate(self.facts):
            fact_id = f"fact_{fact_idx}"
            target_chapter = fact['chapter']
            
            logger.debug(f"Evaluating {fact_id} (target: chapter {target_chapter})")
            
            # Process each configured model
            for model_name in self.model_names:
                # Get fact embedding for this model
                fact_emb = self.embeddings[model_name][fact_id]
                
                # Calculate similarities with text chunks
                similarities = {}
                for chunk_id in self.text_chunks.keys():
                    chunk_emb = self.embeddings[model_name][chunk_id]
                    sim = cosine_similarity([fact_emb], [chunk_emb])[0][0]
                    similarities[chunk_id] = sim
                    
                # Get top match
                top_match = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[0]
                
                results[model_name].append({
                    "fact_id": fact_id,
                    "target_chapter": target_chapter,
                    "top_match": top_match,
                    "all_similarities": similarities
                })
            
        return results
            
        return results
        
    def calculate_precision_at_k(self, retrieval_results: Dict, k_values: List[int] = [1, 3, 5]) -> Dict:
        """Calculate Precision@K metrics with proper chapter-to-scene mapping for configured models"""
        logger.info(f"Calculating Precision@K for k={k_values} with configured models")
        
        precision_results = {}
        
        for model_name in self.model_names:
            precision_results[model_name] = {}
            
            for k in k_values:
                correct = 0
                total = len(retrieval_results[model_name])
                
                for result in retrieval_results[model_name]:
                    fact_chapter = result["target_chapter"]
                    top_k_matches = sorted(result["all_similarities"].items(), 
                                         key=lambda x: x[1], reverse=True)[:k]
                    
                    # Debug trace: log raw similarities structure
                    logger.debug(f"TRACE: fact_chapter={fact_chapter}, raw_similarities_keys={list(result['all_similarities'].keys())[:3]}")
                    logger.debug(f"TRACE: top_k_matches (first 2): {top_k_matches[:2]}")
                    
                    # Proper chapter-to-scene mapping: chapter N -> JaneEyre-scene-00N
                    expected_scene = f"JaneEyre-scene-{fact_chapter:03d}"
                    
                    # Check if expected scene is in top-k matches
                    top_k_scene_ids = [match for match, _ in top_k_matches]
                    logger.debug(f"TRACE: expected_scene={expected_scene}, top_k_scene_ids={top_k_scene_ids[:3]}")
                    
                    if expected_scene in top_k_scene_ids:
                        correct += 1
                        logger.debug(f"✓ Correct match: fact {result['fact_id']} chapter {fact_chapter} -> {expected_scene}")
                    else:
                        logger.debug(f"✗ Missed: fact {result['fact_id']} chapter {fact_chapter} -> expected {expected_scene}, got {top_k_scene_ids}")
                        # TODO: ASSUMPTION - expecting scene IDs to match chapter numbers exactly
                        # CONTEXT: fact_chapter={fact_chapter}, available_scenes={list(result['all_similarities'].keys())[:5]}
                        
                precision_results[model_name][f"p@{k}"] = correct / total if total > 0 else 0
                logger.info(f"{model_name} P@{k}: {correct}/{total} = {precision_results[model_name][f'p@{k}']:.3f}")
                
        return precision_results
            
    def analyze_similarity_distributions(self) -> Dict:
        """Analyze similarity score distributions for configured models"""
        logger.info("Analyzing similarity distributions for configured models")
        
        all_similarities = {model_name: [] for model_name in self.model_names}
        
        # Collect all similarity scores
        for model_name in self.model_names:
            for fact_id in [f"fact_{i}" for i in range(len(self.facts))]:
                fact_emb = self.embeddings[model_name][fact_id]
                
                for chunk_id in self.text_chunks.keys():
                    chunk_emb = self.embeddings[model_name][chunk_id]
                    sim = cosine_similarity([fact_emb], [chunk_emb])[0][0]
                    all_similarities[model_name].append(sim)
                    
        # Calculate statistics
        stats = {}
        for model_name in self.model_names:
            sims = np.array(all_similarities[model_name])
            stats[model_name] = {
                "mean": float(np.mean(sims)),
                "std": float(np.std(sims)),
                "min": float(np.min(sims)),
                "max": float(np.max(sims)),
                "q25": float(np.percentile(sims, 25)),
                "q50": float(np.percentile(sims, 50)),
                "q75": float(np.percentile(sims, 75))
            }
            
        return stats
            
        return stats
        
    def run_complete_evaluation(self) -> Dict:
        """Run the complete evaluation pipeline"""
        logger.info("Starting complete evaluation pipeline")
        start_time = time.time()
        
        # Load models and data
        self.load_models_from_config()
        self.load_data_strict()
        
        # Compute embeddings
        self.compute_embeddings()
        
        # Run evaluations
        retrieval_results = self.evaluate_fact_retrieval()
        precision_results = self.calculate_precision_at_k(retrieval_results)
        similarity_stats = self.analyze_similarity_distributions()
        
        # Compile final results with model specifications from config
        model_specs_for_results = {}
        for model_name in self.model_names:
            spec = self.model_specs[model_name]
            model_specs_for_results[model_name] = {
                "huggingface_id": spec["huggingface_id"],
                "display_name": spec["display_name"]
            }
            
        final_results = {
            "experiment_info": {
                "models": model_specs_for_results,
                "data_stats": {
                    "text_chunks": len(self.text_chunks),
                    "facts": len(self.facts)
                },
                "execution_time": time.time() - start_time
            },
            "retrieval_evaluation": retrieval_results,
            "precision_metrics": precision_results,
            "similarity_distributions": similarity_stats
        }
        
        logger.info("✓ Complete evaluation finished")
        return final_results
        
    def save_results(self, results: Dict, filename: str = "experiment_output.json"):
        """Save results to file with numpy type conversion"""
        filepath = self.output_dir / filename
        logger.info(f"Saving results to {filepath}")
        
        # Convert numpy types to Python native types for JSON serialization
        json_safe_results = self._convert_numpy_types(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"✓ Results saved to {filepath}")
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python native types"""
        import numpy as np
        
        # Handle all numpy scalar types explicitly
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            #logger.debug(f"Converting numpy float: {type(obj)} -> float")
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.integer)):
            logger.debug(f"Converting numpy int: {type(obj)} -> int")
            return int(obj)
        elif isinstance(obj, np.ndarray):
            logger.debug(f"Converting numpy array: shape {obj.shape}")
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            logger.debug(f"Converting numpy bool: {type(obj)} -> bool")
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            # Fallback check for any remaining numpy types
            if hasattr(obj, 'dtype') and hasattr(obj, 'item'):
                logger.debug(f"Converting unknown numpy type: {type(obj)} -> Python native")
                return obj.item()
            return obj
        
    def print_diagnostic_summary(self, results: Dict):
        """Print comprehensive diagnostic summary for configured models"""
        print("\n" + "="*80)
        print("EMBEDDING COMPARISON DIAGNOSTIC SUMMARY")
        print("="*80)
        
        info = results["experiment_info"]
        print(f"Models Compared:")
        for model_name, model_info in info['models'].items():
            print(f"  • {model_name}: {model_info['display_name']} ({model_info['huggingface_id']})")
        print(f"Data: {info['data_stats']['text_chunks']} chunks, {info['data_stats']['facts']} facts")
        print(f"Execution Time: {info['execution_time']:.2f}s")
        
        print(f"\nPRECISION METRICS:")
        for model_name in self.model_names:
            print(f"  {model_name.upper()}:")
            for metric, value in results["precision_metrics"][model_name].items():
                print(f"    {metric}: {value:.3f}")
                
        print(f"\nSIMILARITY DISTRIBUTIONS:")
        for model_name in self.model_names:
            stats = results["similarity_distributions"][model_name]
            print(f"  {model_name.upper()}:")
            print(f"    Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"    Q25/Q50/Q75: {stats['q25']:.3f}/{stats['q50']:.3f}/{stats['q75']:.3f}")
            
        print("="*80)


def parse_arguments():
    """Parse command line arguments with configuration system support"""
    parser = argparse.ArgumentParser(
        description="Embedding Model Comparison Experiment - Configuration-Driven",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard usage with configuration file (RECOMMENDED)
  python embedding_comparison_experiment.py --config comparison.json
  python embedding_comparison_experiment.py --config comparison.json --output-dir results/
  
  # TEMPORARY: Legacy mode for bridge test validation ONLY (WILL BE REMOVED)
  python embedding_comparison_experiment.py --legacy-mode
  python embedding_comparison_experiment.py --legacy-mode --output-dir bridge_test_results
        """
    )
    
    # Configuration system arguments
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file (REQUIRED for normal operation)"
    )
    config_group.add_argument(
        "--legacy-mode",
        action="store_true",
        help="TEMPORARY: Use legacy hardcoded configuration for bridge test validation ONLY (WILL BE REMOVED)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (overrides config file setting, default: current directory)"
    )
    
    # Legacy data path argument (deprecated but maintained for bridge test compatibility)
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data directory (DEPRECATED: use config file data_config.base_path instead)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function with configuration system support"""
    # Parse command line arguments
    args = parse_arguments()
    
    print("🚀 EMBEDDING COMPARISON EXPERIMENT - CONFIGURATION-DRIVEN")
    print("STRICT VALIDATION | NO FALLBACKS | COMPREHENSIVE DIAGNOSTICS")
    if args.output_dir:
        print(f"OUTPUT DIR: {args.output_dir}")
    print("-" * 80)
    
    try:
        # TODO: Initialize configuration system with strict validation
        logger.info("Loading experiment configuration...")
        if args.legacy_mode:
            logger.warning("⚠️  LEGACY MODE ACTIVATED - FOR BRIDGE TEST VALIDATION ONLY")
            config = ComparisonConfig(legacy_mode=True)
        else:
            logger.info(f"Loading configuration from: {args.config}")
            config = ComparisonConfig(config_path=args.config)
            
        # TODO: ASSUMPTION - Configuration loaded successfully, proceeding with experiment
        # CONTEXT: config_source={config.config_source}, legacy_mode={args.legacy_mode}
        
        # DEPRECATED: CLI data path override (use configuration instead)
        if args.data_path:
            logger.warning("⚠️  --data-path argument is DEPRECATED, use config file data_config.base_path instead")
            logger.warning("⚠️  This argument will be ignored in favor of configuration")
            
        # Initialize comparison with configuration-driven approach
        logger.info("Initializing configuration-driven EmbeddingComparison...")
        comparator = EmbeddingComparison(
            config=config,
            output_dir=args.output_dir
        )
        
        # Log experiment information
        exp_info = config.get_experiment_info()
        logger.info(f"Experiment: {exp_info['name']} - {exp_info['description']}")
        
        model_a, model_b = config.get_model_specs()
        logger.info(f"Model A: {model_a['display_name']} ({model_a['huggingface_id']})")
        logger.info(f"Model B: {model_b['display_name']} ({model_b['huggingface_id']})")
        
        # Run complete evaluation
        results = comparator.run_complete_evaluation()
        
        # Save results
        comparator.save_results(results)
        
        # Print diagnostic summary
        comparator.print_diagnostic_summary(results)
        
        print("\n✅ EXPERIMENT COMPLETED SUCCESSFULLY")
        
        # Exit with success code for bridge test
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"EXPERIMENT FAILED: {e}")
        print(f"\n❌ EXPERIMENT FAILED: {e}")
        
        # TODO: ASSUMPTION - Bridge test expects proper exit codes
        # CONTEXT: exception={type(e).__name__}, config_loaded={locals().get('config') is not None}
        # Exit with failure code for bridge test
        sys.exit(1)


if __name__ == "__main__":
    main()
