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

# Configure diagnostic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingComparison:
    def __init__(self, base_path: str = "./data", output_dir: str = None):
        logger.info("Initializing EmbeddingComparison - FRAGILE MODE")
        self.base_path = Path(base_path)
        self.source_path = self.base_path / "source"
        self.facts_path = self.base_path / "facts" / "all_chapters_combined_mistral.json"
        
        # Output directory configuration
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
        else:
            self.output_dir = Path(".")
        
        # NO FALLBACKS - Paths must exist exactly
        self._verify_paths_strict()
        
        # Model configuration - exact names required
        self.model_names = {
            "minilm": "all-MiniLM-L6-v2",
            "qwen": "Qwen/Qwen3-Embedding-0.6B"
        }
        
        # Storage containers
        self.text_chunks = {}
        self.facts = []
        self.embeddings = {"minilm": {}, "qwen": {}}
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
                    
    def load_models_strict(self):
        """Load both models with strict requirements - NO FALLBACKS"""
        logger.info("Loading models in STRICT mode")
        
        # Load MiniLM - must succeed exactly
        logger.debug("Loading all-MiniLM-L6-v2...")
        self.minilm_model = SentenceTransformer(self.model_names["minilm"])
        logger.info(f"✓ MiniLM loaded: {self.minilm_model}")
        
        # Load Qwen using SentenceTransformer - must succeed exactly  
        logger.debug("Loading Qwen/Qwen3-Embedding-0.6B...")
        self.qwen_model = SentenceTransformer(self.model_names["qwen"])
        logger.info(f"✓ Qwen loaded: {self.qwen_model}")
        
    def load_data_strict(self):
        """Load Jane Eyre data with strict validation"""
        logger.info("Loading Jane Eyre data in STRICT mode")
        
        # Load scene files
        scene_files = sorted(self.source_path.glob("JaneEyre-scene-*.txt"))
        for scene_file in scene_files:
            scene_id = scene_file.stem
            logger.debug(f"Loading {scene_id}")
            with open(scene_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            assert len(content) > 0, f"Empty scene file: {scene_file}"
            self.text_chunks[scene_id] = content
            
        logger.info(f"✓ Loaded {len(self.text_chunks)} scene chunks")
        
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
        
    def get_minilm_embedding(self, text: str) -> np.ndarray:
        """Get embedding from MiniLM model"""
        logger.debug(f"Computing MiniLM embedding for text length: {len(text)}")
        embedding = self.minilm_model.encode([text])[0]
        logger.debug(f"MiniLM embedding shape: {embedding.shape}")
        return embedding
        
    def get_qwen_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Qwen model"""
        logger.debug(f"Computing Qwen embedding for text length: {len(text)}")
        embedding = self.qwen_model.encode([text])[0]
        logger.debug(f"Qwen embedding shape: {embedding.shape}")
        return embedding
        
    def compute_embeddings(self):
        """Compute embeddings for all text chunks and facts"""
        logger.info("Computing embeddings for all data")
        
        # Embed text chunks
        for chunk_id, text in self.text_chunks.items():
            logger.debug(f"Computing embeddings for {chunk_id}")
            self.embeddings["minilm"][chunk_id] = self.get_minilm_embedding(text)
            self.embeddings["qwen"][chunk_id] = self.get_qwen_embedding(text)
            
        # Embed facts
        for i, fact in enumerate(self.facts):
            fact_id = f"fact_{i}"
            logger.debug(f"Computing embeddings for {fact_id}")
            fact_text = fact['description']
            self.embeddings["minilm"][fact_id] = self.get_minilm_embedding(fact_text)
            self.embeddings["qwen"][fact_id] = self.get_qwen_embedding(fact_text)
            
        logger.info("✓ All embeddings computed")
        
    def evaluate_fact_retrieval(self) -> Dict:
        """Evaluate fact-to-chapter retrieval accuracy"""
        logger.info("Evaluating fact retrieval accuracy")
        
        results = {"minilm": [], "qwen": []}
        
        for fact_idx, fact in enumerate(self.facts):
            fact_id = f"fact_{fact_idx}"
            target_chapter = fact['chapter']
            
            logger.debug(f"Evaluating {fact_id} (target: chapter {target_chapter})")
            
            # Get fact embedding for both models
            fact_emb_minilm = self.embeddings["minilm"][fact_id]
            fact_emb_qwen = self.embeddings["qwen"][fact_id]
            
            # Calculate similarities with text chunks
            similarities_minilm = {}
            similarities_qwen = {}
            
            for chunk_id in self.text_chunks.keys():
                chunk_emb_minilm = self.embeddings["minilm"][chunk_id]
                chunk_emb_qwen = self.embeddings["qwen"][chunk_id]
                
                sim_minilm = cosine_similarity([fact_emb_minilm], [chunk_emb_minilm])[0][0]
                sim_qwen = cosine_similarity([fact_emb_qwen], [chunk_emb_qwen])[0][0]
                
                similarities_minilm[chunk_id] = sim_minilm
                similarities_qwen[chunk_id] = sim_qwen
                
            # Get top matches
            top_minilm = sorted(similarities_minilm.items(), key=lambda x: x[1], reverse=True)
            top_qwen = sorted(similarities_qwen.items(), key=lambda x: x[1], reverse=True)
            
            results["minilm"].append({
                "fact_id": fact_id,
                "target_chapter": target_chapter,
                "top_match": top_minilm[0],
                "all_similarities": similarities_minilm
            })
            
            results["qwen"].append({
                "fact_id": fact_id,
                "target_chapter": target_chapter,
                "top_match": top_qwen[0],
                "all_similarities": similarities_qwen
            })
            
        return results
        
    def calculate_precision_at_k(self, retrieval_results: Dict, k_values: List[int] = [1, 3, 5]) -> Dict:
        """Calculate Precision@K metrics with proper chapter-to-scene mapping"""
        logger.info(f"Calculating Precision@K for k={k_values}")
        
        precision_results = {}
        
        for model_name in ["minilm", "qwen"]:
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
        """Analyze similarity score distributions"""
        logger.info("Analyzing similarity distributions")
        
        all_similarities = {"minilm": [], "qwen": []}
        
        # Collect all similarity scores
        for model in ["minilm", "qwen"]:
            for fact_id in [f"fact_{i}" for i in range(len(self.facts))]:
                fact_emb = self.embeddings[model][fact_id]
                
                for chunk_id in self.text_chunks.keys():
                    chunk_emb = self.embeddings[model][chunk_id]
                    sim = cosine_similarity([fact_emb], [chunk_emb])[0][0]
                    all_similarities[model].append(sim)
                    
        # Calculate statistics
        stats = {}
        for model in ["minilm", "qwen"]:
            sims = np.array(all_similarities[model])
            stats[model] = {
                "mean": float(np.mean(sims)),
                "std": float(np.std(sims)),
                "min": float(np.min(sims)),
                "max": float(np.max(sims)),
                "q25": float(np.percentile(sims, 25)),
                "q50": float(np.percentile(sims, 50)),
                "q75": float(np.percentile(sims, 75))
            }
            
        return stats
        
    def run_complete_evaluation(self) -> Dict:
        """Run the complete evaluation pipeline"""
        logger.info("Starting complete evaluation pipeline")
        start_time = time.time()
        
        # Load models and data
        self.load_models_strict()
        self.load_data_strict()
        
        # Compute embeddings
        self.compute_embeddings()
        
        # Run evaluations
        retrieval_results = self.evaluate_fact_retrieval()
        precision_results = self.calculate_precision_at_k(retrieval_results)
        similarity_stats = self.analyze_similarity_distributions()
        
        # Compile final results
        final_results = {
            "experiment_info": {
                "models": self.model_names,
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
        """Print comprehensive diagnostic summary"""
        print("\n" + "="*80)
        print("EMBEDDING COMPARISON DIAGNOSTIC SUMMARY")
        print("="*80)
        
        info = results["experiment_info"]
        print(f"Models Compared:")
        print(f"  • MiniLM: {info['models']['minilm']}")
        print(f"  • Qwen: {info['models']['qwen']}")
        print(f"Data: {info['data_stats']['text_chunks']} chunks, {info['data_stats']['facts']} facts")
        print(f"Execution Time: {info['execution_time']:.2f}s")
        
        print(f"\nPRECISION METRICS:")
        for model in ["minilm", "qwen"]:
            print(f"  {model.upper()}:")
            for metric, value in results["precision_metrics"][model].items():
                print(f"    {metric}: {value:.3f}")
                
        print(f"\nSIMILARITY DISTRIBUTIONS:")
        for model in ["minilm", "qwen"]:
            stats = results["similarity_distributions"][model]
            print(f"  {model.upper()}:")
            print(f"    Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"    Q25/Q50/Q75: {stats['q25']:.3f}/{stats['q50']:.3f}/{stats['q75']:.3f}")
            
        print("="*80)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Embedding Model Comparison Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python embedding_comparison_experiment.py
  python embedding_comparison_experiment.py --output-dir bridge_test_results
        """
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: current directory)"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to data directory (default: ./data)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function with CLI support"""
    # Parse command line arguments
    args = parse_arguments()
    
    print("🚀 EMBEDDING COMPARISON EXPERIMENT - FRAGILE MODE")
    print("NO FALLBACKS | EXACT REQUIREMENTS | COMPREHENSIVE DIAGNOSTICS")
    if args.output_dir:
        print(f"OUTPUT DIR: {args.output_dir}")
    print("-" * 80)
    
    try:
        # Initialize comparison with CLI parameters
        comparator = EmbeddingComparison(
            base_path=args.data_path,
            output_dir=args.output_dir
        )
        
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
        # CONTEXT: exception={type(e).__name__}, args.output_dir={args.output_dir}
        # Exit with failure code for bridge test
        sys.exit(1)


if __name__ == "__main__":
    main()
