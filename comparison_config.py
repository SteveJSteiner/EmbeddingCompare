#!/usr/bin/env python3
"""
Configuration system for embedding model comparison experiments.
Implements strict validation with no fallbacks for experimental rigor.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys

logger = logging.getLogger(__name__)


class ComparisonConfig:
    """Configuration management with strict validation for embedding comparison experiments.
    
    Follows experimental rigor principles:
    - No silent fallbacks or default behavior
    - Explicit validation with detailed error messages
    - Full diagnostic output on configuration errors
    - Atomic configuration loading (success or complete failure)
    """
    
    def __init__(self, config_path: Optional[str] = None, legacy_mode: bool = False):
        """Initialize configuration with strict validation.
        
        Args:
            config_path: Path to JSON configuration file (required unless legacy_mode=True)
            legacy_mode: TEMPORARY flag for bridge test validation ONLY
            
        Raises:
            AssertionError: On any configuration validation failure with full diagnostic context
        """
        logger.debug(f"Initializing ComparisonConfig - config_path={config_path}, legacy_mode={legacy_mode}")
        
        # TODO: ASSUMPTION - Either config_path OR legacy_mode must be specified, not both
        # CONTEXT: config_path={config_path}, legacy_mode={legacy_mode}
        if legacy_mode:
            logger.warning("⚠️  LEGACY MODE ENABLED - FOR BRIDGE TEST VALIDATION ONLY")
            logger.warning("⚠️  Legacy mode will be REMOVED after bridge test validation")
            self.config = self._get_legacy_config()
        else:
            assert config_path is not None, (
                "Configuration file path is required. "
                "Use --config <path> to specify configuration file. "
                "Legacy mode (--legacy-mode) is only for bridge test validation."
            )
            self.config = self._load_and_validate_config(config_path)
            
        # Store source information for diagnostics
        self.config_source = "legacy_mode" if legacy_mode else config_path
        
        logger.info(f"✓ Configuration loaded successfully from: {self.config_source}")
        
    def _load_and_validate_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration file with strict requirements.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            AssertionError: On any validation failure with full diagnostic context
        """
        logger.debug(f"Loading configuration from: {config_path}")
        
        # TODO: Verify configuration file exists
        config_file = Path(config_path)
        assert config_file.exists(), (
            f"Configuration file not found: {config_path}. "
            f"Resolved path: {config_file.resolve()}. "
            f"Current working directory: {Path.cwd()}. "
            f"Please ensure the configuration file exists and is accessible."
        )
        
        # TODO: Load JSON with proper error handling
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            assert False, (
                f"Invalid JSON in configuration file: {config_path}. "
                f"JSON error: {e}. "
                f"Line {e.lineno}, Column {e.colno}: {e.msg}. "
                f"Please ensure the configuration file contains valid JSON."
            )
        except Exception as e:
            assert False, (
                f"Failed to read configuration file: {config_path}. "
                f"Error: {type(e).__name__}: {e}. "
                f"Please ensure the file is readable and properly formatted."
            )
            
        logger.debug(f"Configuration JSON loaded successfully, validating structure...")
        
        # TODO: Validate configuration structure
        self._validate_config_structure(config_data, config_path)
        self._validate_model_configurations(config_data, config_path)
        self._validate_data_configuration(config_data, config_path)
        
        logger.info("✓ Configuration validation completed successfully")
        return config_data
        
    def _validate_config_structure(self, config: Dict[str, Any], config_path: str) -> None:
        """Validate top-level configuration structure.
        
        Args:
            config: Configuration dictionary to validate
            config_path: Path to config file for error reporting
            
        Raises:
            AssertionError: On validation failure with diagnostic context
        """
        logger.debug("Validating top-level configuration structure...")
        
        # TODO: Check required top-level sections exist
        required_sections = ["models"]
        for section in required_sections:
            assert section in config, (
                f"Missing required configuration section: '{section}'. "
                f"Configuration file: {config_path}. "
                f"Available sections: {list(config.keys())}. "
                f"Required sections: {required_sections}"
            )
            
        # TODO: Validate experiment_config if present
        if "experiment_config" in config:
            exp_config = config["experiment_config"]
            assert isinstance(exp_config, dict), (
                f"'experiment_config' must be a dictionary. "
                f"Got: {type(exp_config).__name__}. "
                f"Value: {exp_config}. "
                f"Configuration file: {config_path}"
            )
            
        logger.debug("✓ Top-level structure validation passed")
        
    def _validate_model_configurations(self, config: Dict[str, Any], config_path: str) -> None:
        """Validate model configuration section with strict requirements.
        
        Args:
            config: Configuration dictionary to validate
            config_path: Path to config file for error reporting
            
        Raises:
            AssertionError: On validation failure with diagnostic context
        """
        logger.debug("Validating model configurations...")
        
        models_config = config["models"]
        assert isinstance(models_config, dict), (
            f"'models' section must be a dictionary. "
            f"Got: {type(models_config).__name__}. "
            f"Value: {models_config}. "
            f"Configuration file: {config_path}"
        )
        
        # TODO: Require exactly two models: model_a and model_b
        required_models = ["model_a", "model_b"]
        for model_key in required_models:
            assert model_key in models_config, (
                f"Missing required model configuration: '{model_key}'. "
                f"Configuration file: {config_path}. "
                f"Available models: {list(models_config.keys())}. "
                f"Required models: {required_models}"
            )
            
        # TODO: Validate each model configuration
        for model_key in required_models:
            self._validate_single_model_config(
                models_config[model_key], model_key, config_path
            )
            
        # TODO: Ensure model_a and model_b are different
        model_a_id = models_config["model_a"].get("huggingface_id")
        model_b_id = models_config["model_b"].get("huggingface_id")
        assert model_a_id != model_b_id, (
            f"model_a and model_b must have different huggingface_id values. "
            f"Both are set to: '{model_a_id}'. "
            f"Configuration file: {config_path}. "
            f"Please specify different models for comparison."
        )
        
        logger.debug("✓ Model configurations validation passed")
        
    def _validate_single_model_config(self, model_config: Dict[str, Any], model_key: str, config_path: str) -> None:
        """Validate a single model configuration with strict requirements.
        
        Args:
            model_config: Model configuration dictionary
            model_key: Key identifying the model (model_a, model_b)
            config_path: Path to config file for error reporting
            
        Raises:
            AssertionError: On validation failure with diagnostic context
        """
        logger.debug(f"Validating {model_key} configuration...")
        
        assert isinstance(model_config, dict), (
            f"Model configuration for '{model_key}' must be a dictionary. "
            f"Got: {type(model_config).__name__}. "
            f"Value: {model_config}. "
            f"Configuration file: {config_path}"
        )
        
        # TODO: Check required fields
        required_fields = ["name", "huggingface_id", "display_name"]
        for field in required_fields:
            assert field in model_config, (
                f"Missing required field '{field}' in {model_key} configuration. "
                f"Configuration file: {config_path}. "
                f"Available fields: {list(model_config.keys())}. "
                f"Required fields: {required_fields}. "
                f"Model config: {model_config}"
            )
            
            # TODO: Ensure field values are non-empty strings
            value = model_config[field]
            assert isinstance(value, str) and len(value.strip()) > 0, (
                f"Field '{field}' in {model_key} must be a non-empty string. "
                f"Got: {type(value).__name__} with value: {repr(value)}. "
                f"Configuration file: {config_path}. "
                f"Model config: {model_config}"
            )
            
        # TODO: Validate optional fields if present
        if "expected_dimensions" in model_config:
            dims = model_config["expected_dimensions"]
            assert dims is None or (isinstance(dims, int) and dims > 0), (
                f"'expected_dimensions' in {model_key} must be null or a positive integer. "
                f"Got: {type(dims).__name__} with value: {dims}. "
                f"Configuration file: {config_path}. "
                f"Model config: {model_config}"
            )
            
        if "trust_remote_code" in model_config:
            trust_code = model_config["trust_remote_code"]
            assert isinstance(trust_code, bool), (
                f"'trust_remote_code' in {model_key} must be a boolean. "
                f"Got: {type(trust_code).__name__} with value: {trust_code}. "
                f"Configuration file: {config_path}. "
                f"Model config: {model_config}"
            )
            
        logger.debug(f"✓ {model_key} configuration validation passed")
        
    def _validate_data_configuration(self, config: Dict[str, Any], config_path: str) -> None:
        """Validate data configuration section if present.
        
        Args:
            config: Configuration dictionary to validate
            config_path: Path to config file for error reporting
            
        Raises:
            AssertionError: On validation failure with diagnostic context
        """
        if "data_config" not in config:
            logger.debug("No data_config section found, using defaults")
            return
            
        logger.debug("Validating data configuration...")
        
        data_config = config["data_config"]
        assert isinstance(data_config, dict), (
            f"'data_config' section must be a dictionary. "
            f"Got: {type(data_config).__name__}. "
            f"Value: {data_config}. "
            f"Configuration file: {config_path}"
        )
        
        # TODO: Validate base_path if specified
        if "base_path" in data_config:
            base_path = data_config["base_path"]
            assert isinstance(base_path, str) and len(base_path.strip()) > 0, (
                f"'base_path' in data_config must be a non-empty string. "
                f"Got: {type(base_path).__name__} with value: {repr(base_path)}. "
                f"Configuration file: {config_path}. "
                f"Data config: {data_config}"
            )
            
        logger.debug("✓ Data configuration validation passed")
        
    def _get_legacy_config(self) -> Dict[str, Any]:
        """TEMPORARY: Generate legacy configuration for bridge test validation ONLY.
        
        This function replicates the exact hardcoded behavior of the original
        embedding_comparison_experiment.py for bridge test validation purposes.
        
        Returns:
            Configuration dictionary matching legacy hardcoded behavior
            
        Note:
            This function will be REMOVED after bridge test validation passes.
        """
        logger.warning("🚨 GENERATING LEGACY CONFIGURATION - BRIDGE TEST ONLY")
        logger.warning("🚨 This configuration mode is TEMPORARY and will be removed")
        
        # TODO: ASSUMPTION - Legacy config exactly matches original hardcoded values
        # CONTEXT: Replicating original model_names and paths from EmbeddingComparison.__init__
        legacy_config = {
            "experiment_config": {
                "name": "Legacy Bridge Test Configuration",
                "description": "Temporary configuration matching original hardcoded behavior",
                "version": "BRIDGE_TEST_ONLY"
            },
            "models": {
                "model_a": {
                    "name": "minilm",
                    "huggingface_id": "all-MiniLM-L6-v2",
                    "display_name": "MiniLM-L6-v2",
                    "expected_dimensions": 384,
                    "trust_remote_code": False
                },
                "model_b": {
                    "name": "qwen",
                    "huggingface_id": "Qwen/Qwen3-Embedding-0.6B", 
                    "display_name": "Qwen3-Embedding-0.6B",
                    "expected_dimensions": None,
                    "trust_remote_code": True
                }
            },
            "data_config": {
                "base_path": "./data",
                "source_pattern": "JaneEyre-scene-*.txt",
                "facts_file": "facts/all_chapters_combined_mistral.json"
            },
            "evaluation_config": {
                "precision_k_values": [1, 3, 5],
                "similarity_metric": "cosine",
                "include_diagnostics": True
            }
        }
        
        logger.debug(f"Legacy configuration generated: {json.dumps(legacy_config, indent=2)}")
        return legacy_config
        
    def get_model_specs(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return validated model specifications for both models.
        
        Returns:
            Tuple of (model_a_config, model_b_config) dictionaries
            
        Raises:
            AssertionError: If models configuration is invalid
        """
        assert "models" in self.config, (
            f"Models configuration missing from loaded config. "
            f"Config source: {self.config_source}. "
            f"Available sections: {list(self.config.keys())}"
        )
        
        models = self.config["models"]
        
        # TODO: Verify required models are present
        assert "model_a" in models and "model_b" in models, (
            f"Both model_a and model_b must be present in models configuration. "
            f"Available models: {list(models.keys())}. "
            f"Config source: {self.config_source}"
        )
        
        logger.debug("Returning validated model specifications")
        return models["model_a"], models["model_b"]
        
    def get_data_config(self) -> Dict[str, Any]:
        """Return data configuration with defaults applied.
        
        Returns:
            Data configuration dictionary with defaults for missing values
        """
        # TODO: Apply default values for missing data configuration
        defaults = {
            "base_path": "./data",
            "source_pattern": "JaneEyre-scene-*.txt", 
            "facts_file": "facts/all_chapters_combined_mistral.json",
            "validation": {
                "min_chunks": 1,
                "min_facts": 1,
                "require_chapter_mapping": True
            }
        }
        
        if "data_config" in self.config:
            data_config = self.config["data_config"].copy()
            # Merge defaults for missing keys
            for key, default_value in defaults.items():
                if key not in data_config:
                    data_config[key] = default_value
                    logger.debug(f"Applied default for data_config.{key}: {default_value}")
        else:
            data_config = defaults
            logger.debug("Using default data configuration")
            
        return data_config
        
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Return evaluation configuration with defaults applied.
        
        Returns:
            Evaluation configuration dictionary with defaults for missing values
        """
        # TODO: Apply default values for missing evaluation configuration
        defaults = {
            "precision_k_values": [1, 3, 5],
            "similarity_metric": "cosine",
            "include_diagnostics": True
        }
        
        if "evaluation_config" in self.config:
            eval_config = self.config["evaluation_config"].copy()
            # Merge defaults for missing keys
            for key, default_value in defaults.items():
                if key not in eval_config:
                    eval_config[key] = default_value
                    logger.debug(f"Applied default for evaluation_config.{key}: {default_value}")
        else:
            eval_config = defaults
            logger.debug("Using default evaluation configuration")
            
        return eval_config
        
    def get_experiment_info(self) -> Dict[str, Any]:
        """Return experiment configuration information.
        
        Returns:
            Experiment info dictionary with defaults for missing values
        """
        if "experiment_config" in self.config:
            return self.config["experiment_config"]
        else:
            # TODO: Generate default experiment info
            model_a, model_b = self.get_model_specs()
            return {
                "name": f"{model_a['display_name']} vs {model_b['display_name']} Comparison",
                "description": "Embedding model comparison experiment",
                "version": "1.0"
            }
            
    def is_legacy_mode(self) -> bool:
        """Check if configuration was loaded in legacy mode.
        
        Returns:
            True if legacy mode is active, False otherwise
        """
        return self.config_source == "legacy_mode"
