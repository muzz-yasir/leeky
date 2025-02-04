"""Configuration management for leeky."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from .types import ConfigurationError, EngineConfig, TestConfig

class Configuration:
    """Handles loading and validation of configuration from YAML and environment variables."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file and environment variables.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default locations.
            
        Raises:
            ConfigurationError: If configuration is invalid or cannot be loaded.
        """
        self.config_data: Dict[str, Any] = {}
        self._load_defaults()
        
        if config_path:
            self._load_from_yaml(config_path)
        else:
            self._load_from_default_locations()
            
        self._load_from_env()
        self._validate_config()

    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self.config_data = {
            "data_loader": {
                "batch_size": 100,
                "cache_enabled": True,
                "cache_dir": ".cache",
            },
            "test_runner": {
                "batch_size": 10,
                "max_retries": 3,
                "timeout": 30.0,
                "sampling_strategy": "random",
                "sampling_parameters": {},
            },
            "engine": {
                "engine_type": "openai",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 100,
                },
            },
        }

    def _load_from_yaml(self, config_path: str) -> None:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
            
        Raises:
            ConfigurationError: If file cannot be read or contains invalid YAML.
        """
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    self.config_data.update(yaml_config)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {config_path}: {str(e)}")

    def _load_from_default_locations(self) -> None:
        """Load configuration from default locations."""
        default_locations = [
            Path.cwd() / "config.yaml",
            Path.home() / ".leeky" / "config.yaml",
            Path("/etc/leeky/config.yaml"),
        ]
        
        for path in default_locations:
            if path.is_file():
                self._load_from_yaml(str(path))
                break

    def _load_from_env(self) -> None:
        """Load configuration from environment variables.
        
        Environment variables should be prefixed with LEEKY_ and use underscore
        separated uppercase names matching the nested config structure.
        
        Example:
            LEEKY_ENGINE_API_KEY=xyz will set config_data["engine"]["api_key"]=xyz
        """
        for key, value in os.environ.items():
            if key.startswith("LEEKY_"):
                parts = key[6:].lower().split("_")
                current = self.config_data
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                # Convert value based on existing type
                final_key = parts[-1]
                if final_key in current and isinstance(current[final_key], int):
                    current[final_key] = int(value)
                elif final_key in current and isinstance(current[final_key], float):
                    current[final_key] = float(value)
                else:
                    current[final_key] = value

    def _validate_config(self) -> None:
        """Validate the loaded configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid.
        """
        required_sections = ["data_loader", "test_runner", "engine"]
        for section in required_sections:
            if section not in self.config_data:
                raise ConfigurationError(f"Missing required config section: {section}")

    def get_engine_config(self) -> EngineConfig:
        """Get engine configuration.
        
        Returns:
            EngineConfig object with engine configuration.
            
        Raises:
            ConfigurationError: If engine configuration is invalid.
        """
        try:
            engine_data = self.config_data["engine"]
            return EngineConfig(
                engine_type=engine_data["engine_type"],
                api_key=os.getenv("LEEKY_ENGINE_API_KEY", engine_data.get("api_key")),
                parameters=engine_data.get("parameters", {})
            )
        except KeyError as e:
            raise ConfigurationError(f"Missing required engine config key: {str(e)}")

    def get_test_config(self) -> TestConfig:
        """Get test runner configuration.
        
        Returns:
            TestConfig object with test runner configuration.
            
        Raises:
            ConfigurationError: If test runner configuration is invalid.
        """
        try:
            test_data = self.config_data["test_runner"]
            return TestConfig(
                batch_size=test_data["batch_size"],
                max_retries=test_data["max_retries"],
                timeout=float(test_data["timeout"]),
                sampling_strategy=test_data["sampling_strategy"],
                sampling_parameters=test_data.get("sampling_parameters", {})
            )
        except KeyError as e:
            raise ConfigurationError(f"Missing required test config key: {str(e)}")

    def get_data_loader_config(self) -> Dict[str, Any]:
        """Get data loader configuration.
        
        Returns:
            Dictionary with data loader configuration.
            
        Raises:
            ConfigurationError: If data loader configuration is invalid.
        """
        if "data_loader" not in self.config_data:
            raise ConfigurationError("Missing data loader configuration")
        return self.config_data["data_loader"]
