"""Tests for configuration module."""

import pytest
from pathlib import Path
import os
from leeky.core.config import Configuration
from leeky.core.types import ConfigurationError

def test_load_defaults():
    """Test loading default configuration."""
    config = Configuration()
    assert config.config_data is not None
    assert "data_loader" in config.config_data
    assert "test_runner" in config.config_data
    assert "engine" in config.config_data

def test_load_from_yaml(tmp_path):
    """Test loading configuration from YAML file."""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("""
data_loader:
  batch_size: 50
  cache_enabled: false
test_runner:
  batch_size: 5
  max_retries: 2
engine:
  engine_type: "test"
    """)
    
    config = Configuration(str(config_path))
    assert config.config_data["data_loader"]["batch_size"] == 50
    assert config.config_data["test_runner"]["max_retries"] == 2
    assert config.config_data["engine"]["engine_type"] == "test"

def test_load_from_env():
    """Test loading configuration from environment variables."""
    os.environ["LEEKY_ENGINE_API_KEY"] = "test-key"
    os.environ["LEEKY_TEST_RUNNER_BATCH_SIZE"] = "15"
    
    config = Configuration()
    engine_config = config.get_engine_config()
    test_config = config.get_test_config()
    
    assert engine_config.api_key == "test-key"
    assert test_config.batch_size == 15
    
    # Cleanup
    del os.environ["LEEKY_ENGINE_API_KEY"]
    del os.environ["LEEKY_TEST_RUNNER_BATCH_SIZE"]

def test_invalid_config():
    """Test handling of invalid configuration."""
    with pytest.raises(ConfigurationError):
        config = Configuration()
        config.config_data = {}  # Clear required sections
        config._validate_config()

def test_get_configs():
    """Test getting specific configurations."""
    config = Configuration()
    
    engine_config = config.get_engine_config()
    assert engine_config.engine_type == "openai"
    
    test_config = config.get_test_config()
    assert test_config.batch_size == 10
    assert test_config.max_retries == 3
    
    data_loader_config = config.get_data_loader_config()
    assert data_loader_config["batch_size"] == 100
