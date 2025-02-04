"""Tests for prompt manager module."""

import pytest
from datetime import datetime
from pathlib import Path
import json
from ..prompt_manager import PromptManager
from ..types import PromptTemplate, PromptError

@pytest.fixture
def prompt_manager():
    """Create a prompt manager instance for testing."""
    return PromptManager()

@pytest.fixture
def test_template():
    """Create a test template."""
    return {
        "name": "test_prompt",
        "template": "Test prompt with {param}",
        "parameters": {"param": "value"},
        "metadata": {"purpose": "testing"},
        "version": "1.0.0"
    }

def test_register_template(prompt_manager, test_template):
    """Test registering a new template."""
    template = prompt_manager.register_template(**test_template)
    
    assert isinstance(template, PromptTemplate)
    assert template.name == test_template["name"]
    assert template.template == test_template["template"]
    assert template.parameters == test_template["parameters"]
    assert template.metadata == test_template["metadata"]
    assert template.version == test_template["version"]
    assert isinstance(template.created_at, datetime)

def test_register_duplicate_template(prompt_manager, test_template):
    """Test handling of duplicate template registration."""
    prompt_manager.register_template(**test_template)
    
    with pytest.raises(PromptError):
        prompt_manager.register_template(**test_template)

def test_get_template(prompt_manager, test_template):
    """Test getting a template by name."""
    prompt_manager.register_template(**test_template)
    template = prompt_manager.get_template(test_template["name"])
    
    assert isinstance(template, PromptTemplate)
    assert template.name == test_template["name"]

def test_get_nonexistent_template(prompt_manager):
    """Test getting a nonexistent template."""
    with pytest.raises(PromptError):
        prompt_manager.get_template("nonexistent")

def test_get_all_templates(prompt_manager, test_template):
    """Test getting all templates."""
    prompt_manager.register_template(**test_template)
    templates = prompt_manager.get_all_templates()
    
    assert len(templates) == 1
    assert isinstance(templates[0], PromptTemplate)
    assert templates[0].name == test_template["name"]

def test_update_template(prompt_manager, test_template):
    """Test updating an existing template."""
    prompt_manager.register_template(**test_template)
    
    updated = prompt_manager.update_template(
        test_template["name"],
        template="Updated prompt with {param}",
        version="1.0.1"
    )
    
    assert updated.template == "Updated prompt with {param}"
    assert updated.version == "1.0.1"
    assert updated.parameters == test_template["parameters"]  # Unchanged

def test_delete_template(prompt_manager, test_template):
    """Test deleting a template."""
    prompt_manager.register_template(**test_template)
    prompt_manager.delete_template(test_template["name"])
    
    with pytest.raises(PromptError):
        prompt_manager.get_template(test_template["name"])

def test_track_performance(prompt_manager, test_template):
    """Test tracking template performance."""
    prompt_manager.register_template(**test_template)
    
    prompt_manager.track_performance(
        test_template["name"],
        score=0.85,
        metadata={"test_case": "case1"}
    )
    
    metrics = prompt_manager.get_performance_metrics(test_template["name"])
    assert len(metrics) == 1
    assert metrics[0]["score"] == 0.85
    assert metrics[0]["metadata"]["test_case"] == "case1"

def test_track_performance_nonexistent_template(prompt_manager):
    """Test tracking performance for nonexistent template."""
    with pytest.raises(PromptError):
        prompt_manager.track_performance("nonexistent", score=0.5)

def test_save_load_templates(prompt_manager, test_template, tmp_path):
    """Test saving and loading templates."""
    prompt_manager.register_template(**test_template)
    prompt_manager.track_performance(test_template["name"], score=0.85)
    
    save_path = tmp_path / "templates.json"
    prompt_manager.save_templates(save_path)
    
    new_manager = PromptManager()
    new_manager.load_templates(save_path)
    
    loaded_template = new_manager.get_template(test_template["name"])
    assert loaded_template.template == test_template["template"]
    
    metrics = new_manager.get_performance_metrics(test_template["name"])
    assert len(metrics) == 1
    assert metrics[0]["score"] == 0.85

def test_load_invalid_templates_file(prompt_manager, tmp_path):
    """Test loading from invalid templates file."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("invalid json content")
    
    with pytest.raises(PromptError):
        prompt_manager.load_templates(invalid_file)

def test_template_versioning(prompt_manager, test_template):
    """Test template versioning functionality."""
    # Register initial version
    v1 = prompt_manager.register_template(**test_template)
    assert v1.version == "1.0.0"
    
    # Update to new version
    v2 = prompt_manager.update_template(
        test_template["name"],
        template="Version 2 prompt",
        version="2.0.0"
    )
    assert v2.version == "2.0.0"
    assert v2.template == "Version 2 prompt"
    
    # Original template data should be preserved
    assert v2.parameters == v1.parameters
    assert v2.metadata == v1.metadata
