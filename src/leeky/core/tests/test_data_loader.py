"""Tests for data loader module."""

import pytest
from pathlib import Path
from datetime import datetime
import json
from leeky.core.data_loader import DataLoader
from leeky.core.types import TextSource, DataLoadError

@pytest.fixture
def data_loader():
    """Create a data loader instance for testing."""
    config = {
        "batch_size": 2,
        "cache_enabled": True,
        "cache_dir": ".test_cache"
    }
    return DataLoader(config)

@pytest.fixture
def test_files(tmp_path):
    """Create test files for loading."""
    # Create test directory structure
    text_dir = tmp_path / "texts"
    text_dir.mkdir()
    
    # Create test files
    file1 = text_dir / "test1.txt"
    file1.write_text("Test content 1")
    
    file2 = text_dir / "test2.txt"
    file2.write_text("Test content 2")
    
    # Create subdirectory with file
    subdir = text_dir / "subdir"
    subdir.mkdir()
    file3 = subdir / "test3.txt"
    file3.write_text("Test content 3")
    
    return text_dir

def test_load_from_file(data_loader, test_files):
    """Test loading text from a single file."""
    file_path = test_files / "test1.txt"
    result = data_loader.load_from_file(file_path)
    
    assert isinstance(result, TextSource)
    assert result.content == "Test content 1"
    assert result.source_id == str(file_path)
    assert "filename" in result.metadata
    assert isinstance(result.timestamp, datetime)

def test_load_from_directory(data_loader, test_files):
    """Test loading text from a directory."""
    results = data_loader.load_from_directory(test_files)
    
    assert len(results) == 3  # Including subdirectory file
    assert all(isinstance(r, TextSource) for r in results)
    assert any(r.content == "Test content 1" for r in results)
    assert any(r.content == "Test content 2" for r in results)
    assert any(r.content == "Test content 3" for r in results)

def test_load_from_directory_no_recursive(data_loader, test_files):
    """Test loading text from a directory without recursion."""
    results = data_loader.load_from_directory(test_files, recursive=False)
    
    assert len(results) == 2  # Excluding subdirectory file
    assert all(isinstance(r, TextSource) for r in results)
    assert any(r.content == "Test content 1" for r in results)
    assert any(r.content == "Test content 2" for r in results)

def test_load_from_json(data_loader, tmp_path):
    """Test loading text from JSON file."""
    json_file = tmp_path / "test.json"
    json_file.write_text("""[
        {"content": "Test JSON 1", "metadata": {"source": "test"}},
        {"content": "Test JSON 2"}
    ]""")
    
    results = data_loader.load_from_json(json_file)
    
    assert len(results) == 2
    assert all(isinstance(r, TextSource) for r in results)
    assert any(r.content == "Test JSON 1" for r in results)
    assert any(r.content == "Test JSON 2" for r in results)

def test_load_batch(data_loader, test_files):
    """Test batch loading of text sources."""
    all_texts = data_loader.load_from_directory(test_files)
    batches = list(data_loader.load_batch(all_texts))
    
    assert len(batches) == 2  # With batch_size=2, should split 3 texts into 2 batches
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1

def test_invalid_file(data_loader, tmp_path):
    """Test handling of invalid file."""
    with pytest.raises(DataLoadError):
        data_loader.load_from_file(tmp_path / "nonexistent.txt")

def test_invalid_json(data_loader, tmp_path):
    """Test handling of invalid JSON file."""
    json_file = tmp_path / "invalid.json"
    json_file.write_text("invalid json content")
    
    with pytest.raises(DataLoadError):
        data_loader.load_from_json(json_file)

def test_caching(data_loader, test_files):
    """Test text source caching."""
    file_path = test_files / "test1.txt"
    
    # Load file first time
    result1 = data_loader.load_from_file(file_path)
    cache_path = data_loader._get_cache_path(str(file_path))
    
    assert cache_path.exists()
    
    # Load from cache
    result2 = data_loader._load_from_cache(str(file_path))
    assert result2 is not None
    assert result2.content == result1.content
    assert result2.source_id == result1.source_id

def test_preprocess_text(data_loader):
    """Test text preprocessing."""
    text = "  Test text with whitespace  \n"
    processed = data_loader.preprocess_text(text)
    assert processed == "Test text with whitespace"

@pytest.fixture(autouse=True)
def cleanup(data_loader):
    """Clean up cache directory after tests."""
    yield
    if data_loader.cache_dir.exists():
        for file in data_loader.cache_dir.iterdir():
            file.unlink()
        data_loader.cache_dir.rmdir()
