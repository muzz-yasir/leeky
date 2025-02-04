"""
Test module for text_loader functionality
"""
import pytest
from pathlib import Path
from leeky.core.text_loader import TextLoader

def test_load_from_file():
    # Get the path to the test file
    file_path = Path("/Users/myasir/leeky/texts/intercept/corbyn_sanders.txt")
    
    # Load the text
    text = TextLoader.load_from_file(str(file_path))
    
    # Basic verification
    assert text.startswith("The British political")
    assert "Bernie Sanders" in text
    
def test_load_from_directory():
    # Get the directory path
    dir_path = Path("/Users/myasir/leeky/texts/intercept")
    
    # Load all texts
    texts = TextLoader.load_from_directory(str(dir_path), "*.txt")
    
    # Verify we got our text
    assert "corbyn_sanders" in texts
    assert texts["corbyn_sanders"].startswith("The British political")