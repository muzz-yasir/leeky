"""
Module for loading and managing text content from files and directories.
"""
from pathlib import Path
from typing import Dict, Optional

class TextLoader:
    @staticmethod
    def load_from_file(filepath: str) -> str:
        """
        Load text content from a single file.
        
        Args:
            filepath: Path to the text file
            
        Returns:
            The content of the file as a string
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_from_directory(directory_path: str, file_pattern: str = "*.txt") -> Dict[str, str]:
        """
        Load multiple texts from a directory.
        
        Args:
            directory_path: Path to directory containing text files
            file_pattern: Pattern to match files (default: "*.txt")
            
        Returns:
            Dictionary mapping filename (without extension) to file content
        """
        texts = {}
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory_path} does not exist")
            
        for file in directory.glob(file_pattern):
            texts[file.stem] = TextLoader.load_from_file(str(file))
        
        if not texts:
            raise ValueError(f"No files matching pattern {file_pattern} found in {directory_path}")
            
        return texts