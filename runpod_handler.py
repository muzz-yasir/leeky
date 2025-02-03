import runpod
import os
from typing import List, Dict, Any, Optional

from leeky.engines.bloom_engine import BloomEngine
from leeky.engines.openai_engine import OpenAIEngine
from leeky.engines.gptj_engine import GTPJEngine
from leeky.engines.gptneo_engine import GTPNeoEngine
from leeky.engines.t5_engine import T5Engine
from leeky.methods.recital import RecitalTester
from leeky.methods.contextual_recital import ContextualRecitalTester
from leeky.methods.semantic_recital import SemanticRecitalTester, SimilarityType
from leeky.methods.source_veracity import SourceVeracityTester
from leeky.methods.source_recall import SourceRecallTester
from leeky.methods.search import SearchTester

class LeekyTester:
    def __init__(self):
        self.engines = {}
        self.testers = {}
        self.initialize_engines()
        self.initialize_testers()

    def initialize_engines(self):
        """Initialize different engine options"""
        # Initialize all available engines
        engine_params = {"temperature": 0.5, "max_length": 512}
        
        self.engines = {
            "bloom": BloomEngine(parameters=engine_params),
            "gptj": GTPJEngine(parameters=engine_params),
            "gptneo": GTPNeoEngine(parameters=engine_params),
            "t5": T5Engine(parameters=engine_params)
        }

        # Initialize OpenAI engine if API key is available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.engines["openai"] = OpenAIEngine(
                api_key=openai_api_key,
                model="text-davinci-003"
            )

    def initialize_testers(self):
        """Initialize default testers with the default engine (bloom)"""
        default_engine = self.engines["bloom"]
        
        self.testers = {
            "recital": RecitalTester(completion_engine=default_engine),
            "contextual_recital": ContextualRecitalTester(
                completion_engine=default_engine,
                source="default"
            ),
            "semantic_recital": SemanticRecitalTester(
                completion_engine=default_engine,
                similarity_method=SimilarityType.SPACY_SIMILARITY_POINTS,
            ),
            "source_veracity": SourceVeracityTester(completion_engine=default_engine),
            "source_recall": SourceRecallTester(completion_engine=default_engine),
            "search": SearchTester(google_search_method="api")
        }

    def update_engine(self, tester_name: str, engine_name: str):
        """Update the engine for a specific tester"""
        if engine_name not in self.engines:
            raise ValueError(f"Engine {engine_name} not found")
        if tester_name not in self.testers:
            raise ValueError(f"Tester {tester_name} not found")
        
        self.testers[tester_name].completion_engine = self.engines[engine_name]

    def run_tests(
        self,
        text: str,
        num_samples: int = 3,
        source: Optional[str] = None,
        match_list: Optional[List[str]] = None,
        selected_testers: Optional[List[str]] = None,
        engine_config: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Run selected tests with specified configurations
        
        Args:
            text: Input text to analyze
            num_samples: Number of samples to generate
            source: Source context for contextual_recital
            match_list: List of terms to match for source_recall
            selected_testers: List of testers to run (runs all if None)
            engine_config: Dictionary mapping tester names to engine names
        """
        # Update engines if specified
        if engine_config:
            for tester_name, engine_name in engine_config.items():
                self.update_engine(tester_name, engine_name)

        # Update contextual recital source if provided
        if source and "contextual_recital" in self.testers:
            self.testers["contextual_recital"].set_source(source)

        # Determine which testers to run
        testers_to_run = selected_testers if selected_testers else self.testers.keys()
        
        results = {}
        for tester_name in testers_to_run:
            if tester_name not in self.testers:
                continue
                
            tester = self.testers[tester_name]
            
            # Handle source_recall separately due to match_list parameter
            if tester_name == "source_recall" and match_list:
                results[tester_name] = tester.test(
                    text,
                    match_list=match_list,
                    num_samples=num_samples
                )
            else:
                results[tester_name] = tester.test(
                    text,
                    num_samples=num_samples
                )

        # Extract scores
        scores = {
            name: results[name]["score"]
            for name in results
        }

        return {
            "scores": scores,
            "full_results": results
        }

# Initialize the tester as a global variable
leeky_tester = LeekyTester()

def handler(event) -> Dict[str, Any]:
    """
    Handle incoming RunPod requests
    
    Expected input format:
    {
        "input": {
            "text": str,                    # Required: Text to analyze
            "num_samples": int,             # Optional: Number of samples (default: 3)
            "source": str,                  # Optional: Source for contextual_recital
            "match_list": List[str],        # Optional: Terms to match for source_recall
            "selected_testers": List[str],  # Optional: List of testers to run
            "engine_config": Dict[str, str] # Optional: Tester to engine mapping
        }
    }
    """
    try:
        # Extract input parameters
        input_data = event["input"]
        
        # Required parameter
        text = input_data["text"]
        
        # Optional parameters
        num_samples = input_data.get("num_samples", 3)
        source = input_data.get("source")
        match_list = input_data.get("match_list")
        selected_testers = input_data.get("selected_testers")
        engine_config = input_data.get("engine_config")

        # Run the tests
        results = leeky_tester.run_tests(
            text=text,
            num_samples=num_samples,
            source=source,
            match_list=match_list,
            selected_testers=selected_testers,
            engine_config=engine_config
        )
        
        return results
    
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})