"""Example script demonstrating the usage of the refactored leeky package."""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from leeky.core import (
    Configuration,
    DataLoader,
    PromptManager,
    TestRunner,
    Evaluator,
    TokenMatchMetric,
    ResultsManager
)
from src.leeky.core.engines.openai_engine import OpenAIEngine
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Run prompt testing example."""
    try:
        # Load configuration
        config = Configuration("config/theintercept.yaml")
        
        # Initialize components
        data_loader = DataLoader(config.get_data_loader_config())
        prompt_manager = PromptManager()
        prompt_manager.load_from_json("prompts/prompt_templates.json")
        
        # Initialize engine based on config
        engine_config = config.get_engine_config()
        engine = OpenAIEngine(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=engine_config["parameters"]["model"]
        )
        test_runner = TestRunner(
            prompt_manager,
            engine,
            config.get_test_config()
        )
        evaluator = Evaluator([TokenMatchMetric()])
        results_manager = ResultsManager(Path("results"))
        
        # Load test data
        logger.info("Loading text data...")
        texts = data_loader.load_from_json("dataset_extraction/intercept/intercept_articles.json")
        
        # Run tests
        logger.info("Running tests...")
        results = await test_runner.run_batch(
            texts,
            prompt_manager.get_all_templates()
        )
        
        # Evaluate results
        logger.info("Evaluating results...")
        evaluation = evaluator.evaluate_batch(results)
        rankings = evaluator.rank_prompts(evaluation)
        
        # Save results
        logger.info("Saving results...")
        results_manager.save_batch_result(results)
        
        # Generate report
        logger.info("Generating report...")
        results_manager.generate_report(
            results,
            evaluation,
            Path("results/report.pdf")
        )
        
        # Print rankings
        print("\nPrompt Template Rankings:")
        for rank in rankings:
            print(f"{rank['rank']}. {rank['template_name']}: {rank['overall_score']:.3f}")
            print(f"   Metric scores: {rank['metric_scores']}")
        
        # Print statistics
        stats = evaluator.get_statistics(evaluation)
        print("\nEvaluation Statistics:")
        for metric_name, metric_stats in stats.items():
            print(f"\n{metric_name}:")
            for stat_name, value in metric_stats.items():
                print(f"  {stat_name}: {value:.3f}")
        
    except Exception as e:
        logger.error(f"Error running example: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
