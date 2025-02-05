"""Example script demonstrating the usage of the refactored leeky package."""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.leeky.core import (
    Configuration,
    DataLoader,
    PromptManager,
    TestRunner,
    Evaluator,
    TokenMatchMetric,
    CompletionOverlapMetric,
    ResultsManager,
    LCSMetric,
    FuzzyMatchMetric
)
from src.leeky.core.types import PromptTemplate, TemplateType
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
        prompt_manager.load_templates(
            instruction_path="prompts/instruction_templates.json",
            jailbreak_path="prompts/jailbreak_templates.json"
        )
        
        # Initialize engine based on config
        engine = OpenAIEngine(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4"
        )
        test_runner = TestRunner(
            prompt_manager,
            engine,
            config.get_test_config()
        )

        # Initialize evaluator with both metrics
        evaluator = Evaluator([
            # CompletionOverlapMetric(overlap_type="token"),
            LCSMetric(),
            FuzzyMatchMetric()
        ])
        results_manager = ResultsManager(Path("results"))
        
        # Load test data
        logger.info("Loading text data...")
        texts = data_loader.load_from_json("texts/intercept_articles_small_sentences.json")
        
        # Run tests
        logger.info("Running tests...")
        # Get all instruction and jailbreak templates
        instruction_templates = prompt_manager.get_all_templates(TemplateType.INSTRUCTION)
        jailbreak_templates = prompt_manager.get_all_templates(TemplateType.JAILBREAK)
        
        # Create all combinations of instruction and jailbreak templates
        combined_templates = []
        for instruction in instruction_templates:
            # Add instruction template alone
            combined_templates.append(instruction)
            # Add instruction template with each jailbreak template
            for jailbreak in jailbreak_templates:
                combined_prompt = prompt_manager.combine_templates(
                    instruction_name=instruction.name,
                    jailbreak_name=jailbreak.name
                )
                # Create a new template with the combined prompt
                combined_template = PromptTemplate(
                    template=combined_prompt,
                    name=f"{jailbreak.name}_{instruction.name}",
                    version="1.0.0",
                    parameters=instruction.parameters,
                    metadata={
                        "instruction_template": instruction.name,
                        "instruction_text": instruction.template,
                        "jailbreak_template": jailbreak.name,
                        "jailbreak_text": jailbreak.template,
                        "combined_prompt": combined_prompt
                    },
                    created_at=datetime.now(),
                    template_type=TemplateType.INSTRUCTION
                )
                combined_templates.append(combined_template)
        print(len(combined_templates))
        
        results = await test_runner.run_batch(
            texts,
            combined_templates
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
        
        # Print rankings with detailed metric scores
        print("\nPrompt Template Rankings:")
        for rank in rankings:
            print(f"\n{rank['rank']}. {rank['template_name']}: {rank['overall_score']:.3f}")
            print("Metric scores:")
            for metric_name, score in rank['metric_scores'].items():
                print(f"  - {metric_name}: {score:.3f}")
        
        # Print detailed statistics
        stats = evaluator.get_statistics(evaluation)
        print("\nEvaluation Statistics:")
        for metric_name, metric_stats in stats.items():
            print(f"\n{metric_name}:")
            for stat_name, value in metric_stats.items():
                print(f"  {stat_name}: {value:.3f}")
            
        # Print completion overlap examples
        print("\nCompletion Overlap Examples:")
        for result in results.prompt_results[:3]:  # Show first 3 examples
            if result.completion_comparison:
                print(f"\nTemplate: {result.prompt_template['name']}")
                print(f"Context portion length: {len(result.input_text.context_portion)}")
                print(f"Completion portion length: {len(result.input_text.completion_portion)}")
                print(f"Model output length: {len(result.output_text)}")
                print(f"Overlap metrics:")
                for key, value in result.completion_comparison.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Error running example: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
