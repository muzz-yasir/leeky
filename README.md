# Leeky

A Python package for testing and evaluating language model prompts. Leeky provides a modular and extensible framework for experimenting with different prompting techniques, evaluating their effectiveness, and tracking performance over time.

## Features

- Modular architecture for prompt testing and evaluation
- Support for multiple completion engines (OpenAI, etc.)
- Configurable data loading and preprocessing
- Extensible prompt template management
- Async support for efficient batch processing
- Multiple evaluation metrics
- Result storage and analysis
- Performance tracking and visualization
- YAML-based configuration

## Installation

```bash
# Install with poetry
poetry install

# Install additional dependencies
poetry run python -m spacy download en_core_web_lg
```

## Streamlit App Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
```

### Deploying to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and select your forked repository
4. In the deployment settings:
   - Main file path: Enter `app.py` (this is the file in the root directory that contains your Streamlit app)
   - Add your OpenAI API key as a secret:
     - Name: `OPENAI_API_KEY`
     - Value: Your OpenAI API key
5. Click "Deploy"

The app requires the following environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key

## Quick Start

1. Create a configuration file (`config.yaml`):
```yaml
data_loader:
  batch_size: 100
  cache_enabled: true
  cache_dir: ".cache"

test_runner:
  batch_size: 10
  max_retries: 3
  timeout: 30.0
  sampling_strategy: "random"
  sampling_parameters:
    size: 1000

engine:
  engine_type: "openai"
  parameters:
    temperature: 0.7
    max_tokens: 100
    model: "gpt-4"
```

2. Set your OpenAI API key:
```bash
export LEEKY_ENGINE_API_KEY=your-api-key
```

3. Use the package:
```python
async def main():
    # Load configuration
    config = Configuration("config.yaml")
    
    # Initialize components
    data_loader = DataLoader(config.get_data_loader_config())
    prompt_manager = PromptManager()
    test_runner = TestRunner(
        prompt_manager,
        OpenAIEngine(),
        config.get_test_config()
    )
    evaluator = Evaluator([TokenMatchMetric()])
    
    # Load test data
    texts = data_loader.load_from_directory("test_data/")
    
    # Run tests
    results = await test_runner.run_batch(
        texts,
        prompt_manager.get_all_templates()
    )
    
    # Evaluate and rank prompts
    evaluation = evaluator.evaluate_batch(results)
    rankings = evaluator.rank_prompts(evaluation)
```

See `examples/prompt_testing.py` for a complete example.

## Architecture

The package is organized into several core modules:

### Data Loading (`data_loader.py`)
- Handles loading text from files and directories
- Supports batch processing
- Includes caching capabilities
- Handles preprocessing

### Prompt Management (`prompt_manager.py`)
- Manages prompt templates
- Tracks template versions
- Records performance metrics
- Supports template parameters

### Test Runner (`test_runner.py`)
- Executes prompt tests
- Supports async operations
- Handles retries and timeouts
- Configurable sampling strategies

### Evaluator (`evaluator.py`)
- Multiple evaluation metrics
- Prompt effectiveness ranking
- Statistical analysis
- Performance tracking

### Results Management (`results_manager.py`)
- Structured result storage
- Export capabilities (JSON, CSV)
- Report generation
- Basic visualization

### Configuration (`config.py`)
- YAML-based configuration
- Environment variable support
- Parameter validation
- Default configurations

## Adding New Components

### Adding a New Metric
```python
from leeky.core import BaseMetric, MetricResult, PromptResult

class CustomMetric(BaseMetric):
    def evaluate(self, result: PromptResult) -> MetricResult:
        # Implement evaluation logic
        score = calculate_score(result)
        return MetricResult(
            metric=self.metric,
            score=score,
            details={"custom_info": "value"},
            timestamp=datetime.now()
        )

    def evaluate_batch(self, results: Sequence[PromptResult]) -> List[MetricResult]:
        return [self.evaluate(result) for result in results]
```

### Adding a New Engine
```python
from leeky.core import CompletionEngine

class CustomEngine(CompletionEngine):
    async def complete(self, prompt: str, **kwargs) -> str:
        # Implement completion logic
        response = await your_api_call(prompt, **kwargs)
        return response.text
```

### Adding a New Data Source
```python
from leeky.core import DataLoader, TextSource

class CustomDataLoader(DataLoader):
    def load_from_custom_source(self, source_id: str) -> TextSource:
        # Implement custom loading logic
        content = fetch_content(source_id)
        return TextSource(
            content=content,
            source_id=source_id,
            metadata={"source": "custom"},
            timestamp=datetime.now()
        )
```

## Development

### Running Tests
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=leeky

# Run async tests
poetry run pytest --asyncio-mode=auto
```

### Code Style
The project uses black for code formatting:
```bash
poetry run black .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

Apache License 2.0
