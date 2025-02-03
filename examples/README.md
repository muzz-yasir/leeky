# Leeky API Examples

This directory contains example scripts demonstrating how to use the Leeky API endpoint.

## Files

- `endpoint_examples.py`: A comprehensive example showing different ways to use the Leeky API endpoint

## Usage

### Prerequisites

1. Install required packages:
```bash
pip install requests rich
```

2. Set up your RunPod endpoint:
- Deploy the Leeky container to RunPod serverless
- Get your RunPod API key
- Note your endpoint URL

### Running the Examples

1. Update the endpoint configuration in `endpoint_examples.py`:
```python
endpoint = LeekyEndpoint(
    endpoint_url="https://api.runpod.ai/v2/{your-endpoint-id}/run",
    api_key="your-runpod-api-key"
)
```

2. Run the example script:
```bash
python endpoint_examples.py
```

## Available Methods

### 1. Basic Analysis
Simple text analysis with default settings:
```python
result = endpoint.basic_analysis(
    "Your text here",
    num_samples=3
)
```

### 2. Advanced Analysis
Custom analysis with specific configurations:
```python
result = endpoint.advanced_analysis(
    text="Your text here",
    source="Source Context",
    match_list=["term1", "term2"],
    selected_testers=["recital", "source_recall"],
    engine_config={
        "recital": "openai",
        "source_recall": "bloom"
    }
)
```

### 3. Text Comparison
Compare two texts for veracity and recall:
```python
results = endpoint.compare_texts(
    original_text="Original text here",
    modified_text="Modified text here",
    source="Source Context",
    match_list=["term1", "term2"]
)
```

## Response Format

The API responses follow this structure:
```python
{
    "scores": {
        "recital": float,
        "contextual_recital": float,
        "semantic_recital": float,
        "source_veracity": float,
        "source_recall": float,
        "search": float
    },
    "full_results": {
        "test_name": {
            "score": float,
            "samples": [str, ...],
            "details": dict
        },
        ...
    }
}
```

## Available Testers

- `recital`: Basic recital testing
- `contextual_recital`: Recital testing with context
- `semantic_recital`: Semantic similarity testing
- `source_veracity`: Source truthfulness testing
- `source_recall`: Term recall testing
- `search`: Search-based testing

## Available Engines

- `bloom`: BLOOM language model
- `openai`: OpenAI's GPT models (requires API key)
- `gptj`: GPT-J model
- `gptneo`: GPT-Neo model
- `t5`: T5 model

## Error Handling

The example code includes comprehensive error handling for:
- Network timeouts
- HTTP errors
- Invalid responses
- Authentication issues
- General exceptions

All methods return `None` on error, with error messages printed using rich formatting.