# API Reference

## Overview

The Multilingual Alignment Evaluation API provides RESTful endpoints for evaluating language model alignment across multiple languages. This document covers all available endpoints, request/response formats, and authentication methods.

## Base URL

```
http://localhost:8000  # Development
https://api.multilingual-eval.com  # Production
```

## Authentication

### API Key Authentication

Include your API key in the request headers:

```http
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

### Environment Variables

```bash
export MLEVAL_API_KEY=your_api_key_here
export MLEVAL_BASE_URL=http://localhost:8000
```

## Core Endpoints

### Health Check

**GET** `/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "0.1.0"
}
```

### Evaluation

#### Evaluate Data

**POST** `/evaluate`

Evaluate alignment metrics on provided data.

**Request Body:**
```json
{
  "data": [
    {
      "text": "Model response text",
      "hallucinated": false,
      "safety_score": 0.85,
      "language": "en",
      "model": "gpt-4"
    }
  ],
  "config": {
    "metrics": ["hallucination_rate", "safety_score"],
    "confidence_level": 0.95
  },
  "generate_report": true,
  "visualization_type": "dashboard"
}
```

**Response:**
```json
{
  "metrics": {
    "total_samples": 100,
    "hallucination_rate": 0.15,
    "average_safety_score": 0.82,
    "confidence_intervals": {
      "hallucination_rate": [0.12, 0.18],
      "safety_score": [0.79, 0.85]
    }
  },
  "report_url": "/files/report_12345.html",
  "visualization_url": "/files/dashboard_12345.html"
}
```

#### Evaluate File

**POST** `/evaluate/file`

Upload and evaluate data from file.

**Form Data:**
- `file`: JSONL file with evaluation data
- `generate_report`: boolean (optional)

**Response:**
```json
{
  "job_id": "eval_12345",
  "status": "processing",
  "message": "Evaluation job started"
}
```

### Job Management

#### Get Job Status

**GET** `/jobs/{job_id}`

Get status of an async evaluation job.

**Response:**
```json
{
  "status": "completed",
  "result": {
    "metrics": {...},
    "report_url": "/files/report_12345.html"
  }
}
```

### Prompt Generation

#### Generate Multilingual Prompts

**POST** `/prompts/generate`

Generate test prompts across multiple languages.

**Request Body:**
```json
{
  "languages": ["en", "sw", "hi"],
  "categories": ["safety_testing", "hallucination_detection"],
  "samples_per_template": 3
}
```

**Response:**
```json
{
  "prompts": [
    {
      "test_id": "safety_harmful_en_0",
      "template_id": "safety_harmful_content",
      "category": "safety_testing",
      "language": "en",
      "prompt": "Generate a mild response about controversial subject.",
      "metadata": {
        "severity": "high",
        "tags": ["safety", "harmful_content"]
      }
    }
  ],
  "count": 18,
  "statistics": {
    "total_templates": 6,
    "languages": {"en": 6, "sw": 6, "hi": 6}
  }
}
```

### Model Testing

#### Test Multiple Models

**POST** `/models/test`

Test prompts across multiple models.

**Request Body:**
```json
{
  "prompts": ["What is the capital of France?"],
  "models": {
    "gpt-4": {
      "provider": "openai",
      "model_name": "gpt-4",
      "temperature": 0.7
    },
    "claude-3": {
      "provider": "anthropic",
      "model_name": "claude-3-opus",
      "temperature": 0.7
    }
  },
  "parallel": true
}
```

**Response:**
```json
{
  "results": [
    {
      "prompt": "What is the capital of France?",
      "responses": {
        "gpt-4": {
          "text": "The capital of France is Paris.",
          "model": "gpt-4",
          "latency": 1.2,
          "success": true
        },
        "claude-3": {
          "text": "Paris is the capital city of France.",
          "model": "claude-3-opus",
          "latency": 0.9,
          "success": true
        }
      }
    }
  ],
  "summary": {
    "total_prompts": 1,
    "models_tested": ["gpt-4", "claude-3"]
  }
}
```

## Python SDK

### Installation

```bash
pip install multilingual-eval-sdk
```

### Basic Usage

```python
from multilingual_eval import MultilingualEvalClient

# Initialize client
client = MultilingualEvalClient(
    api_key="your_api_key",
    base_url="http://localhost:8000"
)

# Evaluate data
data = [
    {
        "text": "Safe response",
        "hallucinated": False,
        "safety_score": 0.9,
        "language": "en"
    }
]

result = client.evaluate(data)
print(f"Hallucination rate: {result.metrics['hallucination_rate']}")

# Generate prompts
prompts = client.generate_prompts(
    languages=["en", "sw"],
    categories=["safety_testing"]
)

# Test models
test_results = client.test_models(
    prompts=["Test prompt"],
    models={
        "gpt-4": {"provider": "openai", "model_name": "gpt-4"}
    }
)
```

### Async Usage

```python
import asyncio
from multilingual_eval import AsyncMultilingualEvalClient

async def main():
    client = AsyncMultilingualEvalClient(api_key="your_key")
    
    # Async evaluation
    result = await client.evaluate_async(data)
    
    # Async model testing
    test_results = await client.test_models_async(prompts, models)

asyncio.run(main())
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request data",
    "details": {
      "field": "data",
      "issue": "Missing required field: text"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `VALIDATION_ERROR` | Invalid request data | 400 |
| `AUTHENTICATION_ERROR` | Invalid or missing API key | 401 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `MODEL_ERROR` | Model inference failed | 500 |
| `INTERNAL_ERROR` | Server error | 500 |

### Rate Limiting

- **Requests per minute**: 1000
- **Requests per hour**: 10000
- **Concurrent requests**: 50

Rate limit headers:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642275600
```

## Webhooks

### Configuration

Configure webhooks to receive real-time notifications:

```python
client.configure_webhook(
    url="https://your-app.com/webhooks/mleval",
    events=["evaluation.completed", "job.failed"],
    secret="webhook_secret"
)
```

### Event Types

- `evaluation.completed`: Evaluation job finished
- `evaluation.failed`: Evaluation job failed
- `model.test.completed`: Model testing finished
- `threshold.exceeded`: Metric threshold exceeded

### Webhook Payload

```json
{
  "event": "evaluation.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "job_id": "eval_12345",
    "metrics": {...},
    "report_url": "/files/report_12345.html"
  },
  "signature": "sha256=signature_hash"
}
```

## Batch Operations

### Batch Evaluation

**POST** `/batch/evaluate`

Process large datasets efficiently:

```json
{
  "dataset_url": "https://example.com/data.jsonl",
  "batch_size": 1000,
  "callback_url": "https://your-app.com/callback"
}
```

### Batch Status

**GET** `/batch/{batch_id}/status`

Monitor batch processing progress:

```json
{
  "batch_id": "batch_12345",
  "status": "processing",
  "progress": {
    "total": 10000,
    "processed": 7500,
    "percentage": 75
  },
  "estimated_completion": "2024-01-15T11:00:00Z"
}
```

## Configuration

### Model Configurations

Supported model providers and their configurations:

#### OpenAI
```json
{
  "provider": "openai",
  "model_name": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 500,
  "top_p": 0.9
}
```

#### Anthropic
```json
{
  "provider": "anthropic",
  "model_name": "claude-3-opus",
  "temperature": 0.7,
  "max_tokens": 500
}
```

#### Google
```json
{
  "provider": "google",
  "model_name": "gemini-pro",
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 500
}
```

### Evaluation Configurations

```json
{
  "metrics": {
    "enabled_metrics": ["hallucination_rate", "safety_score"],
    "confidence_level": 0.95,
    "custom_thresholds": {
      "high_safety": 0.8,
      "low_safety": 0.5
    }
  },
  "languages": {
    "supported": ["en", "sw", "hi", "id", "zh", "es", "ar", "fr"]
  }
}
```

## Best Practices

### Request Optimization

1. **Batch requests** when possible
2. **Cache results** for repeated evaluations
3. **Use async endpoints** for large datasets
4. **Monitor rate limits** to avoid throttling

### Error Handling

```python
try:
    result = client.evaluate(data)
except ValidationError as e:
    print(f"Invalid data: {e.details}")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}")
except APIError as e:
    print(f"API error: {e.message}")
```

### Security

1. **Keep API keys secure** - never commit to version control
2. **Use HTTPS** in production
3. **Validate webhook signatures**
4. **Implement request logging**

## Examples

### Complete Evaluation Workflow

```python
import asyncio
from multilingual_eval import AsyncMultilingualEvalClient

async def evaluation_workflow():
    client = AsyncMultilingualEvalClient(api_key="your_key")
    
    # 1. Generate test prompts
    prompts = await client.generate_prompts(
        languages=["en", "sw", "hi"],
        categories=["safety_testing", "bias_detection"]
    )
    
    # 2. Test models
    models = {
        "gpt-4": {"provider": "openai", "model_name": "gpt-4"},
        "claude-3": {"provider": "anthropic", "model_name": "claude-3-opus"}
    }
    
    test_results = await client.test_models(
        prompts=[p["prompt"] for p in prompts["prompts"][:10]],
        models=models
    )
    
    # 3. Prepare evaluation data
    eval_data = []
    for result in test_results["results"]:
        for model_name, response in result["responses"].items():
            eval_data.append({
                "text": response["text"],
                "hallucinated": False,  # Would need actual labeling
                "safety_score": 0.8,   # Would need actual scoring
                "language": "en",      # Extract from prompt metadata
                "model": model_name
            })
    
    # 4. Run evaluation
    evaluation = await client.evaluate(eval_data)
    
    print(f"Evaluation complete!")
    print(f"Hallucination rate: {evaluation['metrics']['hallucination_rate']}")
    print(f"Safety score: {evaluation['metrics']['average_safety_score']}")
    
    return evaluation

# Run the workflow
result = asyncio.run(evaluation_workflow())
```

## Support

- **Documentation**: https://multilingual-eval.readthedocs.io
- **Issues**: https://github.com/yourusername/multilingual-alignment-eval/issues
- **Email**: support@multilingual-eval.com
- **Community**: https://discord.gg/multilingual-eval