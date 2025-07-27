# Multilingual Alignment Evaluation Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive platform for evaluating language model alignment across multiple languages with a focus on safety, bias detection, and cultural appropriateness. The platform supports 8 languages and provides enterprise-grade features including real-time monitoring, A/B testing, and cost optimization.

## 🎯 Objectives

- **Multilingual Safety**: Evaluate alignment performance across diverse languages
- **Bias Detection**: Identify and measure various types of bias in model responses
- **Cultural Appropriateness**: Assess cultural sensitivity and contextual awareness
- **Hallucination Detection**: Reduce false information and improve factual accuracy
- **Comprehensive Analysis**: Compare models, techniques, and approaches systematically

## ✨ Features

### Core Evaluation Capabilities
- **8 Language Support**: English, Swahili, Hindi, Indonesian, Chinese, Spanish, Arabic, French
- **Multiple Metrics**: Safety scores, bias detection, hallucination rates, cultural appropriateness
- **Statistical Analysis**: Confidence intervals, significance testing, effect size calculation
- **Cross-Language Comparison**: Analyze performance disparities across languages

### Enterprise Features
- **Real-time Monitoring**: Live metrics tracking with configurable alerts
- **A/B Testing Framework**: Scientific experiment design and statistical analysis
- **Cost Optimization**: Budget tracking and model recommendation engine
- **Bias Detection**: 8 bias types with cultural sensitivity analysis
- **Model Adapters**: Support for OpenAI, Anthropic, Google, HuggingFace, and local models

### User Interfaces
- **CLI Tool**: Powerful command-line interface for automation
- **Web Dashboard**: Interactive Streamlit-based dashboard
- **REST API**: Full-featured API with async support and job queues

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
- [Components](#components)
- [Configuration](#configuration)
- [Examples](#examples)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## 📦 Installation

### Requirements
- Python 3.8 or higher
- 4GB+ RAM recommended
- Internet connection for API-based models

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/multilingual-alignment-eval.git
cd multilingual-alignment-eval

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Docker Installation

```bash
# Build and run with Docker
docker-compose up --build

# Access web dashboard at http://localhost:8501
# Access API at http://localhost:8000
```

### Environment Setup

```bash
# Copy and configure environment variables
cp .env.example .env

# Edit .env with your API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

## 🚀 Quick Start

### 1. Basic Evaluation

```bash
# Run evaluation on sample data
python scripts/evaluate.py data/sample_responses.jsonl

# Generate comprehensive report
python scripts/evaluate.py data/sample_responses.jsonl --report --output results/
```

### 2. Generate Test Prompts

```bash
# Generate multilingual prompts
mleval prompts generate --languages en,sw,hi --categories safety_testing --output prompts.jsonl

# Test specific bias categories
mleval prompts generate --categories bias_detection --templates 10 --variations 3
```

### 3. Launch Web Dashboard

```bash
# Start interactive dashboard
streamlit run dashboard.py

# Open browser to http://localhost:8501
```

### 4. Start API Server

```bash
# Launch REST API
python api_server.py

# Access API documentation at http://localhost:8000/docs
```

## 📖 Usage

### Command Line Interface

The `mleval` command provides comprehensive functionality:

```bash
# Evaluation Commands
mleval evaluate data.jsonl --metrics all --confidence 0.95
mleval evaluate-batch --input-dir data/ --parallel 4
mleval evaluate-realtime --monitor --alerts

# Data Processing
mleval data validate input.jsonl --schema strict
mleval data preprocess input.jsonl --clean --normalize
mleval data augment input.jsonl --multiply 3 --noise 0.1

# Prompt Management  
mleval prompts generate --help
mleval prompts test --models gpt-4,claude-3 --parallel
mleval prompts validate --templates prompts/ --languages all

# Analysis and Reporting
mleval report generate results.json --format html,pdf
mleval bias analyze data.jsonl --types all --cultural-context
mleval monitor start --config monitoring.json

# A/B Testing
mleval ab create --name "safety-test" --models model1,model2
mleval ab analyze experiment_123 --statistical-test welch
mleval ab report experiment_123 --format detailed

# Model Management
mleval models list --providers all
mleval models test --model gpt-4 --prompts test.jsonl
mleval models compare --models gpt-4,claude-3 --benchmark safety
```

### Python API

```python
from scripts.evaluate import MultilingualEvaluator
from scripts.prompt_manager import MultilingualPromptManager
from scripts.model_adapters import ModelFactory

# Initialize evaluator
evaluator = MultilingualEvaluator(config_path="config/default_config.json")

# Load and evaluate data
import pandas as pd
data = pd.read_json("data/responses.jsonl", lines=True)
metrics = evaluator.calculate_metrics(data)

print(f"Hallucination Rate: {metrics['hallucination_rate']:.3f}")
print(f"Average Safety Score: {metrics['average_safety_score']:.3f}")

# Generate multilingual prompts
prompt_manager = MultilingualPromptManager()
prompts = prompt_manager.generate_test_suite(
    languages=["en", "sw", "hi"],
    categories=["safety_testing", "bias_detection"],
    samples_per_template=5
)

# Test multiple models
factory = ModelFactory()
gpt4 = factory.create_adapter("openai", model_name="gpt-4")
claude = factory.create_adapter("anthropic", model_name="claude-3-opus")

# Run comparative analysis
results = []
for prompt in prompts[:10]:
    gpt4_response = gpt4.generate(prompt["prompt"])
    claude_response = claude.generate(prompt["prompt"])
    
    results.append({
        "prompt": prompt["prompt"],
        "gpt4_response": gpt4_response,
        "claude_response": claude_response,
        "language": prompt["language"]
    })
```

### Advanced Usage

```python
# Real-time monitoring
from scripts.monitoring import RealtimeMonitor

monitor = RealtimeMonitor()
monitor.start_monitoring(
    models=["gpt-4", "claude-3"],
    metrics=["safety_score", "bias_score"],
    thresholds={"safety_score": 0.8, "bias_score": 0.3}
)

# A/B testing
from scripts.ab_testing import ABTestingFramework

ab_framework = ABTestingFramework()
experiment = ab_framework.create_experiment(
    name="safety_comparison",
    models={"control": "gpt-4", "treatment": "claude-3"},
    sample_size=1000,
    significance_level=0.05
)

# Bias detection
from scripts.bias_detection import BiasDetector

bias_detector = BiasDetector()
bias_analysis = bias_detector.analyze_responses(
    responses=data,
    bias_types=["gender", "racial", "cultural"],
    cultural_context=True
)
```

## 🧩 Components

### Core Modules

| Module | Description | Key Features |
|--------|-------------|--------------|
| `evaluate.py` | Main evaluation engine | Metrics calculation, statistical analysis |
| `data_processing.py` | Data validation and preprocessing | Schema validation, cleaning, augmentation |
| `prompt_manager.py` | Multilingual prompt templates | 8 languages, multiple categories |
| `model_adapters.py` | Unified model interface | 5 providers, error handling, cost tracking |
| `bias_detection.py` | Bias and fairness analysis | 8 bias types, cultural sensitivity |
| `monitoring.py` | Real-time performance monitoring | Alerts, thresholds, multi-channel notifications |
| `ab_testing.py` | A/B testing framework | Statistical testing, power analysis |
| `cost_optimization.py` | Cost tracking and optimization | Budget controls, model recommendations |

### User Interfaces

| Interface | Technology | Purpose |
|-----------|------------|---------|
| CLI (`mleval.py`) | Click | Automation, batch processing |
| Dashboard (`dashboard.py`) | Streamlit | Interactive analysis, visualization |
| API (`api_server.py`) | FastAPI | Programmatic access, integration |

### Supported Models

| Provider | Models | Features |
|----------|--------|----------|
| OpenAI | GPT-4, GPT-3.5-turbo | Complete API support |
| Anthropic | Claude-3 (Opus, Sonnet, Haiku) | Safety-focused evaluation |
| Google | Gemini Pro, PaLM | Multilingual capabilities |
| HuggingFace | Open-source models | Local and hosted inference |
| Local | Custom models | Self-hosted deployment |

## ⚙️ Configuration

### Basic Configuration

```json
{
  "evaluation": {
    "languages": {
      "supported": ["en", "sw", "hi", "id", "zh", "es", "ar", "fr"],
      "low_resource": ["sw", "hi", "id"]
    },
    "metrics": {
      "enabled": ["hallucination_rate", "safety_score", "bias_score"],
      "confidence_level": 0.95,
      "min_sample_size": 30
    }
  },
  "models": {
    "default_provider": "openai",
    "timeout": 30,
    "retry_attempts": 3,
    "parallel_requests": 5
  }
}
```

### Advanced Configuration

```json
{
  "monitoring": {
    "enabled": true,
    "check_interval": 300,
    "thresholds": {
      "safety_score": {"min": 0.8, "max": 1.0},
      "bias_score": {"min": 0.0, "max": 0.3}
    },
    "notifications": {
      "email": {"enabled": true, "recipients": ["admin@example.com"]},
      "slack": {"enabled": true, "webhook_url": "https://..."}
    }
  },
  "ab_testing": {
    "default_significance_level": 0.05,
    "min_sample_size": 100,
    "max_experiment_duration": "30d"
  },
  "bias_detection": {
    "enabled_types": ["gender", "racial", "cultural", "socioeconomic"],
    "cultural_sensitivity": "high",
    "custom_patterns": "patterns/custom_bias.json"
  }
}
```

### Environment Variables

```bash
# Model API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
HUGGINGFACE_API_KEY=your_hf_key

# Database Configuration (optional)
DATABASE_URL=postgresql://user:pass@localhost/mleval
REDIS_URL=redis://localhost:6379

# Monitoring Configuration
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
SMTP_SERVER=smtp.gmail.com
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Security
SECRET_KEY=your_secret_key_here
API_KEY_SALT=random_salt_for_key_generation
```

## 📋 Examples

### Example 1: Safety Evaluation Across Languages

```python
# Evaluate safety across multiple languages
import pandas as pd
from scripts.evaluate import MultilingualEvaluator

# Load multilingual responses
data = pd.read_json("data/multilingual_safety_test.jsonl", lines=True)

# Run evaluation
evaluator = MultilingualEvaluator()
results = evaluator.calculate_metrics(data)

# Analyze per-language performance
per_lang = results["per_language_metrics"]
for lang, metrics in per_lang.items():
    print(f"{lang}: Safety={metrics['average_safety_score']:.3f}, "
          f"Hallucination={metrics['hallucination_rate']:.3f}")
```

### Example 2: Bias Detection Pipeline

```python
# Comprehensive bias analysis
from scripts.bias_detection import BiasDetector
from scripts.data_processing import DataProcessor

# Process and analyze data
processor = DataProcessor()
clean_data = processor.preprocess(raw_data)

detector = BiasDetector()
bias_results = detector.analyze_responses(
    clean_data, 
    bias_types=["gender", "racial", "cultural"],
    generate_report=True
)

# Generate detailed report
report = detector.generate_fairness_report(bias_results)
print(f"Report saved to: {report['report_path']}")
```

### Example 3: A/B Testing Workflow

```python
# Set up and run A/B test
from scripts.ab_testing import ABTestingFramework

ab = ABTestingFramework()

# Create experiment
experiment = ab.create_experiment(
    name="prompt_optimization",
    control_group={"model": "gpt-4", "prompt_template": "standard"},
    treatment_group={"model": "gpt-4", "prompt_template": "enhanced"},
    sample_size=500,
    metrics=["safety_score", "helpfulness_score"]
)

# Collect data and analyze
ab.collect_responses(experiment, test_prompts)
results = ab.analyze_experiment(experiment.id)

if results["significant"]:
    print(f"Significant improvement: {results['effect_size']:.3f}")
    print(f"Recommendation: {results['recommendation']}")
```

### Example 4: Custom Metrics

```python
# Create and register custom metric
from scripts.custom_metrics import BaseMetricCalculator, MetricSpecification, MetricType

class DomainSpecificMetric(BaseMetricCalculator):
    def __init__(self):
        spec = MetricSpecification(
            name="domain_accuracy",
            description="Domain-specific accuracy metric",
            metric_type=MetricType.CONTINUOUS,
            range=(0.0, 1.0)
        )
        super().__init__(spec)
    
    def calculate(self, data: str, **kwargs) -> float:
        # Custom calculation logic
        return domain_accuracy_score

# Register and use
from scripts.evaluate import ExtendedMultilingualEvaluator

evaluator = ExtendedMultilingualEvaluator()
evaluator.register_custom_metric(DomainSpecificMetric(), "domain_specific")
```

### Example 5: Real-time Monitoring

```python
# Set up monitoring dashboard
from scripts.monitoring import RealtimeMonitor

monitor = RealtimeMonitor(config_path="config/monitoring.json")

# Start monitoring multiple models
monitor.add_model("gpt-4", thresholds={"safety_score": 0.8})
monitor.add_model("claude-3", thresholds={"safety_score": 0.8})

# Configure alerts
monitor.configure_alerts(
    email={"enabled": True, "recipients": ["team@company.com"]},
    slack={"enabled": True, "channel": "#ml-alerts"}
)

monitor.start_monitoring()
```

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Types of Contributions

- **Bug Reports**: Submit detailed bug reports with reproduction steps
- **Feature Requests**: Propose new features or enhancements
- **Code Contributions**: Submit pull requests for bug fixes or new features
- **Documentation**: Improve documentation, examples, or tutorials
- **Language Support**: Add new language support or improve existing translations
- **Dataset Contributions**: Share evaluation datasets or benchmarks

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/multilingual-alignment-eval.git
cd multilingual-alignment-eval

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Run tests
pytest tests/

# Run code quality checks
black .
flake8 .
mypy scripts/
```

### Contribution Guidelines

1. **Code Style**: Follow PEP 8 and use Black for formatting
2. **Testing**: Add tests for new features and ensure all tests pass
3. **Documentation**: Update documentation for any API changes
4. **Commit Messages**: Use clear, descriptive commit messages
5. **Pull Requests**: Submit focused PRs with clear descriptions

### Adding New Languages

See our [Adding New Languages Guide](docs/adding_languages.md) for detailed instructions on adding support for new languages.

### Creating Custom Metrics

Check out the [Custom Metrics Guide](docs/custom_metrics.md) for information on creating and integrating custom evaluation metrics.

## 📚 Documentation

- **[API Reference](docs/api_reference.md)**: Complete API documentation
- **[Architecture Overview](docs/architecture.md)**: System design and components
- **[Adding Languages](docs/adding_languages.md)**: Guide for language support
- **[Custom Metrics](docs/custom_metrics.md)**: Creating custom evaluation metrics

## 📖 Citation

If you use this platform in your research, please cite:

```bibtex
@software{multilingual_alignment_eval,
  title={Multilingual Alignment Evaluation Platform},
  author={Florence Stokes},
  year={2024},
  url={https://github.com/yourusername/multilingual-alignment-eval},
  license={MIT}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI, Anthropic, Google, and HuggingFace for model APIs
- The open-source community for tools and libraries
- Contributors and researchers who provided feedback and suggestions

## 📞 Support

- **Documentation**: [Read the docs](docs/)
- **Email**: caveascascavca537@gmail.com

---

Made with ❤️ for the multilingual AI safety community.
