# Multilingual Alignment Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive toolkit for evaluating alignment and safety of Large Language Models (LLMs) across multiple languages, with special focus on low-resource languages.

## 🌟 Features

- **Multi-language Support**: Evaluate models in 8+ languages including Swahili, Hindi, and Bahasa
- **Comprehensive Metrics**: Hallucination detection, safety scoring, and bias analysis
- **Model Agnostic**: Support for OpenAI, Anthropic, Google, HuggingFace, and local models
- **Advanced Preprocessing**: Data validation, augmentation, and quality assurance
- **Rich Visualizations**: Interactive dashboards and detailed reports
- **Extensible Framework**: Easy to add new languages, metrics, and models
- **CLI Tools**: User-friendly command-line interface for all operations

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

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for local model inference

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multilingual-alignment-eval.git
cd multilingual-alignment-eval

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Environment Setup

Create a `.env` file for API keys:

```bash
# API Keys (add as needed)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
HUGGINGFACE_API_KEY=your_hf_key
```

## 🎯 Quick Start

### 1. Generate Sample Data

```bash
python mleval.py generate -n 100 -o sample_data.jsonl
```

### 2. Evaluate the Data

```bash
python mleval.py evaluate sample_data.jsonl -o results.json -v
```

### 3. Create Visualizations

```bash
python mleval.py visualize -i results.json -t dashboard -o dashboard.html
```

## 📖 Usage

### Command Line Interface

The `mleval` CLI provides access to all major functionalities:

```bash
# Show available commands
python mleval.py --help

# Evaluate a dataset
python mleval.py evaluate data/responses.jsonl \
    --output results.json \
    --report report.html \
    --visualize

# Preprocess and validate data
python mleval.py preprocess raw_data.jsonl \
    --output clean_data.jsonl \
    --augment 0.2 \
    --strict

# Generate multilingual prompts
python mleval.py prompts \
    --languages en sw hi \
    --categories safety_testing \
    --output prompts.json

# Test multiple models
python mleval.py test-models prompts.json \
    --models openai:gpt-4 anthropic:claude-3 \
    --output model_comparison.json \
    --parallel

# Create visualizations
python mleval.py visualize \
    --input results.json \
    --type heatmap \
    --output language_heatmap.png
```

### Python API

```python
from scripts.evaluate import MultilingualEvaluator
from scripts.data_processing import DataValidator, DataPreprocessor
from scripts.prompt_manager import MultilingualPromptManager
from scripts.model_adapters import ModelFactory, ModelConfig

# Initialize evaluator
evaluator = MultilingualEvaluator(config_path='config/default_config.json')

# Load and evaluate data
df = evaluator.load_data('data/responses.jsonl')
metrics = evaluator.calculate_metrics(df)

# Generate report
report = evaluator.generate_report(metrics, output_path='report.html')

# Create prompts
prompt_manager = MultilingualPromptManager()
test_suite = prompt_manager.generate_test_suite(
    languages=['en', 'sw', 'hi'],
    categories=['safety_testing', 'hallucination_detection']
)

# Test models
model = ModelFactory.create('openai', ModelConfig(model_name='gpt-4'))
response = model.generate("What is the capital of Kenya?")
```

## 🧩 Components

### 1. Evaluation Module (`scripts/evaluate.py`)
- Comprehensive metrics calculation
- Statistical analysis with confidence intervals
- Multi-dimensional evaluation (language, model, category)
- Export in multiple formats

### 2. Data Processing (`scripts/data_processing.py`)
- Data validation and quality checks
- Preprocessing pipeline
- Data augmentation for robustness testing
- Automatic error detection and correction

### 3. Prompt Management (`scripts/prompt_manager.py`)
- Multilingual prompt templates
- Dynamic prompt generation
- Variation strategies for robustness
- Category-based organization

### 4. Model Adapters (`scripts/model_adapters.py`)
- Unified interface for different LLM providers
- Async/sync execution modes
- Automatic retry and error handling
- Token counting and optimization

### 5. Visualization (`scripts/visualization.py`)
- Interactive Plotly dashboards
- Statistical plots with Matplotlib/Seaborn
- Multi-format report generation
- Customizable themes

## ⚙️ Configuration

The system uses a comprehensive configuration file (`config/default_config.json`):

```json
{
  "evaluation": {
    "metrics": {
      "enabled_metrics": ["hallucination_rate", "safety_score"],
      "custom_thresholds": {
        "high_safety": 0.8,
        "acceptable_hallucination_rate": 0.1
      }
    },
    "languages": {
      "supported": ["en", "sw", "hi", "id", "zh", "es", "ar", "fr"],
      "low_resource": ["sw", "hi", "id"]
    }
  }
}
```

## 📊 Examples

### Example 1: Evaluating Model Safety Across Languages

```python
# Load multilingual dataset
df = pd.read_json('multilingual_responses.jsonl', lines=True)

# Initialize evaluator with custom config
evaluator = MultilingualEvaluator(config_path='config/safety_focused.json')

# Calculate metrics
metrics = evaluator.calculate_metrics(df)

# Generate comprehensive report
report_gen = ReportGenerator()
report_gen.generate_html_report(metrics, df, 'safety_analysis.html')
```

### Example 2: Red Team Testing

```python
# Generate adversarial prompts
prompt_manager = MultilingualPromptManager()
red_team_suite = prompt_manager.generate_test_suite(
    languages=['en', 'sw'],
    categories=['jailbreak_roleplay', 'injection_code'],
    samples_per_template=5
)

# Test models
evaluator = MultiModelEvaluator({
    'gpt-4': ModelFactory.create('openai', ModelConfig(model_name='gpt-4')),
    'claude': ModelFactory.create('anthropic', ModelConfig(model_name='claude-3'))
})

results = evaluator.evaluate_dataset([p['prompt'] for p in red_team_suite])
```

### Example 3: Custom Visualization

```python
from scripts.visualization import EvaluationVisualizer

# Create custom visualization
visualizer = EvaluationVisualizer(style='academic')

# Generate language comparison heatmap
fig = visualizer.plot_language_heatmap(
    df, 
    metric='safety_score',
    save_path='language_safety_heatmap.pdf'
)

# Create temporal trends
fig = visualizer.plot_temporal_trends(df, save_path='safety_trends.png')
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=scripts --cov-report=html

# Run specific test module
python -m pytest tests/test_all.py::TestMultilingualEvaluator -v
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
flake8 scripts/
black scripts/ --check
```

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- [API Reference](docs/api_reference.md)
- [Architecture Overview](docs/architecture.md)
- [Adding New Languages](docs/adding_languages.md)
- [Custom Metrics Guide](docs/custom_metrics.md)

## 📝 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{multilingual_alignment_eval,
  author = {Stokes, Florence},
  title = {Multilingual Alignment Evaluation: A Toolkit for LLM Safety Assessment},
  year = {2024},
  url = {https://github.com/yourusername/multilingual-alignment-eval}
}
```

## 🏆 Acknowledgments

This project builds upon research in:
- Multilingual NLP and low-resource language processing
- AI safety and alignment
- Large language model evaluation

Special thanks to the open-source community and researchers working on making AI safer and more accessible across all languages.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Author**: Florence Stokes
- **Email**: caveascascavca537@gmail.com
- **Issues**: [GitHub Issues](https://github.com/yourusername/multilingual-alignment-eval/issues)

---

<p align="center">
Made with ❤️ for a safer, more inclusive AI future
</p>
