# Contributing to Multilingual Alignment Evaluation

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Process](#development-process)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please be respectful and inclusive in all interactions.

## Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/multilingual-alignment-eval.git
   cd multilingual-alignment-eval
   ```

2. **Set Up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .
   ```

3. **Configure Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## How to Contribute

### Reporting Bugs

- Use the issue tracker to report bugs
- Include a clear description and steps to reproduce
- Add relevant logs and error messages
- Specify your environment (OS, Python version, etc.)

### Suggesting Features

- Open an issue to discuss new features
- Explain the use case and benefits
- Consider implementation complexity

### Adding Language Support

To add support for a new language:

1. Update `config/default_config.json`:
   ```json
   "languages": {
     "supported": [..., "new_lang_code"],
     "language_names": {
       "new_lang_code": "Language Name"
     }
   }
   ```

2. Add prompt templates in `scripts/prompt_manager.py`:
   ```python
   templates={
     "en": "English template",
     "new_lang_code": "Translated template"
   }
   ```

3. Add test data in the new language

4. Update documentation

### Adding New Metrics

1. Implement metric calculation in `scripts/evaluate.py`:
   ```python
   def calculate_new_metric(self, df: pd.DataFrame) -> float:
       """Calculate new metric."""
       # Implementation
       return metric_value
   ```

2. Add metric to configuration
3. Update visualization if needed
4. Add tests

## Development Process

### Branch Naming

- Feature: `feature/description`
- Bug fix: `fix/description`
- Documentation: `docs/description`
- Performance: `perf/description`

### Commit Messages

Follow the conventional commits specification:

```
type(scope): subject

body

footer
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `perf`: Performance improvements

Example:
```
feat(evaluate): add cultural sensitivity metric

Add a new metric to evaluate cultural appropriateness of model responses
across different languages and regions.

Closes #123
```

## Style Guidelines

### Python Code Style

- Follow PEP 8
- Use type hints for function parameters and returns
- Maximum line length: 88 characters (Black default)
- Use descriptive variable names

```python
def calculate_metric(
    data: pd.DataFrame,
    threshold: float = 0.8
) -> Dict[str, float]:
    """
    Calculate evaluation metric.
    
    Args:
        data: Evaluation data
        threshold: Score threshold
        
    Returns:
        Dictionary of metric results
    """
    # Implementation
```

### Documentation

- Use Google-style docstrings
- Include examples in docstrings when helpful
- Keep README and other docs up to date

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=scripts --cov-report=html

# Run specific test file
pytest tests/test_evaluate.py

# Run tests in parallel
pytest -n auto
```

### Writing Tests

- Add tests for new features
- Maintain test coverage above 80%
- Use descriptive test names
- Include edge cases

Example test:
```python
def test_calculate_metrics_with_empty_data():
    """Test metric calculation handles empty datasets."""
    evaluator = MultilingualEvaluator()
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError):
        evaluator.calculate_metrics(empty_df)
```

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Include parameter descriptions and return types
- Add usage examples for complex functions

### User Documentation

- Update README.md for user-facing changes
- Add examples to the examples/ directory
- Update API documentation if needed

## Submitting Changes

1. **Create a Pull Request**
   - Push your branch to your fork
   - Open a PR against the main branch
   - Fill out the PR template completely

2. **PR Title Format**
   ```
   type: Brief description
   ```

3. **PR Description Should Include:**
   - What changes were made and why
   - Link to related issues
   - Testing performed
   - Screenshots for UI changes

4. **Code Review Process**
   - All PRs require at least one review
   - Address reviewer feedback
   - Keep PRs focused and reasonably sized

5. **Checklist Before Submitting:**
   - [ ] Tests pass locally
   - [ ] Code follows style guidelines
   - [ ] Documentation is updated
   - [ ] Commit messages are clear
   - [ ] PR description is complete

## Development Tips

### Debugging

```python
# Use logging instead of print
import logging
logger = logging.getLogger(__name__)
logger.debug("Debug information")
```

### Performance

- Profile code for performance bottlenecks
- Use appropriate data structures
- Consider memory usage for large datasets

### Security

- Never commit API keys or secrets
- Validate all user inputs
- Use secure defaults

## Getting Help

- Check existing issues and documentation
- Ask questions in discussions
- Join our community chat (if available)

Thank you for contributing to making AI evaluation more accessible across all languages!