# Contributing to Kramer

Thank you for your interest in contributing to Kramer! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/yourusername/kramer.git
cd kramer
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**

```bash
pip install -e ".[dev]"
```

4. **Set up your API key**

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_code_executor.py

# Run with coverage
pytest --cov=kramer --cov-report=html tests/

# Run only fast tests (skip integration)
pytest tests/ -k "not integration"
```

### Code Quality

```bash
# Format code
black src/ tests/ examples/

# Lint code
ruff check src/ tests/ examples/

# Type checking (if you add mypy)
mypy src/
```

### Before Submitting

1. **Run the test suite**: Ensure all tests pass
2. **Format your code**: Use `black` for consistent formatting
3. **Lint your code**: Fix any issues found by `ruff`
4. **Update tests**: Add tests for new features
5. **Update documentation**: Keep README and docstrings current

## Pull Request Process

1. **Create a feature branch**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
   - Write clear, concise commit messages
   - Keep commits focused and atomic
   - Add tests for new functionality

3. **Test thoroughly**

```bash
pytest tests/
```

4. **Update documentation**
   - Update README if adding features
   - Add docstrings to new functions/classes
   - Update CHANGELOG.md

5. **Submit PR**
   - Provide clear description of changes
   - Reference any related issues
   - Ensure CI tests pass

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use type hints where appropriate
- Write descriptive variable names
- Keep functions focused and small

### Docstrings

Use Google-style docstrings:

```python
def my_function(arg1: str, arg2: int) -> bool:
    """
    Short description of the function.

    Longer description if needed, explaining the purpose,
    behavior, and any important details.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong
    """
    pass
```

### Testing

- Write unit tests for all new functions
- Use descriptive test names: `test_<what>_<condition>_<expected_result>`
- Use fixtures for common setup
- Mock external dependencies (API calls, etc.)

## Project Structure

```
kramer/
├── src/kramer/          # Main source code
│   ├── __init__.py
│   ├── data_analysis_agent.py
│   ├── code_executor.py
│   ├── result_parser.py
│   └── notebook_manager.py
├── tests/               # Test suite
│   ├── conftest.py     # Pytest fixtures
│   └── test_*.py       # Test modules
├── examples/            # Usage examples
├── data/                # Sample datasets
└── outputs/             # Generated outputs
```

## Adding New Features

### New Analysis Capabilities

1. Add method to appropriate class
2. Write comprehensive tests
3. Update documentation
4. Add example usage

### New Data Sources

1. Create new data loader in appropriate module
2. Add tests with sample data
3. Update README with supported formats
4. Add example

## Reporting Issues

### Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
- Relevant logs or error messages

### Feature Requests

Include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (optional)
- Examples of similar features elsewhere (optional)

## Code Review Process

1. Maintainers will review PRs within 1 week
2. Address feedback in new commits
3. Once approved, maintainers will merge
4. Your contribution will be credited in CHANGELOG

## Community Guidelines

- Be respectful and constructive
- Help others learn and grow
- Share knowledge and experience
- Follow the Code of Conduct

## Questions?

- Open an issue for questions
- Join discussions in GitHub Discussions
- Tag maintainers for urgent matters

Thank you for contributing to Kramer! 🎉
