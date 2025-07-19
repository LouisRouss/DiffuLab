# Contributing to DiffuLab

We welcome contributions to DiffuLab! This guide will help you understand our development workflow, coding standards, and how to submit your contributions.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branching Strategy](#branching-strategy)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [Astral UV](https://docs.astral.sh/uv/) package manager
- Git

### Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/DiffuLab.git
   cd DiffuLab
   ```

3. **Set up the development environment**:
   ```bash
   # Install dependencies including dev tools
   uv sync --all-extras

   # Install the package in editable mode
   uv pip install -e .
   ```

4. **Configure accelerate** (required for training):
   ```bash
   accelerate config
   ```

5. **Set up pre-commit hooks** (recommended):
   ```bash
   uv run pre-commit install
   ```

6. **Add the original repository as upstream**:
   ```bash
   git remote add upstream https://github.com/LouisRouss/DiffuLab.git
   ```

## Branching Strategy

We follow a structured branching model to keep the repository organized and maintain code quality.

### Branch Naming Convention

Use descriptive branch names with the following prefixes:

- **`feature/`** - New features or enhancements
  - Example: `feature/add-edm-sampler`
  - Example: `feature/gradient-checkpointing-dit`

- **`fix/`** - Bug fixes
  - Example: `fix/ddpm-timestep-bug`
  - Example: `fix/memory-leak-training`

- **`refactor/`** - Code refactoring without functional changes
  - Example: `refactor/simplify-dit-architecture`
  - Example: `refactor/optimize-dataloader`

- **`docs/`** - Documentation updates
  - Example: `docs/update-installation-guide`
  - Example: `docs/add-architecture-diagrams`

- **`test/`** - Adding or improving tests
  - Example: `test/add-unit-tests-diffusion`
  - Example: `test/integration-tests-training`

- **`chore/`** - Maintenance tasks (dependencies, CI, etc.)
  - Example: `chore/update-dependencies`
  - Example: `chore/improve-ci-pipeline`

### Workflow

1. **Create a new branch** from `main`:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our [code standards](#code-standards)

3. **Commit your changes** with clear, descriptive messages:
   ```bash
   git add .
   git commit -m "feat: add EDM implementation

   - Implement EDM (Elucidating the Design Space of Diffusion-Based Generative Models)
   - Add configuration options for EDM parameters
   - Include unit tests for EDM functionality
   - Update documentation with usage examples"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on GitHub

## Code Standards

### Code Style

We use automated tools to maintain consistent code style:

- **Formatter**: Ruff Format
- **Linter**: Ruff with isort integration
- **Type Checker**: Pyright
- **Spell Checker**: Typos

### Running Code Quality Checks

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking (if configured)
uv run pyright

# Run all checks
uv run pre-commit run --all-files
```

### Code Organization

- **Source code**: Place all source code in `src/diffulab/`
- **Configuration**: Store config files in `configs/`
- **Examples**: Add example scripts to `examples/`
- **Tests**: Write tests in `tests/` directory
- **Documentation**: Update relevant documentation

### Coding Conventions

1. **Imports**: Use absolute imports and follow isort conventions
2. **Docstrings**: Use Google-style docstrings for all public functions and classes
3. **Type hints**: Include type hints for function signatures
4. **Variable naming**: Use descriptive names following Python conventions
5. **Constants**: Use UPPER_CASE for constants

Example:
```python
import torch
from jaxtyping import Float
from torch import nn


class DiTBlock(nn.Module):
    """Diffusion Transformer block with attention and feedforward layers.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to input dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Implementation here...

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq dim"],
        timestep: Float[torch.Tensor, "batch"] | None = None,
    ) -> Float[torch.Tensor, "batch seq dim"]:
        """Forward pass through the DiT block."""
        # Implementation here...
```

## Testing

### Writing Tests

1. **Unit tests**: Test individual functions and classes
2. **Integration tests**: Test interaction between components
3. **Functional tests**: Test end-to-end functionality

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/diffulab

# Run specific test file
uv run pytest tests/test_diffusion.py

# Run tests matching pattern
uv run pytest -k "test_ddpm"
```

### Test Organization

- Place tests in the `tests/` directory
- Mirror the source structure in test files
- Use descriptive test names: `test_function_name_expected_behavior`
- Include both positive and negative test cases

## Submitting Changes

### Pull Request Process

1. **Ensure your branch is up to date**:
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Run all quality checks**:
   ```bash
   uv run ruff format .
   uv run ruff check .
   uv run pytest
   ```

3. **Write a clear PR description**:
   - Describe what your changes do
   - Reference any related issues
   - Include testing instructions
   - Add screenshots/examples if relevant

4. **Update documentation** if needed

5. **Request review** from maintainers

### PR Title Format

Use conventional commit format:
- `feat: add new feature`
- `fix: resolve bug in component`
- `docs: update documentation`
- `refactor: improve code structure`
- `test: add missing tests`
- `chore: update dependencies`

### Review Process

- All PRs require at least one review from a maintainer
- Address feedback promptly
- Keep PRs focused and reasonably sized
- Be responsive to comments and suggestions

## Reporting Issues

### Bug Reports

When reporting bugs, include:
- Clear description of the issue
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (Python version, OS, etc.)
- Relevant code snippets or logs
- Minimal reproducible example if possible

### Feature Requests

For new features:
- Describe the use case and motivation
- Provide examples of how it would be used
- Consider implementation complexity
- Check if it aligns with project goals

### Using Issue Templates

Use the provided issue templates when available, or follow this structure:

```markdown
## Description
Brief description of the issue/feature

## Steps to Reproduce (for bugs)
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version:
- DiffuLab version:
- OS:
- Additional dependencies:

## Additional Context
Any other relevant information
```

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Collaborate openly and transparently

### Communication

- Use clear, descriptive language
- Be patient with questions and learning
- Provide helpful, actionable feedback
- Acknowledge good contributions

### Getting Help

- Check existing issues and documentation first
- Use descriptive titles for questions
- Provide context and examples
- Be specific about what you've tried

## Development Priorities

Based on the current roadmap, high-priority contributions include:

1. **Core Features**:
   - Different parametrizations for DDPM loss (eps, x0, v-parameterization)
   - Unit tests and CI integration
   - Gradient checkpointing for DiT
   - EDM implementation
   - Sampler abstraction

2. **Infrastructure**:

   - Comprehensive test suite

3. **Advanced Features**:
   - LoRA/DoRA fine-tuning support
   - RL post training

## Resources

- [Project README](README.md)
- [Configuration Examples](configs/)
- [Training Examples](examples/)
- [Issue Tracker](https://github.com/LouisRouss/DiffuLab/issues)

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Reach out to maintainers
- Check existing issues and PRs for similar questions

Thank you for contributing to DiffuLab! 🚀
