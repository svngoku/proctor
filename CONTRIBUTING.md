# Contributing to Proctor

Thank you for your interest in contributing to Proctor! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in the [Issues](https://github.com/svngoku/proctor/issues).
2. If not, create a new issue using the bug report template.
3. Provide a clear description and steps to reproduce.

### Suggesting Features

1. Check if the feature has already been suggested in the [Issues](https://github.com/svngoku/proctor/issues).
2. If not, create a new issue using the feature request template.
3. Describe the feature and its potential benefits clearly.

### Pull Requests

1. Fork the repository.
2. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and write tests for them.
4. Run the tests and linting:
   ```bash
   uv pip install -e ".[dev]"
   pytest
   ruff check .
   ```
5. Commit your changes following [conventional commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: add new feature"
   ```
6. Push to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Open a pull request against the main branch.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/svngoku/proctor.git
   cd proctor
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```

4. For working with KNN features:
   ```bash
   uv pip install -e ".[knn]"
   ```

5. For all features:
   ```bash
   uv pip install -e ".[all]"
   ```

## Running Tests

```bash
pytest
```

## Coding Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) coding standards
- Use type hints for function parameters and return values
- Document your code with docstrings following the Google style
- Keep functions small and focused on a single responsibility
- Use descriptive variable names

## Documentation

- Update documentation for any feature, behavior change, or bug fix
- Use clear and concise language
- Include examples where appropriate

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update or add tests for your changes
3. Update documentation if needed
4. The PR will be merged once it passes all checks and has been reviewed

## Release Process

1. Bump the version in `proctor/__init__.py` following [semantic versioning](https://semver.org/)
2. Create a pull request with the version bump
3. Once merged, create a new release tag with the version number (e.g., `v0.1.0`)
4. GitHub Actions will automatically build and publish to PyPI

## Thank You!

Your contributions to this project are greatly appreciated!