# Proctor AI: A Python Library for Prompt Engineering Techniques

<p align="center">
  <img src="assets/proctor.png" alt="Proctor Logo" width="200"/>
</p>

[![PyPI version](https://badge.fury.io/py/proctor-ai.svg)](https://badge.fury.io/py/proctor-ai)
[![CI](https://github.com/svngoku/proctor/actions/workflows/python-package.yml/badge.svg)](https://github.com/svngoku/proctor/actions/workflows/python-package.yml)

`proctor-ai` is a comprehensive Python package designed to implement and explore a variety of text-based prompt engineering techniques. It provides a structured way to apply different prompting strategies to interact with Large Language Models (LLMs), using [LiteLLM](https://github.com/BerriAI/litellm) and [OpenRouter](https://openrouter.ai/) as the default backend.

The library is based on the hierarchical structure of prompting techniques outlined in the initial project documentation (`docs/protoc.md`).

## Features

*   **Hierarchical Technique Implementation:** Organizes prompting techniques into categories:
    *   Zero-Shot (e.g., `EmotionPrompting`, `RolePrompting`, `SelfAsk`)
    *   Few-Shot (e.g., `ExampleGeneration`, `KNN`)
    *   Thought Generation (e.g., `ChainOfThought`, `ZeroShotCoT`, `FewShotCoT`)
    *   Decomposition (e.g., `DECOMP`)
    *   Self-Criticism (e.g., `ChainOfVerification`)
    *   Ensembling (e.g., `SelfConsistency`)
*   **Base Classes:** Provides `PromptTechnique` as an extensible base class for creating custom techniques.
*   **Composability:** Allows combining multiple techniques sequentially using `CompositeTechnique`.
*   **LLM Backend:** Uses [LiteLLM](https://github.com/BerriAI/litellm) to interact with various LLM APIs, configured for [OpenRouter](https://openrouter.ai/) by default.
*   **Configuration:** Easily configure API keys and models via environment variables or a `.env` file.
*   **Utilities:** Includes helper functions like `dedent_prompt`.
*   **Logging:** Integrated logging using `rich` for clear, colorized console output showing inputs, prompts, and responses.
*   **Advanced KNN:** Optional implementation of KNN technique with proper text embeddings and semantic similarity (requires additional dependencies).
*   **Error Handling:** Robust error handling with automatic retries for transient API errors.
*   **Performance Optimization:** Caching mechanisms for technique instances and embeddings to improve performance.

## Installation

1.  **Clone the repository (if developing):**
    ```bash
    # Replace with your actual repository URL
    git clone https://github.com/svngoku/proctor.git # Updated repo URL
    cd proctor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Using venv (requires Python 3)
    python3 -m venv .venv
    source .venv/bin/activate # Use activate.fish for fish shell
    
    # Or using uv
    # uv venv
    # source .venv/bin/activate
    ```

3.  **Install the package:**
    *   **For usage:**
        ```bash
        # Install from PyPI
        pip install proctor-ai
        
        # Or using uv
        uv pip install proctor-ai
        
        # Or install directly from GitHub
        pip install git+https://github.com/svngoku/proctor.git
        ```
    *   **For development (from the cloned repo root):**
        ```bash
        # Use uv for basic installation
        uv pip install -e .
        
        # With advanced KNN features
        uv pip install -e ".[knn]"
        
        # With development tools (pytest, ruff, etc.)
        uv pip install -e ".[dev]"
        
        # With all features and tools
        uv pip install -e ".[all]"
        
        # Or use pip
        pip install -e ".[all]"
        ```
        This installs the package in editable mode with your chosen optional dependencies.

## Configuration

The library requires an OpenRouter API key to function.

1.  **Create a `.env` file** in the root of the `proctor` package directory (i.e., `proctor/.env`).
2.  **Add your API key:**
    ```dotenv
    # proctor/.env
    OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY_HERE"
    
    # Optional: Specify a default model to override the default in config.py
    # OPENROUTER_MODEL="mistralai/mistral-7b-instruct"
    ```
    The library uses `python-dotenv` to automatically load these variables when the example scripts are run or when the library is imported.

## Usage

See the `examples/` directory (`proctor/examples/`) for detailed usage patterns.

**Basic Example:**

```python
import os
from dotenv import load_dotenv
from proctor import get_technique, list_techniques

# Load API key from .env file in the current working directory
# Ensure your .env file is in the directory from where you run the script
load_dotenv()

# List available techniques
print("Available techniques:", list_techniques())

# Get a specific technique instance
technique_name = "zero_shot_cot"
cot_technique = get_technique(technique_name)

if cot_technique:
    problem = "Explain the theory of relativity in simple terms."
    
    # Generate the prompt (useful for inspection)
    prompt = cot_technique.generate_prompt(problem)
    print(f"\n--- Generated {technique_name} Prompt ---")
    print(prompt)
    print("--- End Prompt ---")

    # Execute the technique (calls the LLM via LiteLLM/OpenRouter)
    # Check if API key is present before executing
    if os.environ.get("OPENROUTER_API_KEY") and os.environ.get("OPENROUTER_API_KEY") != "YOUR_API_KEY_HERE":
        print(f"\n--- Executing {technique_name} --- ")
        response = cot_technique.execute(problem)
        print(f"\n--- LLM Response ---")
        print(response)
        print("--- End Response ---")
    else:
        print("\nSkipping LLM execution: OPENROUTER_API_KEY not set or is placeholder in .env file.")
else:
    print(f"Technique '{technique_name}' not found.")

```

**Using Composite Techniques:**

```python
from proctor import CompositeTechnique, RolePrompting, ChainOfThought

# Define a composite technique
expert_cot = CompositeTechnique(
    name="Expert Chain-of-Thought",
    identifier="custom-expert-cot",
    techniques=[
        RolePrompting(),      # First, set the role
        ChainOfThought()    # Then, apply structured CoT
    ]
)

problem = "Plan a three-day trip to Kyoto, Japan, focusing on historical sites."

# Generate the combined prompt
prompt = expert_cot.generate_prompt(problem, role="experienced travel planner")
print(prompt)

# Execute (requires API key)
# response = expert_cot.execute(problem, role="experienced travel planner")
# print(response)
```

## Logging

The library uses Python's `logging` module configured with `rich` to provide colorized output in the console. When techniques are executed, you will see:

*   The technique being executed (Magenta)
*   Input text (Cyan)
*   System prompt, if used (Yellow)
*   Generated prompt (Blue)
*   LLM response (Green)

Set the logging level via environment variable if needed (e.g., `LOG_LEVEL=DEBUG`).

## Quick Start

After installation, here's a simple example to get started:

```python
# Install the package
# pip install proctor-ai

import os
from dotenv import load_dotenv
from proctor import ZeroShotCoT

# Load your API key
load_dotenv()

# Create a technique instance
cot = ZeroShotCoT()

# Use it with any problem
problem = "What are the benefits of renewable energy?"
response = cot.execute(problem)
print(response)
```

## Development

1.  Clone the repository.
2.  Set up a virtual environment (see Installation).
3.  Install in development mode: `make install-dev` or `uv pip install -e ".[dev,all]"`.

### Using the Makefile

The project includes a comprehensive Makefile for development tasks:

```bash
# Install dependencies
make install-dev

# Run linting and formatting
make lint

# Run tests
make test

# Run core tests (excluding optional features)
make test-core

# Run tests with coverage
make test-cov

# Check code style without fixing
make check

# Clean build artifacts
make clean

# Build package
make build

# Deploy to Test PyPI
make deploy-test-permissive

# Deploy to Production PyPI
make deploy-prod-permissive

# See all available commands
make help
```

### Manual Commands

If you prefer running commands directly:

```bash
# Running Tests
pytest tests/ -v

# Linting
ruff check proctor/ tests/ examples/
ruff format proctor/ tests/ examples/

# Building
python -m build
```

## Deployment

The project uses automated CI/CD via GitHub Actions:

- **Pull Requests**: Run tests and linting
- **Push to main/master**: Deploy to Test PyPI automatically
- **Tagged releases**: Deploy to Production PyPI automatically

### Creating a Release

1. Update the version in `pyproject.toml`:
   ```bash
   make version-bump-patch  # or version-bump-minor, version-bump-major
   ```

2. Commit and push the version change:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to X.Y.Z"
   git push
   ```

3. Create and push a tag:
   ```bash
   git tag v0.1.2  # Replace with your version
   git push origin v0.1.2
   ```

4. GitHub Actions will automatically:
   - Run tests
   - Build the package
   - Deploy to PyPI

### Manual Deployment

For manual deployment using the Makefile:

```bash
# Deploy to Test PyPI
make deploy-test-permissive

# Deploy to Production PyPI
make deploy-prod-permissive
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`make test-core`)
5. Run the linter (`make lint`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Clone the repo
git clone https://github.com/svngoku/proctor.git
cd proctor

# Set up development environment
make install-dev

# Run tests to ensure everything works
make test-core

# See all available commands
make help
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.