# Proctor: A Python Library for Prompt Engineering Techniques

<p align="center">
  <img src="assets/proctor.png" alt="Proctor Logo" width="200"/>
</p>

[![PyPI version](https://badge.fury.io/py/proctor.svg)](https://badge.fury.io/py/proctor) <!-- Placeholder: Replace once published -->
[![CI](https://github.com/svngoku/proctor/actions/workflows/python-package.yml/badge.svg)](https://github.com/svngoku/proctor/actions/workflows/python-package.yml) <!-- Updated username/repo -->

`proctor` is a comprehensive Python package designed to implement and explore a variety of text-based prompt engineering techniques. It provides a structured way to apply different prompting strategies to interact with Large Language Models (LLMs), using [LiteLLM](https://github.com/BerriAI/litellm) and [OpenRouter](https://openrouter.ai/) as the default backend.

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
        # Install from PyPI (once published)
        # uv pip install proctor 
        
        # Or install directly from GitHub (replace with your repo URL)
        # uv pip install git+https://github.com/svngoku/proctor.git
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
        # pip install -e ".[all]"
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

## Development

1.  Clone the repository.
2.  Set up a virtual environment (see Installation).
3.  Install in editable mode: `uv pip install -e .`.
4.  Install development dependencies (if any are added later, e.g., for testing):
    ```bash
    # Example: uv pip install -e ".[dev]"
    ```

### Running Tests (Placeholder)

```bash
# pytest
```

### Linting (Placeholder)

```bash
# ruff check .
# ruff format .
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if one is added, or refer to `pyproject.toml`).