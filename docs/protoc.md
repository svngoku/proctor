<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# extract all the information and build it from scratch and use litellm with openrouter as a llm engine and dedent for prompts

I'll create a comprehensive Python package that implements the text-based prompting techniques hierarchy shown in the image, using litellm with OpenRouter as the LLM engine and textwrap.dedent for prompt formatting.

## Project Structure

Let's start with the project directory structure:

```
prompt_techniques/
├── pyproject.toml
├── README.md
├── prompt_techniques/
│   ├── __init__.py
│   ├── base.py
│   ├── config.py
│   ├── utils.py
│   ├── zero_shot/
│   │   ├── __init__.py
│   │   └── techniques.py
│   ├── few_shot/
│   │   ├── __init__.py
│   │   └── techniques.py
│   ├── thought_generation/
│   │   ├── __init__.py
│   │   └── techniques.py
│   ├── ensembling/
│   │   ├── __init__.py
│   │   └── techniques.py
│   ├── self_criticism/
│   │   ├── __init__.py
│   │   └── techniques.py
│   └── decomposition/
│       ├── __init__.py
│       └── techniques.py
└── examples/
    ├── basic_usage.py
    └── advanced_usage.py
```


## Core Implementation

### pyproject.toml

```toml
[build-system]
requires = ["setuptools&gt;=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "prompt_techniques"
version = "0.1.0"
authors = [
  {name = "Your Name", email = "your.email@example.com"},
]
description = "A comprehensive package for text-based prompting techniques"
readme = "README.md"
requires-python = "&gt;=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
dependencies = [
    "litellm&gt;=1.0.0",
    "openai&gt;=1.0.0",
]

[project.urls]
"Homepage" = "https://github.com/yourusername/prompt_techniques"
"Bug Tracker" = "https://github.com/yourusername/prompt_techniques/issues"
```


### prompt_techniques/config.py

```python
"""
Configuration for LLM services using litellm with openrouter.
"""
import os
from typing import Dict, Any

# Default configuration for the LLM
DEFAULT_LLM_CONFIG = {
    "model": "openai/gpt-4o",  # Default model to use
    "api_base": "https://openrouter.ai/api/v1",
    "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
    "max_tokens": 1000,
    "temperature": 0.7,
}

def get_llm_config() -&gt; Dict[str, Any]:
    """
    Get the LLM configuration with environment variables if available.
    Returns:
        Dict[str, Any]: The LLM configuration
    """
    config = DEFAULT_LLM_CONFIG.copy()
    
    # Override with environment variables if present
    if os.environ.get("OPENROUTER_API_KEY"):
        config["api_key"] = os.environ.get("OPENROUTER_API_KEY")
    
    if os.environ.get("OPENROUTER_MODEL"):
        config["model"] = os.environ.get("OPENROUTER_MODEL")
        
    return config
```


### prompt_techniques/utils.py

```python
"""
Utility functions for prompt techniques.
"""
import textwrap
from typing import Dict, Any, Optional, List
import litellm
from .config import get_llm_config

def dedent_prompt(prompt: str) -&gt; str:
    """
    Remove common leading whitespace from a multi-line prompt string.
    
    Args:
        prompt (str): The prompt string to dedent
        
    Returns:
        str: The dedented prompt
    """
    return textwrap.dedent(prompt).strip()

def call_llm(
    prompt: str, 
    system_prompt: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None
) -&gt; str:
    """
    Call the LLM with the given prompt using litellm with openrouter.
    
    Args:
        prompt (str): The user prompt to send
        system_prompt (Optional[str]): Optional system prompt to use
        config_override (Optional[Dict[str, Any]]): Override default config values
        
    Returns:
        str: The LLM response content
    """
    config = get_llm_config()
    
    # Apply config overrides if provided
    if config_override:
        config.update(config_override)
    
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = litellm.completion(
            model=config["model"],
            messages=messages,
            api_base=config["api_base"],
            api_key=config["api_key"],
            max_tokens=config.get("max_tokens", 1000),
            temperature=config.get("temperature", 0.7),
        )
        return response.choices[^0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return f"Error: {str(e)}"
```


### prompt_techniques/base.py

```python
"""
Base classes for prompt techniques.
"""
from typing import Any, Dict, Optional, List, Union
from abc import ABC, abstractmethod
from .utils import dedent_prompt, call_llm

class PromptTechnique(ABC):
    """
    Base class for all prompt techniques.
    """
    
    def __init__(
        self, 
        name: str, 
        identifier: str, 
        description: str = ""
    ):
        """
        Initialize a prompt technique.
        
        Args:
            name (str): Name of the technique
            identifier (str): Technique identifier (e.g., "2.2.2")
            description (str): Description of the technique
        """
        self.name = name
        self.identifier = identifier
        self.description = description
    
    @abstractmethod
    def generate_prompt(self, input_text: str, **kwargs) -&gt; str:
        """
        Generate a prompt using this technique.
        
        Args:
            input_text (str): The input text to process
            **kwargs: Additional arguments for prompt generation
            
        Returns:
            str: The generated prompt
        """
        pass
    
    def execute(
        self, 
        input_text: str, 
        system_prompt: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -&gt; str:
        """
        Execute the technique on input text and return the LLM response.
        
        Args:
            input_text (str): The input text
            system_prompt (Optional[str]): Optional system prompt
            llm_config (Optional[Dict[str, Any]]): LLM configuration overrides
            **kwargs: Additional arguments for prompt generation
            
        Returns:
            str: The LLM response
        """
        prompt = self.generate_prompt(input_text, **kwargs)
        return call_llm(prompt, system_prompt, llm_config)
    
    def __str__(self) -&gt; str:
        return f"{self.name} ({self.identifier})"
    
    def __repr__(self) -&gt; str:
        return f"{self.__class__.__name__}(name='{self.name}', identifier='{self.identifier}')"


class CompositeTechnique(PromptTechnique):
    """
    A technique that combines multiple techniques.
    """
    
    def __init__(
        self, 
        name: str, 
        identifier: str, 
        techniques: List[PromptTechnique],
        description: str = ""
    ):
        """
        Initialize a composite technique.
        
        Args:
            name (str): Name of the technique
            identifier (str): Technique identifier
            techniques (List[PromptTechnique]): List of techniques to compose
            description (str): Description of the technique
        """
        super().__init__(name, identifier, description)
        self.techniques = techniques
    
    def generate_prompt(self, input_text: str, **kwargs) -&gt; str:
        """
        Generate a prompt by applying all techniques in sequence.
        
        Args:
            input_text (str): The input text to process
            **kwargs: Additional arguments for prompt generation
            
        Returns:
            str: The generated prompt
        """
        prompt = input_text
        for technique in self.techniques:
            prompt = technique.generate_prompt(prompt, **kwargs)
        return prompt
```


## Technique Implementations

Here are implementations for a few techniques from each category. In a full implementation, you'd implement all the techniques shown in the image.

### prompt_techniques/zero_shot/techniques.py

```python
"""
Implementation of Zero-Shot prompting techniques.
"""
from ..base import PromptTechnique
from ..utils import dedent_prompt

class EmotionPrompting(PromptTechnique):
    """
    Emotion Prompting incorporates emotional cues to guide the model.
    """
    
    def __init__(self):
        """Initialize Emotion Prompting technique."""
        super().__init__(
            name="Emotion Prompting",
            identifier="2.2.2",
            description="Incorporates emotional cues in prompts to guide responses."
        )
    
    def generate_prompt(self, input_text: str, emotion: str = "excited", **kwargs) -&gt; str:
        """
        Generate an emotion-based prompt.
        
        Args:
            input_text (str): Input text
            emotion (str): Emotion to incorporate (default: "excited")
            **kwargs: Additional arguments
            
        Returns:
            str: Generated prompt with emotional cues
        """
        prompt = dedent_prompt(f"""
        I want you to feel {emotion} about helping me with this task.
        
        Task: {input_text}
        
        Please approach this with {emotion} energy and enthusiasm.
        """)
        return prompt


class RolePrompting(PromptTechnique):
    """
    Role Prompting assigns a specific role to the model.
    """
    
    def __init__(self):
        """Initialize Role Prompting technique."""
        super().__init__(
            name="Role Prompting",
            identifier="2.2.2",
            description="Assigns a specific role to the model to guide its responses."
        )
    
    def generate_prompt(self, input_text: str, role: str = "expert", **kwargs) -&gt; str:
        """
        Generate a role-based prompt.
        
        Args:
            input_text (str): Input text
            role (str): Role to assign to the model
            **kwargs: Additional arguments
            
        Returns:
            str: Generated prompt with role assignment
        """
        prompt = dedent_prompt(f"""
        I want you to act as a {role} in this field. 
        
        {input_text}
        
        Please respond as a {role} would.
        """)
        return prompt


class SelfAsk(PromptTechnique):
    """
    Self-Ask encourages the model to ask and answer its own questions.
    """
    
    def __init__(self):
        """Initialize Self-Ask technique."""
        super().__init__(
            name="Self-Ask",
            identifier="2.2.2",
            description="Prompts the model to ask and answer its own questions."
        )
    
    def generate_prompt(self, input_text: str, **kwargs) -&gt; str:
        """
        Generate a self-ask prompt.
        
        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
            
        Returns:
            str: Generated self-ask prompt
        """
        prompt = dedent_prompt(f"""
        Question: {input_text}
        
        To answer this question, I'll need to ask myself some follow-up questions:
        1. [Ask a relevant question]
        [Answer that question]
        
        2. [Ask another relevant question]
        [Answer that question]
        
        Based on these answers, the final answer is:
        """)
        return prompt
```


### prompt_techniques/few_shot/techniques.py

```python
"""
Implementation of Few-Shot prompting techniques.
"""
from typing import List, Dict
from ..base import PromptTechnique
from ..utils import dedent_prompt

class ExampleGeneration(PromptTechnique):
    """
    Example Generation prompting generates examples for few-shot learning.
    """
    
    def __init__(self):
        """Initialize Example Generation technique."""
        super().__init__(
            name="Example Generation",
            identifier="2.2.1.1",
            description="Generates examples for few-shot learning."
        )
    
    def generate_prompt(self, input_text: str, examples: List[Dict[str, str]] = None, **kwargs) -&gt; str:
        """
        Generate a few-shot prompt with examples.
        
        Args:
            input_text (str): Input text
            examples (List[Dict[str, str]]): List of example dictionaries with 'input' and 'output' keys
            **kwargs: Additional arguments
            
        Returns:
            str: Generated few-shot prompt
        """
        if not examples:
            examples = [
                {"input": "Example input 1", "output": "Example output 1"},
                {"input": "Example input 2", "output": "Example output 2"},
                {"input": "Example input 3", "output": "Example output 3"}
            ]
        
        examples_text = "\n\n".join([
            f"Input: {example['input']}\nOutput: {example['output']}"
            for example in examples
        ])
        
        prompt = dedent_prompt(f"""
        I'll show you some examples of how to solve this type of problem:

        {examples_text}

        Now, please solve the following:
        Input: {input_text}
        Output:
        """)
        return prompt


class KNN(PromptTechnique):
    """
    KNN selects examples based on k-nearest neighbors.
    """
    
    def __init__(self):
        """Initialize KNN technique."""
        super().__init__(
            name="KNN",
            identifier="2.2.1.2",
            description="Selects examples using k-nearest neighbors approach."
        )
    
    def generate_prompt(
        self, 
        input_text: str, 
        examples_pool: List[Dict[str, str]] = None,
        k: int = 3,
        **kwargs
    ) -&gt; str:
        """
        Generate a few-shot prompt with KNN-selected examples.
        
        Args:
            input_text (str): Input text
            examples_pool (List[Dict[str, str]]): Pool of available examples
            k (int): Number of nearest neighbors to select
            **kwargs: Additional arguments
            
        Returns:
            str: Generated prompt with KNN-selected examples
        """
        # In a real implementation, we'd compute embeddings and find nearest neighbors
        # For this simplified version, we'll just select k random examples
        import random
        if not examples_pool:
            selected_examples = []
        else:
            selected_examples = random.sample(examples_pool, min(k, len(examples_pool)))
        
        examples_text = "\n\n".join([
            f"Input: {example['input']}\nOutput: {example['output']}"
            for example in selected_examples
        ])
        
        prompt = dedent_prompt(f"""
        Here are some examples that are most similar to your query:

        {examples_text}

        Now, for your query:
        Input: {input_text}
        Output:
        """)
        return prompt
```


### prompt_techniques/thought_generation/techniques.py

```python
"""
Implementation of Thought Generation prompting techniques.
"""
from typing import List, Dict
from ..base import PromptTechnique
from ..utils import dedent_prompt

class ChainOfThought(PromptTechnique):
    """
    Chain-of-Thought (CoT) encourages step-by-step reasoning.
    """
    
    def __init__(self):
        """Initialize Chain-of-Thought technique."""
        super().__init__(
            name="Chain-of-Thought",
            identifier="2.2.3",
            description="Encourages step-by-step reasoning before giving an answer."
        )
    
    def generate_prompt(self, input_text: str, **kwargs) -&gt; str:
        """
        Generate a Chain-of-Thought prompt.
        
        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
            
        Returns:
            str: Generated Chain-of-Thought prompt
        """
        prompt = dedent_prompt(f"""
        {input_text}
        
        Let's work through this step-by-step:
        1. 
        2. 
        3. 
        
        Therefore, the answer is:
        """)
        return prompt


class ZeroShotCoT(PromptTechnique):
    """
    Zero-Shot Chain-of-Thought adds "Let's think step by step" to the prompt.
    """
    
    def __init__(self):
        """Initialize Zero-Shot Chain-of-Thought technique."""
        super().__init__(
            name="Zero-Shot CoT",
            identifier="2.2.3.1",
            description="Adds 'Let's think step by step' to encourage reasoning."
        )
    
    def generate_prompt(self, input_text: str, **kwargs) -&gt; str:
        """
        Generate a Zero-Shot Chain-of-Thought prompt.
        
        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
            
        Returns:
            str: Generated Zero-Shot CoT prompt
        """
        prompt = dedent_prompt(f"""
        {input_text}
        
        Let's think step by step.
        """)
        return prompt


class FewShotCoT(PromptTechnique):
    """
    Few-Shot Chain-of-Thought provides examples of reasoning.
    """
    
    def __init__(self):
        """Initialize Few-Shot Chain-of-Thought technique."""
        super().__init__(
            name="Few-Shot CoT",
            identifier="2.2.3.2",
            description="Provides examples of step-by-step reasoning."
        )
    
    def generate_prompt(
        self, 
        input_text: str, 
        examples: List[Dict[str, str]] = None,
        **kwargs
    ) -&gt; str:
        """
        Generate a Few-Shot Chain-of-Thought prompt.
        
        Args:
            input_text (str): Input text
            examples (List[Dict[str, str]]): Examples with 'problem', 'reasoning', and 'answer' keys
            **kwargs: Additional arguments
            
        Returns:
            str: Generated Few-Shot CoT prompt
        """
        if not examples:
            examples = [
                {
                    "problem": "If John has 5 apples and gives 2 to Mary, how many does he have left?",
                    "reasoning": "John starts with 5 apples. He gives 2 apples to Mary. So he has 5 - 2 = 3 apples left.",
                    "answer": "3 apples"
                },
                {
                    "problem": "If a train travels 120 km in 2 hours, what is its speed?",
                    "reasoning": "Speed equals distance divided by time. The train covers 120 km in 2 hours. So its speed is 120 km ÷ 2 hours = 60 km/hour.",
                    "answer": "60 km/hour"
                }
            ]
        
        examples_text = "\n\n".join([
            f"Problem: {example['problem']}\n\nReasoning: {example['reasoning']}\n\nAnswer: {example['answer']}"
            for example in examples
        ])
        
        prompt = dedent_prompt(f"""
        Here are some examples of how to solve problems by reasoning step-by-step:

        {examples_text}

        Now, let's solve this problem using the same step-by-step approach:

        Problem: {input_text}

        Reasoning:
        """)
        return prompt
```


### prompt_techniques/ensembling/techniques.py

```python
"""
Implementation of Ensembling prompting techniques.
"""
from typing import List
from ..base import PromptTechnique
from ..utils import dedent_prompt

class SelfConsistency(PromptTechnique):
    """
    Self-Consistency generates multiple reasoning paths and finds consensus.
    """
    
    def __init__(self):
        """Initialize Self-Consistency technique."""
        super().__init__(
            name="Self-Consistency",
            identifier="2.2.5",
            description="Generates multiple reasoning paths and finds consensus."
        )
    
    def generate_prompt(
        self, 
        input_text: str,
        num_paths: int = 3,
        **kwargs
    ) -&gt; str:
        """
        Generate a Self-Consistency prompt.
        
        Args:
            input_text (str): Input text
            num_paths (int): Number of reasoning paths to generate
            **kwargs: Additional arguments
            
        Returns:
            str: Generated Self-Consistency prompt
        """
        paths = "\n\n".join([
            f"Path {i+1}:\n[Reasoning steps for path {i+1}]\nConclusion: [Answer {i+1}]"
            for i in range(num_paths)
        ])
        
        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll solve this problem through multiple independent reasoning paths:
        
        {paths}
        
        Analyzing the consensus across these paths:
        [Count of each distinct answer]
        [Analysis of differences in reasoning]
        
        The most consistent answer is:
        """)
        return prompt
```


### prompt_techniques/self_criticism/techniques.py

```python
"""
Implementation of Self-Criticism prompting techniques.
"""
from ..base import PromptTechnique
from ..utils import dedent_prompt

class ChainOfVerification(PromptTechnique):
    """
    Chain-of-Verification reviews and verifies reasoning.
    """
    
    def __init__(self):
        """Initialize Chain-of-Verification technique."""
        super().__init__(
            name="Chain-of-Verification",
            identifier="2.2.6",
            description="Reviews and verifies each step of reasoning."
        )
    
    def generate_prompt(self, input_text: str, **kwargs) -&gt; str:
        """
        Generate a Chain-of-Verification prompt.
        
        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
            
        Returns:
            str: Generated Chain-of-Verification prompt
        """
        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        Initial solution approach:
        1. [Step 1]
        2. [Step 2]
        3. [Step 3]
        
        Now let me verify each step:
        
        Verification of Step 1:
        - Is my understanding correct? [Check]
        - Is my calculation/reasoning correct? [Check]
        - Correction if needed: [Correction]
        
        Verification of Step 2:
        - Is my understanding correct? [Check]
        - Is my calculation/reasoning correct? [Check]
        - Correction if needed: [Correction]
        
        Verification of Step 3:
        - Is my understanding correct? [Check]
        - Is my calculation/reasoning correct? [Check]
        - Correction if needed: [Correction]
        
        Final verified solution:
        [Corrected solution]
        """)
        return prompt
```


### prompt_techniques/decomposition/techniques.py

```python
"""
Implementation of Decomposition prompting techniques.
"""
from ..base import PromptTechnique
from ..utils import dedent_prompt

class DECOMP(PromptTechnique):
    """
    DECOMP breaks down complex problems into simpler subproblems.
    """
    
    def __init__(self):
        """Initialize DECOMP technique."""
        super().__init__(
            name="DECOMP",
            identifier="2.2.4",
            description="Breaks down complex problems into simpler subproblems."
        )
    
    def generate_prompt(self, input_text: str, **kwargs) -&gt; str:
        """
        Generate a DECOMP prompt.
        
        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
            
        Returns:
            str: Generated DECOMP prompt
        """
        prompt = dedent_prompt(f"""
        Complex Problem: {input_text}
        
        Let me break this down into simpler subproblems:
        
        Subproblem 1: [Description]
        Solution to Subproblem 1:
        [Solution]
        
        Subproblem 2: [Description]
        Solution to Subproblem 2:
        [Solution]
        
        Subproblem 3: [Description]
        Solution to Subproblem 3:
        [Solution]
        
        Now, combining these solutions to solve the original problem:
        [Combined solution]
        
        Therefore, the answer to the original problem is:
        """)
        return prompt
```


### prompt_techniques/__init__.py

```python
"""
Prompt techniques package for various text-based prompting methods.
"""
from typing import List, Optional

# Import base classes
from .base import PromptTechnique, CompositeTechnique

# Import Zero-Shot techniques
from .zero_shot.techniques import (
    EmotionPrompting,
    RolePrompting,
    SelfAsk
)

# Import Few-Shot techniques
from .few_shot.techniques import (
    ExampleGeneration,
    KNN
)

# Import Thought Generation techniques
from .thought_generation.techniques import (
    ChainOfThought,
    ZeroShotCoT,
    FewShotCoT
)

# Import Ensembling techniques
from .ensembling.techniques import (
    SelfConsistency
)

# Import Self-Criticism techniques
from .self_criticism.techniques import (
    ChainOfVerification
)

# Import Decomposition techniques
from .decomposition.techniques import (
    DECOMP
)

# Version
__version__ = "0.1.0"

# Dictionary of all techniques for easy access
ALL_TECHNIQUES = {
    # Zero-Shot
    "emotion_prompting": EmotionPrompting(),
    "role_prompting": RolePrompting(),
    "self_ask": SelfAsk(),
    
    # Few-Shot
    "example_generation": ExampleGeneration(),
    "knn": KNN(),
    
    # Thought Generation
    "chain_of_thought": ChainOfThought(),
    "zero_shot_cot": ZeroShotCoT(),
    "few_shot_cot": FewShotCoT(),
    
    # Ensembling
    "self_consistency": SelfConsistency(),
    
    # Self-Criticism
    "chain_of_verification": ChainOfVerification(),
    
    # Decomposition
    "decomp": DECOMP()
}

def list_techniques(category: str = None) -&gt; List[str]:
    """
    List available prompting techniques, optionally filtered by category.
    
    Args:
        category (str): Category to filter by (e.g., "zero_shot", "few_shot")
        
    Returns:
        List[str]: List of technique names
    """
    if not category:
        return list(ALL_TECHNIQUES.keys())
    
    return [
        name for name, technique in ALL_TECHNIQUES.items()
        if category in name or (
            hasattr(technique, "identifier") and 
            category in technique.identifier
        )
    ]

def get_technique(name: str) -&gt; Optional[PromptTechnique]:
    """
    Get a technique by name.
    
    Args:
        name (str): Name of the technique
        
    Returns:
        Optional[PromptTechnique]: The technique, or None if not found
    """
    return ALL_TECHNIQUES.get(name)
```


## Usage Examples

### examples/basic_usage.py

```python
"""
Basic usage examples for prompt techniques.
"""
import os

from prompt_techniques import (
    EmotionPrompting,
    RolePrompting,
    ChainOfThought,
    ZeroShotCoT,
    DECOMP
)

# Set OpenRouter API key
os.environ["OPENROUTER_API_KEY"] = "your_api_key_here"

def main():
    # Example problem
    problem = "A train travels from city A to city B at 60 km/h. Another train travels from city B to city A at 90 km/h. The cities are 300 km apart. If both trains depart at the same time, how far from city A will they meet?"
    
    # Try different techniques
    techniques = [
        EmotionPrompting(),
        RolePrompting(),
        ChainOfThought(),
        ZeroShotCoT(),
        DECOMP()
    ]
    
    for technique in techniques:
        print(f"\n{'='*50}")
        print(f"Using technique: {technique.name} ({technique.identifier})")
        print(f"{'='*50}")
        
        # Generate the prompt
        prompt = technique.generate_prompt(problem)
        print("\nGenerated Prompt:")
        print(f"{'~'*50}")
        print(prompt)
        print(f"{'~'*50}")
        
        # If you have an API key, you can uncomment to get the actual response
        # response = technique.execute(problem)
        # print("\nLLM Response:")
        # print(f"{'~'*50}")
        # print(response)
        # print(f"{'~'*50}")
        
    print("\nDone!")

if __name__ == "__main__":
    main()
```


### examples/advanced_usage.py

```python
"""
Advanced usage examples for prompt techniques.
"""
import os

from prompt_techniques import (
    list_techniques,
    get_technique,
    CompositeTechnique,
    ZeroShotCoT,
    ChainOfThought
)

# Set OpenRouter API key
os.environ["OPENROUTER_API_KEY"] = "your_api_key_here"

def demonstrate_composing_techniques():
    """
    Demonstrate composing multiple techniques together.
    """
    # Create a composite technique combining ZeroShotCoT and ChainOfThought
    composite = CompositeTechnique(
        name="Zero-Shot Chain",
        identifier="custom",
        techniques=[ZeroShotCoT(), ChainOfThought()],
        description="Applies zero-shot CoT followed by explicit chain of thought structure"
    )
    
    problem = "How might quantum computing affect cryptography in the next decade?"
    
    print(f"\n{'='*50}")
    print(f"Using composite technique: {composite.name}")
    print(f"{'='*50}")
    
    # Generate the prompt
    prompt = composite.generate_prompt(problem)
    print("\nGenerated Prompt:")
    print(f"{'~'*50}")
    print(prompt)
    print(f"{'~'*50}")

def main():
    print("\n1. Composing Techniques")
    demonstrate_composing_techniques()
    
    print("\n\n2. Listing Available Techniques")
    all_techniques = list_techniques()
    print(f"All available techniques ({len(all_techniques)}):")
    print(", ".join(all_techniques))
    
    print("\nDone!")

if __name__ == "__main__":
    main()
```


## Extending the Package

To implement the full hierarchy from the image, you would continue adding all the remaining techniques in their respective category files following the same pattern. For each technique:

1. Create a class inheriting from `PromptTechnique`
2. Implement the `generate_prompt` method with appropriate prompting templates
3. Add the technique to the `ALL_TECHNIQUES` dictionary in `__init__.py`

This implementation provides a comprehensive foundation for a text-based prompting techniques package that:

1. Reflects the hierarchical structure shown in the image
2. Uses litellm with OpenRouter as the LLM engine

<div>⁂</div>

[^1]: https://pplx-res.cloudinary.com/image/upload/v1744760903/user_uploads/YPQIKMGkzBRXVma/image.jpg
