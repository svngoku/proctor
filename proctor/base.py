"""
Base classes for prompt techniques.
"""

from typing import Any, Dict, Optional, List
from abc import ABC, abstractmethod
from .utils import call_llm, log, LLMError


class PromptTechnique(ABC):
    """
    Base class for all prompt techniques.
    """

    def __init__(self, name: str, identifier: str, description: str = ""):
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
    def generate_prompt(self, input_text: str, **kwargs) -> str:
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
        max_retries: int = 2,
        **kwargs,
    ) -> str:
        """
        Execute the technique on input text and return the LLM response.

        Args:
            input_text (str): The input text
            system_prompt (Optional[str]): Optional system prompt
            llm_config (Optional[Dict[str, Any]]): LLM configuration overrides
            max_retries (int): Maximum number of retry attempts for LLM calls
            **kwargs: Additional arguments for prompt generation

        Returns:
            str: The LLM response

        Raises:
            ValueError: If input validation fails
            LLMError: If there are persistent issues with the LLM call
        """
        log.info(
            f"Executing technique: [bold magenta]{self.name}[/] ({self.identifier})"
        )
        log.info(f"Input Text: [cyan]'{input_text}'[/]")
        if system_prompt:
            log.info(f"System Prompt: [yellow]'{system_prompt}'[/]")
        if llm_config:
            log.info(f"LLM Config Override: {llm_config}")
        if kwargs:
            log.info(f"Additional Args: {kwargs}")

        # Generate prompt (may raise ValueError)
        prompt = self.generate_prompt(input_text, **kwargs)
        log.info(f"Generated Prompt:\n[blue]--- START ---\n{prompt}\n--- END ---[/]")

        try:
            # Call LLM with retry handling
            response = call_llm(prompt, system_prompt, llm_config, max_retries)
            log.info(f"LLM Response:\n[green]--- START ---\n{response}\n--- END ---[/]")
            return response

        except LLMError as e:
            log.error(f"[bold red]LLM Error: {str(e)}[/]")
            # Re-raise the exception for higher-level handling
            raise

        except Exception as e:
            log.exception(f"Unexpected error during execution: {e}")
            raise RuntimeError(f"Unexpected error during execution: {str(e)}")

    def __str__(self) -> str:
        return f"{self.name} ({self.identifier})"

    def __repr__(self) -> str:
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
        description: str = "",
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

    def generate_prompt(self, input_text: str, **kwargs) -> str:
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
            # Assuming generate_prompt returns the modified prompt/text for the next step
            prompt = technique.generate_prompt(prompt, **kwargs)
        return prompt
