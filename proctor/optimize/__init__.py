"""
Automatic Prompt Optimization Module.

This module provides implementations of state-of-the-art automatic prompt
engineering techniques including APE, ProTeGi, and RLPrompt.
"""

from typing import List, Optional, Dict, Any, Type
from abc import ABC, abstractmethod

from ..base import PromptTechnique


class PromptOptimizer(ABC):
    """
    Base class for all prompt optimization algorithms.

    This provides a common interface for different optimization strategies
    allowing users to swap algorithms without code changes.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.optimization_history: List[Dict[str, Any]] = []

    @abstractmethod
    async def optimize(
        self,
        task_description: str,
        seed_prompts: List[str],
        labeled_examples: List[Dict[str, str]],
        technique: Optional[PromptTechnique] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Optimize prompts for a given task.

        Args:
            task_description: Natural language description of the task
            seed_prompts: Initial prompts to start optimization from
            labeled_examples: List of input-output pairs for evaluation
            technique: Optional prompt technique to optimize within
            **kwargs: Algorithm-specific parameters

        Returns:
            Dict containing:
                - 'best_prompt': The optimized prompt string
                - 'score': Performance metric of the best prompt
                - 'metadata': Algorithm-specific metadata
                - 'history': Optimization trajectory
        """
        pass

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Return the history of optimization steps."""
        return self.optimization_history


# Import optimization algorithms (after PromptOptimizer is defined to avoid circular imports)
from .ape import APEOptimizer  # noqa: E402
from .protegi import ProTeGiOptimizer  # noqa: E402
from .rlprompt import RLPromptOptimizer  # noqa: E402

# Export optimizer registry
OPTIMIZERS: Dict[str, Type[PromptOptimizer]] = {
    "ape": APEOptimizer,
    "protegi": ProTeGiOptimizer,
    "rlprompt": RLPromptOptimizer,
}

__all__ = [
    "PromptOptimizer",
    "APEOptimizer",
    "ProTeGiOptimizer",
    "RLPromptOptimizer",
    "OPTIMIZERS",
]
