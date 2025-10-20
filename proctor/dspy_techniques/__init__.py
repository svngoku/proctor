"""
DSPy-powered prompt engineering techniques.

This package contains DSPy-enhanced versions of the prompt engineering
techniques, providing structured outputs, automatic optimization, and
better composition capabilities.

All DSPy techniques inherit from DSPyPromptTechnique and use Signatures
for type-safe input/output handling.
"""

from .thought_generation import (
    DSPyChainOfThought,
    DSPyZeroShotCoT,
    DSPySelfConsistency,
)

__all__ = [
    "DSPyChainOfThought",
    "DSPyZeroShotCoT",
    "DSPySelfConsistency",
]
