"""
Proctor: A Python package for text-based prompting techniques.
"""
from typing import List, Optional, Dict, Type # Added Type

# Import base classes
from .base import PromptTechnique, CompositeTechnique

# Import utility classes and exceptions
from .utils import LLMError

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

# Dictionary of all techniques for easy access (using class types)
ALL_TECHNIQUES: Dict[str, Type[PromptTechnique]] = {
    # Zero-Shot
    "emotion_prompting": EmotionPrompting,
    "role_prompting": RolePrompting,
    "self_ask": SelfAsk,
    
    # Few-Shot
    "example_generation": ExampleGeneration,
    "knn": KNN,
    
    # Thought Generation
    "chain_of_thought": ChainOfThought,
    "zero_shot_cot": ZeroShotCoT,
    "few_shot_cot": FewShotCoT,
    
    # Ensembling
    "self_consistency": SelfConsistency,
    
    # Self-Criticism
    "chain_of_verification": ChainOfVerification,
    
    # Decomposition
    "decomp": DECOMP
}

# Cached technique instances
_TECHNIQUE_INSTANCES: Dict[str, PromptTechnique] = {}

def list_techniques(category: Optional[str] = None) -> List[str]:
    """
    List available prompting techniques, optionally filtered by category identifier prefix.
    
    Args:
        category (Optional[str]): Category identifier prefix (e.g., "2.2.1", "2.2.3") to filter by.
                                If None, lists all techniques.
        
    Returns:
        List[str]: List of technique names (keys in ALL_TECHNIQUES).
    """
    if not category:
        return list(ALL_TECHNIQUES.keys())
    
    # Filter by checking if the technique's identifier starts with the category string
    filtered_techniques = []
    
    for name, technique_cls in ALL_TECHNIQUES.items():
        # Get or create cached instance
        if name not in _TECHNIQUE_INSTANCES:
            _TECHNIQUE_INSTANCES[name] = technique_cls()
            
        # Check if identifier matches the category
        if _TECHNIQUE_INSTANCES[name].identifier.startswith(category):
            filtered_techniques.append(name)
            
    return filtered_techniques

def get_technique(name: str) -> Optional[PromptTechnique]:
    """
    Get an initialized technique instance by name.
    
    Args:
        name (str): Name of the technique (key in ALL_TECHNIQUES).
        
    Returns:
        Optional[PromptTechnique]: An initialized instance of the technique, or None if not found.
        
    Note:
        Instances are cached to improve performance when the same technique
        is requested multiple times.
    """
    # Check cache first
    if name in _TECHNIQUE_INSTANCES:
        return _TECHNIQUE_INSTANCES[name]
        
    # Create new instance if not in cache
    technique_cls = ALL_TECHNIQUES.get(name)
    if technique_cls:
        # Initialize and cache the instance
        instance = technique_cls()
        _TECHNIQUE_INSTANCES[name] = instance
        return instance
        
    return None

# Expose key components directly
__all__ = [
    # Base classes
    "PromptTechnique",
    "CompositeTechnique",
    
    # Techniques
    "EmotionPrompting",
    "RolePrompting",
    "SelfAsk",
    "ExampleGeneration",
    "KNN",
    "ChainOfThought",
    "ZeroShotCoT",
    "FewShotCoT",
    "SelfConsistency",
    "ChainOfVerification",
    "DECOMP",
    
    # Utility functions and constants
    "list_techniques",
    "get_technique",
    "ALL_TECHNIQUES",
    "__version__",
    
    # Exceptions
    "LLMError"
] 