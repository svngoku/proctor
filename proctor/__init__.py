"""
Proctor: A Python package for text-based prompting techniques.
"""

from typing import List, Optional, Dict, Type  # Added Type

# Import base classes
from .base import PromptTechnique, CompositeTechnique

# Import utility classes and exceptions
from .utils import LLMError

# Import Zero-Shot techniques
from .zero_shot.techniques import (
    EmotionPrompting,
    RolePrompting,
    StylePrompting,
    S2A,
    SimToM,
    RaR,
    RF2,
    SelfAsk,
)

# Import Few-Shot techniques
from .few_shot.techniques import (
    ExampleGeneration,
    ExampleOrdering,
    ExemplarSelection,
    SGICL,
    VoteK,
    PromptMining,
    KNN,
)

# Import Thought Generation techniques
from .thought_generation.techniques import (
    ChainOfThought,
    ZeroShotCoT,
    FewShotCoT,
    AnalogicalPrompting,
    StepBackPrompting,
    ThreadOfThought,
    TabCoT,
    ActivePrompt,
    AutoCoT,
    ComplexityBased,
    Contrastive,
    MemoryOfThought,
    UncertaintyRouted,
)

# Import Ensembling techniques
from .ensembling.techniques import (
    SelfConsistency,
    COSP,
    DENSE,
    DiVeRSe,
    MaxMutualInformation,
    MetaCoT,
    MoRE,
    UniversalSelfConsistency,
    USP,
    PromptParaphrasing,
)

# Import Self-Criticism techniques
from .self_criticism.techniques import (
    ChainOfVerification,
    SelfCalibration,
    SelfRefine,
    SelfVerification,
    ReverseCoT,
    CumulativeReasoning,
)

# Import Decomposition techniques
from .decomposition.techniques import (
    DECOMP,
    FaithfulCoT,
    LeastToMost,
    PlanAndSolve,
    ProgramOfThought,
    RecursionOfThought,
    SkeletonOfThought,
    TreeOfThought,
)

# Version
__version__ = "1.0.0"

# Dictionary of all techniques for easy access (using class types)
ALL_TECHNIQUES: Dict[str, Type[PromptTechnique]] = {
    # Zero-Shot
    "emotion_prompting": EmotionPrompting,
    "role_prompting": RolePrompting,
    "style_prompting": StylePrompting,
    "s2a": S2A,
    "simtom": SimToM,
    "rar": RaR,
    "rf2": RF2,
    "self_ask": SelfAsk,
    # Few-Shot
    "example_generation": ExampleGeneration,
    "example_ordering": ExampleOrdering,
    "exemplar_selection": ExemplarSelection,
    "sgicl": SGICL,
    "vote_k": VoteK,
    "prompt_mining": PromptMining,
    "knn": KNN,
    # Thought Generation
    "chain_of_thought": ChainOfThought,
    "zero_shot_cot": ZeroShotCoT,
    "few_shot_cot": FewShotCoT,
    "analogical_prompting": AnalogicalPrompting,
    "step_back_prompting": StepBackPrompting,
    "thread_of_thought": ThreadOfThought,
    "tab_cot": TabCoT,
    "active_prompt": ActivePrompt,
    "auto_cot": AutoCoT,
    "complexity_based": ComplexityBased,
    "contrastive": Contrastive,
    "memory_of_thought": MemoryOfThought,
    "uncertainty_routed": UncertaintyRouted,
    # Ensembling
    "self_consistency": SelfConsistency,
    "cosp": COSP,
    "dense": DENSE,
    "diverse": DiVeRSe,
    "max_mutual_information": MaxMutualInformation,
    "meta_cot": MetaCoT,
    "more": MoRE,
    "universal_self_consistency": UniversalSelfConsistency,
    "usp": USP,
    "prompt_paraphrasing": PromptParaphrasing,
    # Self-Criticism
    "chain_of_verification": ChainOfVerification,
    "self_calibration": SelfCalibration,
    "self_refine": SelfRefine,
    "self_verification": SelfVerification,
    "reverse_cot": ReverseCoT,
    "cumulative_reasoning": CumulativeReasoning,
    # Decomposition
    "decomp": DECOMP,
    "faithful_cot": FaithfulCoT,
    "least_to_most": LeastToMost,
    "plan_and_solve": PlanAndSolve,
    "program_of_thought": ProgramOfThought,
    "recursion_of_thought": RecursionOfThought,
    "skeleton_of_thought": SkeletonOfThought,
    "tree_of_thought": TreeOfThought,
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
    "StylePrompting",
    "S2A",
    "SimToM",
    "RaR",
    "RF2",
    "SelfAsk",
    "ExampleGeneration",
    "ExampleOrdering",
    "ExemplarSelection",
    "SGICL",
    "VoteK",
    "PromptMining",
    "KNN",
    "ChainOfThought",
    "ZeroShotCoT",
    "FewShotCoT",
    "AnalogicalPrompting",
    "StepBackPrompting",
    "ThreadOfThought",
    "TabCoT",
    "ActivePrompt",
    "AutoCoT",
    "ComplexityBased",
    "Contrastive",
    "MemoryOfThought",
    "UncertaintyRouted",
    "SelfConsistency",
    "COSP",
    "DENSE",
    "DiVeRSe",
    "MaxMutualInformation",
    "MetaCoT",
    "MoRE",
    "UniversalSelfConsistency",
    "USP",
    "PromptParaphrasing",
    "ChainOfVerification",
    "SelfCalibration",
    "SelfRefine",
    "SelfVerification",
    "ReverseCoT",
    "CumulativeReasoning",
    "DECOMP",
    "FaithfulCoT",
    "LeastToMost",
    "PlanAndSolve",
    "ProgramOfThought",
    "RecursionOfThought",
    "SkeletonOfThought",
    "TreeOfThought",
    # Utility functions and constants
    "list_techniques",
    "get_technique",
    "ALL_TECHNIQUES",
    "__version__",
    # Exceptions
    "LLMError",
]
