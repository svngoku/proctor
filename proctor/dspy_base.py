"""
DSPy-enhanced base classes for prompt techniques.

This module provides DSPy-powered versions of the base PromptTechnique
and CompositeTechnique classes, enabling structured outputs, automatic
optimization, and better composition.
"""

from typing import Dict, Optional, List
from abc import abstractmethod
import dspy
from .utils import log


class DSPyPromptTechnique(dspy.Module):
    """
    Base class for DSPy-powered prompt techniques.

    This class extends dspy.Module to provide:
    - Structured input/output via Signatures
    - Automatic optimization via Teleprompters
    - Type-safe composition
    - Backward compatibility with string-based interface

    Subclasses should:
    1. Define a Signature class for inputs/outputs
    2. Implement forward() method using DSPy predictors
    3. Optionally implement execute() for backward compatibility

    Example:
        >>> class MyTechniqueSignature(dspy.Signature):
        ...     problem: str = dspy.InputField()
        ...     solution: str = dspy.OutputField()
        ...
        >>> class MyTechnique(DSPyPromptTechnique):
        ...     def __init__(self):
        ...         super().__init__(
        ...             name="My Technique",
        ...             identifier="my-tech"
        ...         )
        ...         self.predictor = dspy.Predict(MyTechniqueSignature)
        ...
        ...     def forward(self, problem):
        ...         return self.predictor(problem=problem)
    """

    def __init__(self, name: str, identifier: str, description: str = ""):
        """
        Initialize a DSPy prompt technique.

        Args:
            name (str): Human-readable name
            identifier (str): Unique identifier
            description (str): Description of the technique
        """
        super().__init__()
        self.name = name
        self.identifier = identifier
        self.description = description

    @abstractmethod
    def forward(self, **kwargs) -> dspy.Prediction:
        """
        Execute the technique with structured inputs/outputs.

        This is the main DSPy interface. Subclasses must implement
        this method using DSPy predictors.

        Args:
            **kwargs: Named inputs matching the Signature

        Returns:
            dspy.Prediction: Structured output with typed fields

        Example:
            >>> result = technique.forward(problem="What is 2+2?")
            >>> print(result.answer)  # Access typed field
        """
        pass

    def execute(
        self,
        input_text: str,
        **kwargs,
    ) -> str:
        """
        Backward-compatible string-based interface.

        This method provides compatibility with the original
        PromptTechnique API. Subclasses can override to customize
        string serialization.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments

        Returns:
            str: String representation of the output

        Note:
            Default implementation calls forward() and converts
            the first output field to a string.
        """
        log.info(
            f"[DSPy] Executing technique: [bold magenta]{self.name}[/] ({self.identifier})"
        )
        log.info(f"[DSPy] Input: [cyan]'{input_text}'[/]")

        try:
            # Call forward with input_text as 'problem' by default
            # Subclasses can override for different field names
            result = self.forward(problem=input_text, **kwargs)

            fields = result.toDict() if hasattr(result, "toDict") else {}
            if fields:
                output = "\n\n".join(f"{k}: {v}" for k, v in fields.items())
                log.info(f"[DSPy] Output:\n[green]{output}[/]")
                return output

            # Fallback to string conversion
            output = str(result)
            log.info(f"[DSPy] Output: [green]{output}[/]")
            return output

        except Exception as e:
            log.exception(f"[DSPy] Error during execution: {e}")
            raise RuntimeError(f"Error during DSPy execution: {str(e)}")

    def __str__(self) -> str:
        return f"{self.name} ({self.identifier}) [DSPy]"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', identifier='{self.identifier}')"


class DSPyCompositeTechnique(DSPyPromptTechnique):
    """
    Composite technique that chains multiple DSPy modules.

    Unlike the string-based CompositeTechnique, this class properly
    composes DSPy modules with type-safe data flow between them.

    Example:
        >>> class ExpertCoT(DSPyCompositeTechnique):
        ...     def __init__(self):
        ...         super().__init__(
        ...             name="Expert CoT",
        ...             identifier="expert-cot"
        ...         )
        ...         self.role = RolePromptingModule()
        ...         self.cot = ChainOfThoughtModule()
        ...
        ...     def forward(self, problem, role="expert"):
        ...         role_context = self.role(problem=problem, role=role)
        ...         return self.cot(
        ...             problem=problem,
        ...             context=role_context.context
        ...         )
    """

    def __init__(
        self,
        name: str,
        identifier: str,
        description: str = "",
        modules: Optional[List[DSPyPromptTechnique]] = None,
    ):
        """
        Initialize a composite technique.

        Args:
            name (str): Name of the composite
            identifier (str): Unique identifier
            description (str): Description
            modules (Optional[List]): List of DSPy modules to compose
        """
        super().__init__(name, identifier, description)
        self.modules = modules or []

        # Register modules for DSPy tracking
        for i, module in enumerate(self.modules):
            setattr(self, f"module_{i}", module)

    def forward(self, **kwargs) -> dspy.Prediction:
        """
        Execute all modules in sequence.

        Default implementation passes output of each module to the next.
        Override for custom composition logic.

        Args:
            **kwargs: Initial inputs

        Returns:
            dspy.Prediction: Final output
        """
        if not self.modules:
            raise ValueError("No modules to execute")

        result = kwargs
        for module in self.modules:
            # Pass previous result as input to next module
            if isinstance(result, dspy.Prediction):
                result = result.toDict()
            result = module.forward(**result)

        return result


class DSPySignatureBuilder:
    """
    Helper class for dynamically building DSPy Signatures.

    This is useful for creating signatures programmatically or
    for techniques that need flexible input/output schemas.

    Example:
        >>> builder = DSPySignatureBuilder("CoT")
        >>> builder.add_input("problem", "The problem to solve")
        >>> builder.add_output("reasoning", "Step-by-step reasoning")
        >>> builder.add_output("answer", "Final answer")
        >>> signature = builder.build()
    """

    def __init__(self, name: str, doc: str = ""):
        """
        Initialize the signature builder.

        Args:
            name (str): Name of the signature class
            doc (str): Docstring for the signature
        """
        self.name = name
        self.doc = doc
        self.fields = {}

    def add_input(self, name: str, desc: str = "", type_=str) -> "DSPySignatureBuilder":
        """
        Add an input field.

        Args:
            name (str): Field name
            desc (str): Field description
            type_: Field type (default: str)

        Returns:
            self: For method chaining
        """
        self.fields[name] = (type_, dspy.InputField(desc=desc))
        return self

    def add_output(
        self, name: str, desc: str = "", type_=str
    ) -> "DSPySignatureBuilder":
        """
        Add an output field.

        Args:
            name (str): Field name
            desc (str): Field description
            type_: Field type (default: str)

        Returns:
            self: For method chaining
        """
        self.fields[name] = (type_, dspy.OutputField(desc=desc))
        return self

    def build(self) -> type:
        """
        Build the signature class.

        Returns:
            type: A new Signature class
        """
        # Create annotations dict
        annotations = {}
        class_dict = {"__doc__": self.doc}

        for field_name, (field_type, field_obj) in self.fields.items():
            annotations[field_name] = field_type
            class_dict[field_name] = field_obj

        class_dict["__annotations__"] = annotations

        # Create the class
        signature_class = type(self.name, (dspy.Signature,), class_dict)

        return signature_class


# Utility function for creating simple signatures
def create_signature(
    name: str, inputs: Dict[str, str], outputs: Dict[str, str], doc: str = ""
) -> type:
    """
    Create a DSPy Signature from simple dictionaries.

    Args:
        name (str): Signature class name
        inputs (Dict[str, str]): Input fields (name -> description)
        outputs (Dict[str, str]): Output fields (name -> description)
        doc (str): Signature docstring

    Returns:
        type: A new Signature class

    Example:
        >>> CoTSignature = create_signature(
        ...     name="CoTSignature",
        ...     inputs={"problem": "The problem to solve"},
        ...     outputs={
        ...         "reasoning": "Step-by-step reasoning",
        ...         "answer": "Final answer"
        ...     },
        ...     doc="Chain of Thought reasoning"
        ... )
    """
    builder = DSPySignatureBuilder(name, doc)

    for input_name, input_desc in inputs.items():
        builder.add_input(input_name, input_desc)

    for output_name, output_desc in outputs.items():
        builder.add_output(output_name, output_desc)

    return builder.build()
