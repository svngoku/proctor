# DSPy Migration Guide for Proctor AI

## Quick Start

### Installation

```bash
# Install with DSPy support
pip install proctor-ai>=0.2.0

# Or install from source
git clone https://github.com/svngoku/proctor.git
cd proctor
pip install -e .
```

### Basic Usage

```python
from dotenv import load_dotenv
from proctor.dspy_lm import configure_dspy_with_litellm
from proctor.dspy_techniques import DSPyChainOfThought

# Load API key
load_dotenv()

# Configure DSPy
configure_dspy_with_litellm()

# Use DSPy technique
cot = DSPyChainOfThought()
result = cot.forward(problem="What is 15% of 200?")

print(result.reasoning)
print(result.answer)
```

---

## Migration Paths

### Option 1: Gradual Migration (Recommended)

Use both classic and DSPy techniques side-by-side:

```python
# Classic techniques still work
from proctor import ChainOfThought

classic_cot = ChainOfThought()
response = classic_cot.execute("problem")  # Returns string

# DSPy techniques available separately
from proctor.dspy_techniques import DSPyChainOfThought

dspy_cot = DSPyChainOfThought()
result = dspy_cot.forward(problem="problem")  # Returns structured output
```

**Benefits:**
- Zero breaking changes
- Test DSPy on specific use cases first
- Gradual learning curve

---

### Option 2: Direct Migration

Replace classic techniques with DSPy versions:

**Before:**
```python
from proctor import ChainOfThought

cot = ChainOfThought()
response = cot.execute("What is 2+2?")
# response is a string
```

**After:**
```python
from proctor.dspy_lm import configure_dspy_with_litellm
from proctor.dspy_techniques import DSPyChainOfThought

configure_dspy_with_litellm()

cot = DSPyChainOfThought()
result = cot.forward(problem="What is 2+2?")
# result.reasoning and result.answer are structured fields

# Or use backward-compatible interface
response = cot.execute("What is 2+2?")
# response is a string (for compatibility)
```

---

## Key Differences

### 1. Initialization

**Classic:**
```python
from proctor import ChainOfThought
cot = ChainOfThought()
```

**DSPy:**
```python
from proctor.dspy_lm import configure_dspy_with_litellm
from proctor.dspy_techniques import DSPyChainOfThought

configure_dspy_with_litellm()  # One-time setup
cot = DSPyChainOfThought()
```

### 2. Execution

**Classic:**
```python
response = cot.execute(input_text)
# Returns: string
```

**DSPy:**
```python
# Structured output (recommended)
result = cot.forward(problem=input_text)
# Returns: dspy.Prediction with typed fields
print(result.reasoning)
print(result.answer)

# OR backward-compatible (returns string)
response = cot.execute(input_text)
```

### 3. Output Format

**Classic:**
```python
response = cot.execute("What is 2+2?")
# response = "Step 1: ... Step 2: ... Answer: 4"
# Need to parse manually
```

**DSPy:**
```python
result = cot.forward(problem="What is 2+2?")
# result.reasoning = "Step 1: ... Step 2: ..."
# result.answer = "4"
# Structured and type-safe
```

---

## Feature Comparison

| Feature | Classic | DSPy |
|---------|---------|------|
| Output Type | String | Structured (Prediction) |
| Type Safety | ❌ | ✅ |
| Auto-Optimization | ❌ | ✅ (BootstrapFewShot) |
| Composability | Limited | ✅ (Full module system) |
| Validation | Manual | ✅ (Metrics) |
| Caching | Basic | ✅ (DSPy built-in) |
| Tracing | Logging only | ✅ (Full traces) |
| Backward Compatible | N/A | ✅ (via execute()) |

---

## Available DSPy Techniques

### Thought Generation
- `DSPyChainOfThought` - Structured CoT with reasoning + answer
- `DSPyZeroShotCoT` - Zero-shot version
- `DSPySelfConsistency` - Multiple paths with voting (real ensemble!)

### Coming Soon
- DSPy versions of all 50+ techniques
- Decomposition techniques
- Self-criticism techniques
- Few-shot techniques with semantic selection

---

## Common Patterns

### Pattern 1: Simple Reasoning

```python
from proctor.dspy_lm import configure_dspy_with_litellm
from proctor.dspy_techniques import DSPyChainOfThought

configure_dspy_with_litellm()

cot = DSPyChainOfThought()
result = cot.forward(problem="Explain photosynthesis")

print("Reasoning:", result.reasoning)
print("Answer:", result.answer)
```

### Pattern 2: Ensemble Reasoning

```python
from proctor.dspy_techniques import DSPySelfConsistency

sc = DSPySelfConsistency(num_paths=5)
result = sc.forward(problem="Complex problem")

print("Consensus:", result.answer)
print("Confidence:", result.confidence)
print("All answers:", result.all_answers)
```

### Pattern 3: Optimization

```python
import dspy
from proctor.dspy_techniques import DSPyChainOfThought

# Training data
trainset = [
    dspy.Example(problem="Q1", answer="A1").with_inputs("problem"),
    dspy.Example(problem="Q2", answer="A2").with_inputs("problem"),
]

# Metric
def accuracy(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# Optimize
teleprompter = dspy.BootstrapFewShot(metric=accuracy)
cot = DSPyChainOfThought()
optimized_cot = teleprompter.compile(cot, trainset=trainset)

# Use optimized version
result = optimized_cot.forward(problem="New question")
```

### Pattern 4: Composition

```python
import dspy
from proctor.dspy_base import DSPyCompositeTechnique
from proctor.dspy_techniques import DSPyChainOfThought

class CustomPipeline(DSPyCompositeTechnique):
    def __init__(self):
        super().__init__(
            name="Custom Pipeline",
            identifier="custom"
        )
        self.step1 = DSPyChainOfThought()
        self.step2 = DSPyChainOfThought()

    def forward(self, problem):
        # First reasoning
        result1 = self.step1(problem=problem)

        # Refine with second pass
        result2 = self.step2(
            problem=f"Given reasoning: {result1.reasoning}, refine the answer"
        )

        return result2

pipeline = CustomPipeline()
result = pipeline.forward(problem="Complex problem")
```

---

## Optimization Guide

### Step 1: Create Training Data

```python
import dspy

trainset = [
    dspy.Example(
        problem="What is 25% of 200?",
        answer="50"
    ).with_inputs("problem"),
    # ... more examples
]
```

### Step 2: Define Metric

```python
def accuracy_metric(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()
```

### Step 3: Optimize

```python
from proctor.dspy_techniques import DSPyChainOfThought

cot = DSPyChainOfThought()

teleprompter = dspy.BootstrapFewShot(
    metric=accuracy_metric,
    max_bootstrapped_demos=3
)

optimized_cot = teleprompter.compile(cot, trainset=trainset)
```

### Step 4: Save & Load

```python
# Save
optimized_cot.save("optimized_cot.json")

# Load
new_cot = DSPyChainOfThought()
new_cot.load("optimized_cot.json")
```

---

## Troubleshooting

### Issue: "DSPy not configured"

```python
# Solution: Configure DSPy before using techniques
from proctor.dspy_lm import configure_dspy_with_litellm
configure_dspy_with_litellm()
```

### Issue: "API key not found"

```python
# Solution: Set environment variable
import os
os.environ["OPENROUTER_API_KEY"] = "your-key"

# Or use .env file
from dotenv import load_dotenv
load_dotenv()
```

### Issue: Rate limits during optimization

```python
# Solution: Use smaller training sets or add delays
teleprompter = dspy.BootstrapFewShot(
    metric=accuracy_metric,
    max_bootstrapped_demos=2,  # Reduce from 3
    max_labeled_demos=2         # Limit examples
)

# Use subset of training data
optimized = teleprompter.compile(cot, trainset=trainset[:5])
```

---

## Best Practices

### 1. Always Configure DSPy First

```python
# At the start of your script
from proctor.dspy_lm import configure_dspy_with_litellm
configure_dspy_with_litellm()
```

### 2. Use Structured Outputs

```python
# Prefer this
result = cot.forward(problem="...")
answer = result.answer

# Over this
response = cot.execute("...")
# Then parsing the string
```

### 3. Optimize on Representative Data

```python
# Good: Real examples from your domain
trainset = [
    dspy.Example(problem="Real Q1", answer="Real A1"),
    dspy.Example(problem="Real Q2", answer="Real A2"),
]

# Bad: Generic or toy examples
trainset = [
    dspy.Example(problem="What is 1+1?", answer="2"),
]
```

### 4. Start Small, Then Scale

```python
# Start with a few examples
trainset_small = trainset[:5]
optimized = teleprompter.compile(cot, trainset=trainset_small)

# Test it
# Then scale up
trainset_large = trainset[:20]
optimized_v2 = teleprompter.compile(cot, trainset=trainset_large)
```

---

## Examples

See the `/examples` directory for complete working examples:

- `examples/dspy_basic_usage.py` - Basic usage patterns
- `examples/dspy_optimization_example.py` - Optimization examples

---

## Resources

- **DSPy Documentation**: https://dspy-docs.vercel.app/
- **Proctor AI + DSPy Plan**: See `DSPY_OPTIMIZATION_PLAN.md`
- **DSPy GitHub**: https://github.com/stanfordnlp/dspy
- **DSPy Paper**: https://arxiv.org/abs/2310.03714

---

## Getting Help

- **Issues**: https://github.com/svngoku/proctor/issues
- **Documentation**: https://github.com/svngoku/proctor
- **Examples**: `examples/dspy_*.py`

---

**Last Updated**: 2025-10-20
**Version**: 0.2.0
