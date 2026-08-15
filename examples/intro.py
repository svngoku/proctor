from dotenv import load_dotenv
import os
from proctor import (
    CompositeTechnique,
    RolePrompting,
    ChainOfThought,
    ChainOfVerification,
    SelfAsk,
    EmotionPrompting,
    list_techniques,
)
from proctor.models import (
    CLAUDE_SONNET_5,
    DEEPSEEK_R1,
    GEMINI_3_7_FLASH,
    LLAMA_4_MAVERICK,
    OPENROUTER_BASE_URL,
)

load_dotenv()

# Check if API key is available
openrouter_key = os.environ.get("OPENROUTER_API_KEY")

print("🔧 Available Techniques:")
print(list_techniques())
print("\n" + "=" * 80)

# Example problem to solve
problem = "How to build a house for a family of 4?"

if not openrouter_key:
    print(
        "❌ OPENROUTER_API_KEY not set. Please set it in your .env file to test different models."
    )
    exit(1)

# Example 1: Gemini 3.7 Flash via OpenRouter
print("🤖 Example 1: Gemini 3.7 Flash via OpenRouter")
print("=" * 80)

gemini_config = {
    "model": GEMINI_3_7_FLASH,
    "api_base": OPENROUTER_BASE_URL,
    "api_key": openrouter_key,
    "temperature": 0.3,
    "max_tokens": 1500,
}

expert_cot = CompositeTechnique(
    name="Expert Chain-of-Thought",
    identifier="custom-expert-cot",
    techniques=[
        RolePrompting(),
        ChainOfThought(),
        ChainOfVerification(),
    ],
)

response = expert_cot.execute(
    problem,
    llm_config=gemini_config,  # Fixed: was 'config', now 'llm_config'
    role="Expert House Builder and Construction Manager",  # This goes to RolePrompting
)
print(f"Gemini Response: {response[:300]}...")

print("\n" + "=" * 80)


print("🧠 Example 2: Claude Sonnet 5 via OpenRouter")
print("=" * 80)

claude_config = {
    "model": CLAUDE_SONNET_5,
    "api_base": OPENROUTER_BASE_URL,
    "api_key": openrouter_key,
    "temperature": 0.7,
    "max_tokens": 2000,
}

claude_technique = ChainOfThought()
response = claude_technique.execute(problem, llm_config=claude_config)
print(f"Claude Response: {response[:300]}...")

print("\n" + "=" * 80)

# Example 3: Using DeepSeek R1 via OpenRouter (Great for reasoning)
print("🔬 Example 3: DeepSeek R1 via OpenRouter")
print("=" * 80)

deepseek_config = {
    "model": DEEPSEEK_R1,
    "api_base": OPENROUTER_BASE_URL,
    "api_key": openrouter_key,
    "temperature": 0.6,
    "max_tokens": 3000,
}

reasoning_technique = CompositeTechnique(
    name="Deep Reasoning Analysis",
    identifier="deep-reasoning",
    techniques=[
        ChainOfThought(),
        SelfAsk(),
        ChainOfVerification(),
    ],
)

response = reasoning_technique.execute(problem, llm_config=deepseek_config)
print(f"DeepSeek R1 Response: {response[:300]}...")

print("\n" + "=" * 80)

print("🦙 Example 4: Llama 4 Maverick via OpenRouter")
print("=" * 80)

llama_config = {
    "model": LLAMA_4_MAVERICK,
    "api_base": OPENROUTER_BASE_URL,
    "api_key": openrouter_key,
    "temperature": 0.6,
    "max_tokens": 2500,
}

llama_technique = EmotionPrompting()
response = llama_technique.execute(
    problem,
    llm_config=llama_config,
    emotion="thoughtful and methodical",  # This goes to EmotionPrompting
)
print(f"Llama 4 Maverick Response: {response[:300]}...")

print("\n" + "=" * 80)

# Example 5: Using a smaller, faster model for quick tasks
print("⚡ Example 5: Mistral Small 2603 (Fast & Efficient)")
print("=" * 80)

mistral_config = {
    "model": "mistralai/mistral-small-2603",
    "api_base": OPENROUTER_BASE_URL,
    "api_key": openrouter_key,
    "temperature": 0.8,
    "max_tokens": 1000,
}

quick_technique = RolePrompting()
response = quick_technique.execute(
    "Give me 3 quick tips for planning a house construction project.",
    llm_config=mistral_config,
    role="Construction Project Manager",
)
print(f"Mistral Small 2603 Response: {response[:300]}...")

print("\n" + "=" * 80)
print("✅ Successfully demonstrated using 5 different OpenRouter models!")
print("💡 Key points:")
print("   - Use 'llm_config=' parameter (not 'config=')")
print("   - Use bare provider/model ids from proctor.models (no openrouter/ prefix)")
print("   - You can override any model configuration per technique")
print("   - Different models have different strengths and pricing")
