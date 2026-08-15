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

# Example 1: Using Google Gemini 2.5 Flash via OpenRouter
print("🤖 Example 1: Google Gemini 2.5 Flash via OpenRouter")
print("=" * 80)

gemini_config = {
    "model": "google/gemini-2.5-flash",  # Remove openrouter/ prefix - LiteLLM detects it from api_base
    "api_base": "https://openrouter.ai/api/v1",
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


print("🧠 Example 2: Claude 4 Sonnet via OpenRouter")
print("=" * 80)

claude_config = {
    "model": "anthropic/claude-3.5-sonnet",  # Fixed model name
    "api_base": "https://openrouter.ai/api/v1",
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
    "model": "deepseek/deepseek-r1",  # Fixed model name
    "api_base": "https://openrouter.ai/api/v1",
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

# Example 4: Using Llama 3.1 405B via OpenRouter (Most capable open model)
print("🦙 Example 4: Llama 4 Scout via OpenRouter")
print("=" * 80)

llama_config = {
    "model": "meta-llama/llama-3.3-70b-instruct",  # Fixed model name (using available model)
    "api_base": "https://openrouter.ai/api/v1",
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
print(f"Llama 4 Scout Response: {response[:300]}...")

print("\n" + "=" * 80)

# Example 5: Using a smaller, faster model for quick tasks
print("⚡ Example 5: Mistral Small 3.1 24B (Fast & Efficient)")
print("=" * 80)

mistral_config = {
    "model": "mistralai/mistral-small",  # Fixed model name
    "api_base": "https://openrouter.ai/api/v1",
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
print(f"Mistral Small 3.1 24B Response: {response[:300]}...")

print("\n" + "=" * 80)
print("✅ Successfully demonstrated using 5 different OpenRouter models!")
print("💡 Key points:")
print("   - Use 'llm_config=' parameter (not 'config=')")
print("   - DO NOT prefix models with 'openrouter/' - LiteLLM detects it from api_base")
print("   - You can override any model configuration per technique")
print("   - LiteLLM handles the API differences automatically")
print("   - Different models have different strengths and pricing")
