"""
Advanced usage examples for the proctor package.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file in the current directory (proctor/)
load_dotenv()

# Assuming the package is installed or PYTHONPATH is set correctly
from proctor import (
    list_techniques,
    get_technique,
    CompositeTechnique,
    ZeroShotCoT,
    ChainOfThought,
    RolePrompting # Added for composite example
)

# --- IMPORTANT --- 
# Replace with your actual OpenRouter API key or set the environment variable
# os.environ["OPENROUTER_API_KEY"] = "YOUR_OPENROUTER_API_KEY"
# --- IMPORTANT --- 

# It's recommended to use the .env file (created in the same directory)
# instead of setting the variable directly in the code.

# Check if the API key is set, otherwise skip execution
API_KEY_SET = bool(os.environ.get("OPENROUTER_API_KEY"))

def demonstrate_composing_techniques():
    """
    Demonstrate composing multiple techniques together.
    """
    print(f"\n{'='*50}")
    print("Demonstrating Composite Technique")
    print(f"{'='*50}")

    # Create a composite technique combining RolePrompting and ZeroShotCoT
    # Note: The sequence matters. RolePrompting sets the context first.
    composite_technique = CompositeTechnique(
        name="Expert Zero-Shot CoT",
        identifier="custom-role-zeroshotcot",
        techniques=[
            RolePrompting(), # First, set the role
            ZeroShotCoT()    # Then, apply ZeroShotCoT to the role-prompted input
        ],
        description="Acts as an expert and then thinks step-by-step."
    )
    
    problem = "How might quantum computing affect cryptography in the next decade? Consider both potential vulnerabilities and new cryptographic methods."
    
    print(f"Using composite technique: {composite_technique.name} ({composite_technique.identifier})")
    
    # Generate the prompt using the composite technique
    # Pass arguments needed by constituent techniques (e.g., role for RolePrompting)
    prompt = composite_technique.generate_prompt(problem, role="leading cryptographer")
    print("\nGenerated Composite Prompt:")
    print(f"{'~'*50}")
    print(prompt)
    print(f"{'~'*50}")

    # Execute the composite technique (if API key is set)
    if API_KEY_SET:
        print("\nCalling LLM with composite prompt...")
        response = composite_technique.execute(problem, role="leading cryptographer")
        print("\nLLM Response (Composite Technique):")
        print(f"{'~'*50}")
        print(response)
        print(f"{'~'*50}")
    else:
        print("\nSkipping LLM execution: OPENROUTER_API_KEY not set.")

def demonstrate_listing_and_getting_techniques():
    """
    Demonstrate listing available techniques and getting one by name.
    """
    print(f"\n\n{'='*50}")
    print("Listing and Getting Techniques")
    print(f"{'='*50}")

    # List all techniques
    all_tech_names = list_techniques()
    print(f"\nAll available techniques ({len(all_tech_names)}):")
    print(", ".join(all_tech_names))
    
    # List techniques in a specific category (using identifier prefix)
    few_shot_category = "2.2.1"
    few_shot_tech_names = list_techniques(category=few_shot_category)
    print(f"\nFew-Shot techniques (identifier starts with {few_shot_category}):")
    print(", ".join(few_shot_tech_names))

    # Get a specific technique instance by name
    technique_name = "chain_of_verification"
    cov_technique = get_technique(technique_name)
    if cov_technique:
        print(f"\nSuccessfully retrieved technique: {cov_technique.name}")
        print(f"Description: {cov_technique.description}")
        # You could now generate a prompt or execute it:
        # prompt = cov_technique.generate_prompt("Explain black holes simply.")
        # print(f"\nGenerated prompt for {technique_name}:\n{prompt}")
    else:
        print(f"\nCould not find technique: {technique_name}")

def main():
    demonstrate_composing_techniques()
    demonstrate_listing_and_getting_techniques()
    
    print("\nDone!")

if __name__ == "__main__":
    # Load environment variables before checking API_KEY_SET
    load_dotenv()
    API_KEY_SET = bool(os.environ.get("OPENROUTER_API_KEY")) # Re-check after loading .env

    # Add a check for the API key before running main
    if not API_KEY_SET or os.environ.get("OPENROUTER_API_KEY") == "YOUR_API_KEY_HERE": # Also check placeholder
        print("Warning: OPENROUTER_API_KEY environment variable is not set or is placeholder.")
        print("LLM execution will be skipped in the examples.")
        print("Please set it in the '.env' file if you want to interact with the OpenRouter API.")
    main() 