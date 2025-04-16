import os
from dotenv import load_dotenv
from proctor import CompositeTechnique, RolePrompting, ChainOfThought, ChainOfVerification, SelfConsistency

load_dotenv()

# Get a specific technique instance

expert_cot = CompositeTechnique(
    name="Expert Chain-of-Thought",
    identifier="custom-expert-cot",
    techniques=[
        RolePrompting(),
        ChainOfThought(),
        ChainOfVerification(),
        SelfConsistency(),
    ]
)

problem = "How to build a EKS cluster in AWS?"

# Generate the combined prompt
# prompt = expert_cot.generate_prompt(problem, role="experienced travel planner")
# print(prompt)


response = expert_cot.execute(problem, role="DevOps Engineer", verbose=True)
response