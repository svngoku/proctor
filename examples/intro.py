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

problem = "What is the most impactful fact about the moon?"


response = expert_cot.execute(problem, role="Expert Moon Facts Researcher", verbose=True)
response