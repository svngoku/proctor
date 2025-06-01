"""
Implementation of Decomposition prompting techniques.
"""

from ..base import PromptTechnique
from ..utils import dedent_prompt


class DECOMP(PromptTechnique):
    """
    DECOMP breaks down complex problems into simpler subproblems.
    (Note: Guides the LLM to perform decomposition)
    """

    def __init__(self):
        """Initialize DECOMP technique."""
        super().__init__(
            name="DECOMP",
            identifier="2.2.4",
            description="Breaks down complex problems into simpler subproblems.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a DECOMP prompt.

        Args:
            input_text (str): Input text (the complex problem)
            **kwargs: Additional arguments:
                - num_subproblems (int): Number of subproblems to identify (default: 3)
                - approach (str): Decomposition approach ("sequential", "parallel", "hierarchical")
                - domain (str): Optional specific domain context
                - clear_dependencies (bool): Whether to explicitly track dependencies between subproblems

        Returns:
            str: Generated DECOMP prompt
        """
        num_subproblems = kwargs.get("num_subproblems", 3)
        approach = kwargs.get("approach", "sequential")
        domain = kwargs.get("domain", "")
        clear_dependencies = kwargs.get("clear_dependencies", False)

        domain_context = f" in the {domain} domain" if domain else ""

        approach_guidance = {
            "sequential": "Break the problem down into sequential steps, where each subproblem builds on the previous one.",
            "parallel": "Identify independent aspects of the problem that can be solved separately and then combined.",
            "hierarchical": "Break the problem into major components, then further decompose each component as needed.",
        }.get(
            approach,
            "Break the problem down into manageable parts that are easier to solve individually.",
        )

        dependencies_text = (
            "\n- Explicitly note how each subproblem depends on or relates to others"
            if clear_dependencies
            else ""
        )

        # Generate subproblems dynamically
        subproblems = ""
        for i in range(num_subproblems):
            subproblems += f"""
        Subproblem {i + 1}: [Identify and precisely describe a clear, specific aspect of the main problem]
        - Why this subproblem is important: [Explain why solving this contributes to the overall solution]
        - Key information needed: [Identify what data/concepts are needed to solve this part]
        
        Solution to Subproblem {i + 1}:
        [Solve this subproblem with clear, systematic reasoning]
        
        """

        prompt = dedent_prompt(f"""
        # Complex Problem Analysis{domain_context}:
        
        Problem Statement: {input_text}
        
        ## Decomposition Strategy:
        {approach_guidance}{dependencies_text}
        
        ## Breaking Down the Problem:
        {subproblems}
        ## Integration and Synthesis:
        [Explain how the solutions to subproblems connect and build toward the complete solution]
        [Identify any important insights that emerge from combining the partial solutions]
        [Address any remaining aspects not covered by the individual subproblems]
        
        ## Final Comprehensive Solution:
        [Provide the complete, integrated solution to the original problem]
        
        ## Verification:
        [Verify that the solution addresses all aspects of the original problem]
        [Check for consistency and correctness across subproblem solutions]
        """)
        return prompt


class FaithfulCoT(PromptTechnique):
    """
    Faithful CoT ensures reasoning steps are faithful to the problem requirements.

    This technique emphasizes maintaining fidelity to the original problem
    throughout the reasoning process, avoiding drift or misinterpretation.
    """

    def __init__(self):
        """Initialize Faithful CoT technique."""
        super().__init__(
            name="Faithful CoT",
            identifier="2.2.4",
            description="Ensures reasoning steps remain faithful to problem requirements.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Faithful CoT prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - fidelity_checks (List[str]): Specific fidelity aspects to monitor
                - constraint_tracking (bool): Whether to explicitly track constraints

        Returns:
            str: Generated Faithful CoT prompt
        """
        fidelity_checks = kwargs.get(
            "fidelity_checks",
            ["problem_alignment", "constraint_adherence", "scope_maintenance"],
        )
        constraint_tracking = kwargs.get("constraint_tracking", True)

        checks_text = "\n".join(
            [
                f"- {check.replace('_', ' ').title()}: [Monitor {check.replace('_', ' ')}]"
                for check in fidelity_checks
            ]
        )

        constraint_text = (
            """
        
        Constraint Tracking:
        [Explicitly list and track all constraints throughout reasoning]
        """
            if constraint_tracking
            else ""
        )

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll solve this using Faithful CoT, ensuring every reasoning step remains faithful to the original problem.
        
        Problem Fidelity Framework:
        {checks_text}{constraint_text}
        
        Step 1: Problem Understanding and Constraint Identification
        [Carefully analyze the problem and identify all requirements and constraints]
        [FIDELITY CHECK: Does my understanding align with the original problem?]
        
        Step 2: Solution Approach Selection
        [Choose approach that directly addresses the problem requirements]
        [FIDELITY CHECK: Does this approach stay within problem scope?]
        
        Step 3: Systematic Problem Solving
        [Execute solution while continuously checking fidelity to requirements]
        [FIDELITY CHECK: Are all steps addressing the actual problem?]
        
        Step 4: Solution Validation
        [Verify solution meets all original requirements and constraints]
        [FIDELITY CHECK: Does the solution fully and faithfully address the problem?]
        
        Faithful Solution:
        [Final answer that maintains complete fidelity to the original problem]
        """)
        return prompt


class LeastToMost(PromptTechnique):
    """
    Least-to-Most prompting solves problems by starting with simpler cases.

    This technique begins with the simplest version of a problem and gradually
    builds up to more complex versions, using insights from simpler cases.
    """

    def __init__(self):
        """Initialize Least-to-Most technique."""
        super().__init__(
            name="Least-to-Most",
            identifier="2.2.4",
            description="Solves problems by starting with simpler cases and building up.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Least-to-Most prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - complexity_levels (int): Number of complexity levels to use
                - progression_strategy (str): Strategy for complexity progression

        Returns:
            str: Generated Least-to-Most prompt
        """
        complexity_levels = kwargs.get("complexity_levels", 3)
        progression_strategy = kwargs.get("progression_strategy", "gradual")

        strategy_guidance = {
            "gradual": "Gradually increase complexity with small incremental steps",
            "exponential": "Increase complexity exponentially at each level",
            "targeted": "Focus on specific aspects that increase complexity",
        }.get(progression_strategy, "Gradually increase complexity")

        levels_text = []
        for i in range(complexity_levels):
            level_num = i + 1
            if level_num == 1:
                levels_text.append(f"""
        Level {level_num} - Simplest Case:
        [Identify and solve the simplest version of this problem]
        [Focus on core concepts without complications]
        Solution {level_num}: [Simple case solution]
        """)
            elif level_num == complexity_levels:
                levels_text.append(f"""
        Level {level_num} - Full Complexity:
        [Apply insights from previous levels to solve the original problem]
        [Build upon all previous solutions]
        Solution {level_num}: [Complete solution to original problem]
        """)
            else:
                levels_text.append(f"""
        Level {level_num} - Intermediate Case:
        [Increase complexity from Level {level_num - 1}]
        [Build upon previous insights while adding new elements]
        Solution {level_num}: [Intermediate solution]
        """)

        levels_content = "\n".join(levels_text)

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll use Least-to-Most prompting to solve this by starting with simpler cases.
        
        Progression Strategy: {strategy_guidance}
        Complexity Levels: {complexity_levels}
        
        {levels_content}
        
        Pattern Recognition:
        [Identify patterns and insights that emerged across complexity levels]
        [Note how solutions evolved from simple to complex cases]
        
        Final Integrated Solution:
        [Complete solution leveraging insights from all complexity levels]
        """)
        return prompt


class PlanAndSolve(PromptTechnique):
    """
    Plan-and-Solve separates planning from execution for systematic problem-solving.

    This technique first creates a detailed plan for solving the problem,
    then systematically executes that plan step by step.
    """

    def __init__(self):
        """Initialize Plan-and-Solve technique."""
        super().__init__(
            name="Plan-and-Solve",
            identifier="2.2.4",
            description="Separates planning from execution for systematic problem-solving.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Plan-and-Solve prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - planning_depth (str): Depth of planning phase
                - execution_monitoring (bool): Whether to monitor execution progress

        Returns:
            str: Generated Plan-and-Solve prompt
        """
        planning_depth = kwargs.get("planning_depth", "detailed")
        execution_monitoring = kwargs.get("execution_monitoring", True)

        depth_guidance = {
            "basic": "Create a high-level plan with major steps",
            "detailed": "Develop a comprehensive plan with specific sub-steps",
            "exhaustive": "Create an exhaustive plan covering all possible contingencies",
        }.get(planning_depth, "Develop a comprehensive plan")

        monitoring_text = (
            """
        
        Execution Monitoring:
        [After each step, assess progress and adjust plan if needed]
        """
            if execution_monitoring
            else ""
        )

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll use Plan-and-Solve to systematically approach this problem.
        
        PHASE 1 - PLANNING:
        
        Planning Depth: {planning_depth} - {depth_guidance}
        
        Problem Analysis:
        [Analyze the problem to understand requirements and constraints]
        
        Goal Definition:
        [Clearly define what constitutes a successful solution]
        
        Resource Assessment:
        [Identify available information, tools, and methods]
        
        Solution Strategy:
        [Outline the overall approach to solving the problem]
        
        Detailed Plan:
        Step 1: [Specific action with clear objective]
        Step 2: [Next action building on Step 1]
        Step 3: [Continue systematic progression]
        [Add more steps as needed]
        
        Risk Assessment:
        [Identify potential challenges and mitigation strategies]
        
        PHASE 2 - EXECUTION:
        
        Systematic Execution:{monitoring_text}
        
        Executing Step 1:
        [Implement first step according to plan]
        [Result and assessment]
        
        Executing Step 2:
        [Implement second step according to plan]
        [Result and assessment]
        
        [Continue for all planned steps]
        
        PHASE 3 - VALIDATION:
        
        Plan Adherence Check:
        [Verify that execution followed the plan appropriately]
        
        Goal Achievement Assessment:
        [Confirm that the solution meets the original objectives]
        
        Final Solution:
        [Present the complete solution achieved through planned execution]
        """)
        return prompt


class ProgramOfThought(PromptTechnique):
    """
    Program-of-Thought expresses reasoning as executable programs or algorithms.

    This technique structures reasoning in a program-like format with clear
    inputs, processes, and outputs, making reasoning more systematic and verifiable.
    """

    def __init__(self):
        """Initialize Program-of-Thought technique."""
        super().__init__(
            name="Program-of-Thought",
            identifier="2.2.4",
            description="Expresses reasoning as executable programs or algorithms.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Program-of-Thought prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - programming_style (str): Style of program structure
                - include_debugging (bool): Whether to include debugging steps

        Returns:
            str: Generated Program-of-Thought prompt
        """
        programming_style = kwargs.get("programming_style", "procedural")
        include_debugging = kwargs.get("include_debugging", True)

        style_guidance = {
            "procedural": "Structure as a sequence of procedures and functions",
            "object_oriented": "Structure using objects and methods",
            "functional": "Structure using functional programming principles",
            "algorithmic": "Structure as a clear algorithm with defined steps",
        }.get(programming_style, "Structure as a systematic program")

        debugging_text = (
            """
        
        DEBUGGING AND TESTING:
        [Test the program logic with sample inputs]
        [Identify and fix any logical errors]
        [Verify program correctness]
        """
            if include_debugging
            else ""
        )

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll solve this using Program-of-Thought, expressing my reasoning as an executable program.
        
        Programming Style: {programming_style} - {style_guidance}
        
        PROGRAM SPECIFICATION:
        
        Input: {input_text}
        Output: [Define expected output format]
        
        PROGRAM DESIGN:
        
        Main Function:
        def solve_problem(input_data):
            # Initialize variables
            [Define necessary variables and data structures]
            
            # Step 1: [First logical operation]
            result_1 = process_step_1(input_data)
            
            # Step 2: [Second logical operation]
            result_2 = process_step_2(result_1)
            
            # Step 3: [Continue logical sequence]
            result_3 = process_step_3(result_2)
            
            # Final processing
            final_result = finalize_solution(result_3)
            
            return final_result
        
        HELPER FUNCTIONS:
        
        def process_step_1(data):
            [Implement first processing step]
            return processed_data
        
        def process_step_2(data):
            [Implement second processing step]
            return processed_data
        
        [Define additional helper functions as needed]
        
        PROGRAM EXECUTION:
        
        [Execute the program step by step]
        [Show intermediate results at each step]
        [Trace through the logical flow]{debugging_text}
        
        PROGRAM OUTPUT:
        [Final result from program execution]
        """)
        return prompt


class RecursionOfThought(PromptTechnique):
    """
    Recursion-of-Thought applies recursive thinking to break down problems.

    This technique uses recursive problem-solving strategies, breaking problems
    into smaller instances of the same problem type.
    """

    def __init__(self):
        """Initialize Recursion-of-Thought technique."""
        super().__init__(
            name="Recursion-of-Thought",
            identifier="2.2.4",
            description="Applies recursive thinking to break down problems.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Recursion-of-Thought prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - recursion_depth (int): Maximum recursion depth
                - base_case_strategy (str): Strategy for identifying base cases

        Returns:
            str: Generated Recursion-of-Thought prompt
        """
        recursion_depth = kwargs.get("recursion_depth", 4)
        base_case_strategy = kwargs.get("base_case_strategy", "simplest")

        strategy_guidance = {
            "simplest": "Identify the simplest case that can be solved directly",
            "minimal": "Find the minimal problem instance that requires no further breakdown",
            "trivial": "Locate trivial cases with obvious solutions",
        }.get(base_case_strategy, "Identify cases that can be solved directly")

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll solve this using Recursion-of-Thought, breaking it into smaller instances of the same problem.
        
        Base Case Strategy: {base_case_strategy} - {strategy_guidance}
        Maximum Recursion Depth: {recursion_depth}
        
        RECURSIVE PROBLEM ANALYSIS:
        
        Step 1: Base Case Identification
        [Identify the simplest version of this problem that can be solved directly]
        Base Case: [Define the base case]
        Base Solution: [Solve the base case]
        
        Step 2: Recursive Case Definition
        [Define how to break the current problem into smaller instances]
        Recursive Breakdown: [Show how to reduce problem size]
        
        Step 3: Recursive Solution Structure
        
        def solve_recursively(problem, depth=0):
            # Check recursion depth limit
            if depth > {recursion_depth}:
                return "Maximum depth reached"
            
            # Base case check
            if is_base_case(problem):
                return solve_base_case(problem)
            
            # Recursive case
            smaller_problems = break_down_problem(problem)
            partial_solutions = []
            
            for sub_problem in smaller_problems:
                solution = solve_recursively(sub_problem, depth + 1)
                partial_solutions.append(solution)
            
            # Combine partial solutions
            final_solution = combine_solutions(partial_solutions)
            return final_solution
        
        RECURSIVE EXECUTION:
        
        Level 0 (Original Problem):
        [Apply recursive breakdown to original problem]
        
        Level 1 (Sub-problems):
        [Show first level of problem decomposition]
        
        Level 2 (Further breakdown):
        [Continue recursive decomposition if needed]
        
        [Continue until base cases are reached]
        
        SOLUTION COMBINATION:
        [Combine solutions from recursive calls to build final answer]
        
        Recursive Solution:
        [Final answer built through recursive problem-solving]
        """)
        return prompt


class SkeletonOfThought(PromptTechnique):
    """
    Skeleton-of-Thought creates a reasoning skeleton before filling in details.

    This technique first establishes the overall structure and key points
    of the reasoning, then fills in the detailed content.
    """

    def __init__(self):
        """Initialize Skeleton-of-Thought technique."""
        super().__init__(
            name="Skeleton-of-Thought",
            identifier="2.2.4",
            description="Creates a reasoning skeleton before filling in details.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Skeleton-of-Thought prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - skeleton_depth (str): Depth of skeleton structure
                - expansion_strategy (str): Strategy for expanding skeleton

        Returns:
            str: Generated Skeleton-of-Thought prompt
        """
        skeleton_depth = kwargs.get("skeleton_depth", "detailed")
        expansion_strategy = kwargs.get("expansion_strategy", "systematic")

        depth_guidance = {
            "basic": "Create a high-level skeleton with main points",
            "detailed": "Develop a comprehensive skeleton with sub-points",
            "exhaustive": "Create an exhaustive skeleton covering all aspects",
        }.get(skeleton_depth, "Develop a comprehensive skeleton")

        strategy_guidance = {
            "systematic": "Expand each skeleton point systematically",
            "priority": "Expand high-priority skeleton points first",
            "iterative": "Iteratively refine and expand the skeleton",
        }.get(expansion_strategy, "Expand systematically")

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll use Skeleton-of-Thought to first create a reasoning structure, then fill in details.
        
        Skeleton Depth: {skeleton_depth} - {depth_guidance}
        Expansion Strategy: {expansion_strategy} - {strategy_guidance}
        
        PHASE 1 - SKELETON CREATION:
        
        Main Reasoning Structure:
        I. [First major reasoning point]
           A. [Sub-point 1]
           B. [Sub-point 2]
           C. [Sub-point 3]
        
        II. [Second major reasoning point]
            A. [Sub-point 1]
            B. [Sub-point 2]
            C. [Sub-point 3]
        
        III. [Third major reasoning point]
             A. [Sub-point 1]
             B. [Sub-point 2]
             C. [Sub-point 3]
        
        IV. [Conclusion/Solution point]
            A. [Final synthesis]
            B. [Answer formulation]
        
        PHASE 2 - SKELETON EXPANSION:
        
        Expanding Point I: [First major reasoning point]
        A. [Detailed explanation of sub-point 1]
        B. [Detailed explanation of sub-point 2]
        C. [Detailed explanation of sub-point 3]
        
        Expanding Point II: [Second major reasoning point]
        A. [Detailed explanation of sub-point 1]
        B. [Detailed explanation of sub-point 2]
        C. [Detailed explanation of sub-point 3]
        
        Expanding Point III: [Third major reasoning point]
        A. [Detailed explanation of sub-point 1]
        B. [Detailed explanation of sub-point 2]
        C. [Detailed explanation of sub-point 3]
        
        Expanding Point IV: [Conclusion/Solution]
        A. [Detailed synthesis of all points]
        B. [Complete answer formulation]
        
        PHASE 3 - INTEGRATION:
        
        [Integrate all expanded points into a coherent, complete solution]
        
        Skeleton-Based Solution:
        [Final answer built through structured skeleton approach]
        """)
        return prompt


class TreeOfThought(PromptTechnique):
    """
    Tree-of-Thought explores multiple reasoning paths in a tree structure.

    This technique explores different reasoning branches, evaluates them,
    and selects the most promising paths for further exploration.
    """

    def __init__(self):
        """Initialize Tree-of-Thought technique."""
        super().__init__(
            name="Tree-of-Thought",
            identifier="2.2.4",
            description="Explores multiple reasoning paths in a tree structure.",
        )

    def generate_prompt(self, input_text: str, **kwargs) -> str:
        """
        Generate a Tree-of-Thought prompt.

        Args:
            input_text (str): Input text
            **kwargs: Additional arguments
                - branching_factor (int): Number of branches to explore at each level
                - tree_depth (int): Maximum depth of the reasoning tree
                - evaluation_criteria (List[str]): Criteria for evaluating branches

        Returns:
            str: Generated Tree-of-Thought prompt
        """
        branching_factor = kwargs.get("branching_factor", 3)
        tree_depth = kwargs.get("tree_depth", 3)
        evaluation_criteria = kwargs.get(
            "evaluation_criteria", ["feasibility", "completeness", "efficiency"]
        )

        criteria_text = ", ".join(evaluation_criteria)

        # Generate tree structure
        tree_levels = []
        for level in range(tree_depth):
            level_num = level + 1
            if level == 0:
                tree_levels.append(f"""
        Level {level_num} - Initial Branches:
        Branch 1.1: [First reasoning approach]
        Branch 1.2: [Second reasoning approach]
        Branch 1.3: [Third reasoning approach]
        
        Evaluation of Level {level_num} branches using criteria: {criteria_text}
        Selected branches for expansion: [Choose most promising branches]
        """)
            else:
                tree_levels.append(f"""
        Level {level_num} - Extended Branches:
        [Expand selected branches from Level {level}]
        Branch {level_num}.1: [Extension of selected branch]
        Branch {level_num}.2: [Alternative extension]
        Branch {level_num}.3: [Another extension]
        
        Evaluation of Level {level_num} branches using criteria: {criteria_text}
        Selected branches for further exploration: [Choose best branches]
        """)

        tree_content = "\n".join(tree_levels)

        prompt = dedent_prompt(f"""
        Problem: {input_text}
        
        I'll use Tree-of-Thought to explore multiple reasoning paths systematically.
        
        Tree Parameters:
        - Branching Factor: {branching_factor}
        - Maximum Depth: {tree_depth}
        - Evaluation Criteria: {criteria_text}
        
        TREE EXPLORATION:
        
        Root: Problem Analysis
        [Analyze the problem to identify potential reasoning directions]
        
        {tree_content}
        
        BRANCH COMPARISON AND SELECTION:
        
        Final Branch Evaluation:
        [Compare all viable branches using evaluation criteria]
        [Identify the most promising complete reasoning path]
        
        Optimal Path Selection:
        [Select the best reasoning path through the tree]
        
        SOLUTION SYNTHESIS:
        
        [Follow the optimal path to generate the final solution]
        [Incorporate insights from explored alternative branches]
        
        Tree-of-Thought Solution:
        [Final answer derived from systematic tree exploration]
        """)
        return prompt
