"""
Automatic Prompt Engineer (APE) implementation.

Based on the paper: "Large Language Models Are Human-Level Prompt Engineers"
https://arxiv.org/abs/2211.01910
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random

from ..base import PromptTechnique
from ..utils import call_llm, log
from . import PromptOptimizer


@dataclass
class PromptCandidate:
    """Represents a candidate prompt with its evaluation score."""
    prompt: str
    score: float
    metadata: Dict[str, Any]


class APEOptimizer(PromptOptimizer):
    """
    Automatic Prompt Engineer (APE) optimizer.
    
    APE generates and evaluates natural language instructions to find
    the best prompt for a given task using a small labeled dataset.
    """
    
    def __init__(self):
        super().__init__(
            name="APE",
            description="Automatic Prompt Engineer - generates and ranks prompts"
        )
    
    async def optimize(
        self,
        task_description: str,
        seed_prompts: List[str],
        labeled_examples: List[Dict[str, str]],
        technique: Optional[PromptTechnique] = None,
        num_candidates: int = 50,
        num_demos: int = 3,
        temperature: float = 0.7,
        eval_batch_size: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize prompts using the APE algorithm.
        
        Args:
            task_description: Natural language description of the task
            seed_prompts: Initial prompts to inspire generation
            labeled_examples: List of {"input": str, "output": str} pairs
            technique: Optional prompt technique to optimize within
            num_candidates: Number of prompt candidates to generate
            num_demos: Number of demonstrations to include in generation
            temperature: Temperature for prompt generation
            eval_batch_size: Batch size for parallel evaluation
            
        Returns:
            Optimization results including best prompt and history
        """
        log.info(f"[bold cyan]Starting APE optimization for task: {task_description}[/]")
        
        # Step 1: Generate prompt candidates
        candidates = await self._generate_candidates(
            task_description,
            seed_prompts,
            labeled_examples[:num_demos],
            num_candidates,
            temperature
        )
        
        log.info(f"Generated {len(candidates)} prompt candidates")
        
        # Step 2: Evaluate candidates on labeled examples
        scored_candidates = await self._evaluate_candidates(
            candidates,
            labeled_examples,
            technique,
            eval_batch_size
        )
        
        if not scored_candidates:
            raise RuntimeError("APE produced no scored candidates")

        # Step 3: Select best prompt
        best_candidate = max(scored_candidates, key=lambda x: x.score)
        
        # Store optimization history
        self.optimization_history.append({
            "task": task_description,
            "num_candidates": len(candidates),
            "num_examples": len(labeled_examples),
            "best_score": best_candidate.score,
            "all_scores": [c.score for c in scored_candidates]
        })
        
        return {
            "best_prompt": best_candidate.prompt,
            "score": best_candidate.score,
            "metadata": {
                "algorithm": "APE",
                "num_candidates_evaluated": len(candidates),
                "technique_used": technique.name if technique else None,
                **best_candidate.metadata
            },
            "history": self.optimization_history,
            "all_candidates": [
                {"prompt": c.prompt, "score": c.score}
                for c in sorted(scored_candidates, key=lambda x: x.score, reverse=True)[:10]
            ]
        }
    
    async def _generate_candidates(
        self,
        task_description: str,
        seed_prompts: List[str],
        demo_examples: List[Dict[str, str]],
        num_candidates: int,
        temperature: float
    ) -> List[str]:
        """Generate prompt candidates using forward generation."""
        
        # Create demonstrations string
        demos = "\n".join([
            f"Input: {ex['input']}\nOutput: {ex['output']}"
            for ex in demo_examples
        ])
        
        # Use seed prompts for inspiration if provided
        seed_section = ""
        if seed_prompts:
            seed_section = "\nHere are some example instructions for inspiration:\n" + \
                          "\n".join([f"- {p}" for p in seed_prompts])
        
        generation_prompt = f"""Task: {task_description}

Examples:
{demos}
{seed_section}

Generate {num_candidates} diverse instruction prompts that would help an AI system perform this task correctly. 
Each instruction should be clear, specific, and approach the task from a different angle.

Output each instruction on a new line, numbered 1 to {num_candidates}:"""
        
        try:
            response = await asyncio.to_thread(
                call_llm,
                generation_prompt,
                system_prompt="You are an expert prompt engineer helping to discover effective prompts.",
                config_override={"temperature": temperature}
            )
            
            # Parse generated candidates
            candidates = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove numbering if present
                if line and line[0].isdigit() and '.' in line[:3]:
                    line = line.split('.', 1)[1].strip()
                if line:
                    candidates.append(line)
            
            # Ensure we have enough candidates
            while len(candidates) < num_candidates:
                candidates.append(random.choice(seed_prompts) if seed_prompts else 
                               f"Please {task_description}")
            
            return candidates[:num_candidates]
            
        except Exception as e:
            log.error(f"Error generating candidates: {e}")
            raise
    
    async def _evaluate_candidates(
        self,
        candidates: List[str],
        labeled_examples: List[Dict[str, str]],
        technique: Optional[PromptTechnique],
        batch_size: int
    ) -> List[PromptCandidate]:
        """Evaluate candidates on labeled examples."""
        
        scored_candidates = []
        
        # Process candidates in batches
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            batch_tasks = []
            
            for candidate in batch:
                task = self._evaluate_single_candidate(
                    candidate,
                    labeled_examples,
                    technique
                )
                batch_tasks.append(task)
            
            # Run batch evaluations in parallel
            batch_results = await asyncio.gather(*batch_tasks)
            scored_candidates.extend(batch_results)
            
            log.info(f"Evaluated batch {i//batch_size + 1}/{(len(candidates) + batch_size - 1)//batch_size}")
        
        return scored_candidates
    
    async def _evaluate_single_candidate(
        self,
        candidate: str,
        labeled_examples: List[Dict[str, str]],
        technique: Optional[PromptTechnique]
    ) -> PromptCandidate:
        """Evaluate a single candidate prompt."""
        
        correct = 0
        total = len(labeled_examples)
        
        for example in labeled_examples:
            try:
                # Construct full prompt
                if technique:
                    # Use the technique's prompt generation
                    full_prompt = technique.generate_prompt(
                        example["input"],
                        custom_instructions=candidate,
                    )
                else:
                    full_prompt = f"{candidate}\n\nInput: {example['input']}\nOutput:"
                
                # Get model response
                response = await asyncio.to_thread(
                    call_llm,
                    full_prompt,
                    config_override={"temperature": 0.0, "max_tokens": 100}
                )
                
                # Simple exact match evaluation (can be improved)
                if self._evaluate_response(response.strip(), example["output"]):
                    correct += 1
                    
            except Exception as e:
                log.debug(f"Error evaluating candidate: {e}")
                continue
        
        score = correct / total if total > 0 else 0.0
        
        return PromptCandidate(
            prompt=candidate,
            score=score,
            metadata={
                "correct": correct,
                "total": total,
                "technique": technique.name if technique else "raw"
            }
        )
    
    def _evaluate_response(self, response: str, expected: str) -> bool:
        """
        Evaluate if response matches expected output.
        
        This is a simple implementation - can be enhanced with:
        - Semantic similarity
        - Task-specific evaluation metrics
        - LLM-based evaluation
        """
        # Normalize strings for comparison
        response = response.lower().strip()
        expected = expected.lower().strip()
        
        # Check exact match
        if response == expected:
            return True
        
        # Check if expected is contained in response
        if expected in response:
            return True
        
        # For numeric outputs, try float comparison
        try:
            return abs(float(response) - float(expected)) < 0.01
        except (ValueError, TypeError):
            pass
        
        return False
