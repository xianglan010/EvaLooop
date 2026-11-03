"""Main evaluator class for EvaLoop framework."""

import json
import logging
import random
from typing import Dict, Any, List
from pathlib import Path

from .config import EvaluationConfig
from ..models.factory import ModelFactory
from ..tasks.factory import TaskFactory
from ..evaluation.cycle import BatchEvaluationCycle
from ..evaluation.py_testing import EvalPlusCodeTester, load_mbpp_dataset


class EvaLoopEvaluator:
    """Main evaluator class for running EvaLoop experiments."""
    
    # Constants
    TASK_OUTPUT_KEY_TEMPLATE = "task_{}_output"
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: Evaluation configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model_factory = ModelFactory()
        self.task_factory = TaskFactory()
        
        self.logger.info(f"Initialized EvaLoopEvaluator with config: {config.experiment_name}")
    
    def run_code_generation_summarization(self) -> List[Dict[str, Any]]:
        """
        Run code generation and summarization evaluation.
        
        Returns:
            List of evaluation results (single model result in list for consistency).
        """
        self.logger.info("Starting code generation and summarization evaluation")
        
        # Setup components
        model = self._create_model()
        tasks = self._create_tasks()
        prompts, py_problems = self._load_dataset()
        tester = self._create_tester(py_problems)
        
        # Run evaluation
        model_result = self._run_evaluation_cycles(model, tasks, prompts, py_problems, tester)
        
        # Save and return results
        self._save_results([model_result], "code_generation_summarization")
        return [model_result]
    
    def run_code_translation(self) -> Dict[str, Any]:
        """
        Run code translation evaluation.
        
        Note: This method is currently not implemented. 
        It's kept as an interface for future extension.
        
        Returns:
            Dictionary containing evaluation results.
        """
        raise NotImplementedError("Code translation evaluation is not currently supported")
    
    def _create_model(self):
        """Create and configure the model."""
        model_config = self.config.get_model_config()
        model = self.model_factory.create_model(model_config)
        self.logger.info(f"Created model for evaluation: {model}")
        return model
    
    def _create_tasks(self):
        """Create task instances."""
        task_configs = self.config.get_task_configs("code_generation_summarization")
        tasks = [self.task_factory.create_task(config) for config in task_configs]
        self.logger.info(f"Created {len(tasks)} tasks for evaluation")
        return tasks
    
    def _load_dataset(self):
        """Load and prepare the dataset."""
        py_problems, ground_truth = load_mbpp_dataset()
        self.logger.info(f"Loaded MBPP dataset with {len(py_problems)} problems")

        if self.config.task_limit is not None:
            all_task_ids = list(py_problems)
            if self.config.task_limit > len(all_task_ids):
                raise ValueError(
                    f"task_limit={self.config.task_limit} exceeds dataset size "
                    f"{len(all_task_ids)}"
                )
            selected_ids = random.Random(self.config.task_seed).sample(
                all_task_ids, self.config.task_limit
            )
            py_problems = {task_id: py_problems[task_id] for task_id in selected_ids}
            ground_truth = {
                task_id: ground_truth[task_id] for task_id in selected_ids
            }
            self.logger.info(
                "Selected %d tasks with seed %d",
                len(selected_ids), self.config.task_seed
            )
        
        # Prepare prompts from dataset
        prompts = []
        for task_id, problem in py_problems.items():
            prompts.append({
                "task_id": task_id,
                "prompt": problem["prompt"]
            })
        
        self.logger.info(f"Prepared {len(prompts)} prompts from MBPP dataset")
        return prompts, py_problems
    
    def _create_tester(self, py_problems):
        """Create the code tester."""
        _, ground_truth = load_mbpp_dataset()  # Get ground truth
        ground_truth = {
            task_id: ground_truth[task_id] for task_id in py_problems
        }
        return EvalPlusCodeTester(py_problems, ground_truth)
    
    def _run_evaluation_cycles(self, model, tasks, prompts, py_problems, tester) -> Dict[str, Any]:
        """Run the main evaluation cycles."""
        self.logger.info(f"Evaluating model: {model}")
        
        # Create evaluation cycle runner
        cycle_runner = BatchEvaluationCycle(model, tasks)
        
        # Initialize results structure
        model_result = self._initialize_model_result(model, prompts)
        
        # Run cycles until max_cycles or all prompts fail
        current_prompts = prompts.copy()
        current_cycle = 0
        
        while current_cycle < self.config.max_cycles and current_prompts:
            current_cycle += 1
            self.logger.info(f"Starting cycle {current_cycle}/{self.config.max_cycles}")
            
            # Run single cycle
            cycle_output = cycle_runner.run_single_cycle(current_prompts, current_cycle)
            
            # Process results and prepare for next cycle
            current_prompts = self._process_cycle_results(
                cycle_output, model_result, tasks, py_problems, tester, current_cycle
            )
            self._save_checkpoint(model_result, current_cycle)
            
            if not current_prompts:
                self.logger.info(f"All prompts failed or completed after cycle {current_cycle}")
                break
        
        # Finalize results
        self._finalize_model_result(model_result)
        
        return model_result

    def _save_checkpoint(self, model_result: Dict[str, Any], cycle: int) -> None:
        """Atomically save the current state so long GPU runs are inspectable."""
        checkpoint = dict(model_result)
        checkpoint["completed_cycles"] = cycle
        output_dir = Path(self.config.output_dir)
        output_file = output_dir / f"{self.config.experiment_name}_checkpoint.json"
        temporary_file = output_file.with_suffix(".tmp")
        temporary_file.write_text(
            json.dumps([checkpoint], indent=2), encoding="utf-8"
        )
        temporary_file.replace(output_file)
    
    def _initialize_model_result(self, model, prompts) -> Dict[str, Any]:
        """Initialize the model result structure."""
        model_result = {
            "model": str(model),
            "configuration": self.config.loop_type,
            "task_seed": self.config.task_seed,
            "task_limit": self.config.task_limit,
            # Loop budget M; needed by the analyzer to normalize the ASL metrics
            "max_cycles": self.config.max_cycles,
            "prompt_results": []
        }
        
        for prompt in prompts:
            prompt_result = {
                "task_id": prompt["task_id"],
                "initial_prompt": prompt["prompt"],
                "cycles": [],
                "successful_cycles": 0,
                "max_cycles_reached": False
            }
            model_result["prompt_results"].append(prompt_result)
        
        return model_result
    
    def _process_cycle_results(self, cycle_output, model_result, tasks, py_problems, tester, current_cycle) -> List[Dict[str, Any]]:
        """Process cycle results and return prompts for next cycle."""
        next_cycle_prompts = []
        
        for cycle_result in cycle_output["cycle_results"]:
            task_id = cycle_result["task_id"]
            problem = py_problems.get(task_id)
            
            # Find corresponding prompt result
            prompt_result = next(r for r in model_result["prompt_results"] if r["task_id"] == task_id)
            
            # Test and validate the code generation output
            success = self._test_and_validate_output(cycle_result, tasks, task_id, problem, tester)
            cycle_result["success"] = success
            
            # Add cycle result to prompt result
            prompt_result["cycles"].append(cycle_result)
            
            # Update successful cycles and prepare for next cycle if successful
            if success:
                prompt_result["successful_cycles"] = current_cycle
                next_prompt = self._prepare_next_cycle_prompt(cycle_result, tasks, task_id)
                if next_prompt:
                    next_cycle_prompts.append(next_prompt)
        
        return next_cycle_prompts
    
    def _test_and_validate_output(self, cycle_result, tasks, task_id, problem, tester) -> bool:
        """Test and validate the code generation output."""
        # Find code generation task
        code_gen_index = self._find_code_generation_task_index(tasks)
        
        if code_gen_index is None:
            return False
        
        output_key = self.TASK_OUTPUT_KEY_TEMPLATE.format(code_gen_index + 1)
        
        if output_key not in cycle_result:
            return False
        
        # Test the generated code
        code_output = cycle_result[output_key]
        test_results = self._test_python_code(code_output, task_id, problem, tester)
        
        # Store test results
        test_results["plus_details"] = "".join(map(str, test_results["plus_details"]))
        cycle_result["test_results"] = test_results
        
        return test_results.get("success", False)
    
    def _find_code_generation_task_index(self, tasks) -> int:
        """Find the index of the task whose output should be functionally tested."""
        return next((i for i, task in enumerate(tasks)
                     if getattr(task, "produces_testable_code", False)), None)
    
    def _prepare_next_cycle_prompt(self, cycle_result, tasks, task_id) -> Dict[str, Any]:
        """Prepare prompt for the next cycle."""
        last_task_index = len(tasks)
        last_output_key = self.TASK_OUTPUT_KEY_TEMPLATE.format(last_task_index)
        
        if last_output_key in cycle_result:
            return {
                "task_id": task_id,
                "prompt": cycle_result[last_output_key]
            }
        
        return None
    
    def _finalize_model_result(self, model_result):
        """Finalize model results with summary statistics."""
        # Mark prompts that reached max cycles
        for prompt_result in model_result["prompt_results"]:
            prompt_result["max_cycles_reached"] = (
                prompt_result["successful_cycles"] == self.config.max_cycles
            )
        
        # Calculate average successful cycles
        if model_result["prompt_results"]:
            avg_cycles = sum(
                r["successful_cycles"] for r in model_result["prompt_results"]
            ) / len(model_result["prompt_results"])
            model_result["average_successful_cycles"] = avg_cycles
    
    def _test_python_code(self, output: str, task_id: str, problem: Dict, tester: EvalPlusCodeTester) -> Dict[str, Any]:
        """Test code output using the EvalPlus tester."""
        try:
            test_results = tester.test_code(
                output, 
                task_id, 
                problem["entry_point"]
            )
            return test_results
        except Exception as e:
            self.logger.error(f"Error testing code for task {task_id}: {e}")
            return {"success": False, "error": str(e)}
    
    
    def _save_results(self, results: List[Dict[str, Any]], experiment_type: str):
        """Save evaluation results to file."""
        output_file = Path(self.config.output_dir) / f"{self.config.experiment_name}_{experiment_type}_results.json"
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_file}")
    
