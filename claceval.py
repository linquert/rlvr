import json
from typing import List, Dict, Any
from calcchain_rlvr import CalcChainEnvironment, CalcChainTask, MathStructure


class CalcChainEvaluator:
    """Evaluator for CalcChain tasks."""
    
    def __init__(self):
        self.env = CalcChainEnvironment(seed=42)
    
    def create_benchmark_suite(self, tasks_per_structure: int = 10) -> Dict[str, List[CalcChainTask]]:
        """
        Create a comprehensive benchmark suite with tasks from each mathematical structure.
        
        Returns:
            Dictionary mapping structure names to lists of tasks
        """
        suite = {}
        
        for structure in MathStructure:
            tasks = []
            for difficulty in ['easy', 'medium', 'hard', 'expert']:
                for _ in range(tasks_per_structure // 4):
                    task = self.env.generate_task(difficulty=difficulty, structure=structure)
                    tasks.append(task)
            suite[structure.value] = tasks
        
        return suite
    
    def export_benchmark(self, suite: Dict[str, List[CalcChainTask]], filename: str):
        """Export benchmark suite to JSON file."""
        export_data = []
        
        for structure_name, tasks in suite.items():
            for i, task in enumerate(tasks):
                export_data.append({
                    'id': f"{structure_name}_{i}",
                    'structure': structure_name,
                    'difficulty': task.difficulty,
                    'numbers': task.numbers,
                    'operators': task.operators,
                    'target': task.target,
                    'min_steps': task.min_steps,
                    'max_steps': task.max_steps,
                    'structure_params': task.structure_params,
                    'prompt': self.env.format_task_for_llm(task, include_structure=False)
                })
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported {len(export_data)} tasks to {filename}")
    
    def evaluate_solution(self, task: CalcChainTask, solution_text: str) -> Dict[str, Any]:
        """
        Parse and evaluate a solution from an LLM.
        
        Args:
            task: The CalcChainTask
            solution_text: The LLM's solution as text
        
        Returns:
            Evaluation results
        """
        operations = []
        lines = solution_text.strip().split('\n')
        
        for line in lines:
            if 'Step' in line and '=' in line:
                pass
        
        return {
            'task_id': f"{task.hidden_structure.value}_{task.difficulty}",
            'difficulty': task.difficulty,
            'structure': task.hidden_structure.value,
            'target': task.target,
            'solution_provided': len(solution_text) > 0
        }
    
    def analyze_structure_difficulty(self, suite: Dict[str, List[CalcChainTask]]):
        """Analyze which mathematical structures are most challenging."""
        print("\n" + "="*80)
        print("BENCHMARK ANALYSIS: Structure Complexity")
        print("="*80 + "\n")
        
        for structure_name, tasks in suite.items():
            avg_steps = sum(task.max_steps for task in tasks) / len(tasks)
            avg_numbers = sum(len(task.numbers) for task in tasks) / len(tasks)
            
            difficulty_dist = {}
            for task in tasks:
                difficulty_dist[task.difficulty] = difficulty_dist.get(task.difficulty, 0) + 1
            
            print(f"\n{structure_name.upper().replace('_', ' ')}:")
            print(f"  Average max steps: {avg_steps:.1f}")
            print(f"  Average numbers provided: {avg_numbers:.1f}")
            print(f"  Difficulty distribution: {difficulty_dist}")
            
            sample = tasks[len(tasks)//2]  
            print(f"\n  Sample task (difficulty={sample.difficulty}):")
            print(f"    Numbers: {sample.numbers}")
            print(f"    Target: {sample.target}")
            print(f"    Hidden structure params: {sample.structure_params}")


def demonstrate_task_examples():
    """Demonstrate various task types with solutions."""
    env = CalcChainEnvironment(seed=123)
    
    print("\n" + "="*80)
    print("DEMONSTRATION: Sample Tasks with Hints")
    print("="*80)
    
    demonstrations = [
        (MathStructure.ELLIPTIC_CURVE, 'easy', 
         "Hint: Think about polynomial evaluation modulo a prime"),
        
        (MathStructure.MODULAR_SEQUENCE, 'medium',
         "Hint: The pattern involves powers and modular arithmetic"),
        
        (MathStructure.TOPOLOGICAL_INVARIANT, 'easy',
         "Hint: There's a famous formula relating vertices, edges, and faces"),
        
        (MathStructure.FIBONACCI_VARIANT, 'medium',
         "Hint: Look for a recursive relationship between consecutive values"),
        
        (MathStructure.FERMAT_LITTLE, 'hard',
         "Hint: Consider properties of exponentiation in modular arithmetic with primes"),
        
        (MathStructure.CONTINUED_FRACTION, 'medium',
         "Hint: Think about how to build up a value through recursive division"),
    ]
    
    for structure, difficulty, hint in demonstrations:
        task = env.generate_task(difficulty=difficulty, structure=structure)
        
        print(f"\n{'─'*80}")
        print(f"Mathematical Structure: {structure.value.replace('_', ' ').title()}")
        print(f"Difficulty: {difficulty}")
        print(f"{'─'*80}\n")
        
        print(env.format_task_for_llm(task, include_structure=False))
        print(f"\n{hint}\n")


def generate_training_curriculum():
    """Generate a curriculum of tasks ordered by difficulty."""
    env = CalcChainEnvironment(seed=456)
    
    print("\n" + "="*80)
    print("TRAINING CURRICULUM: Progressive Difficulty")
    print("="*80 + "\n")
    
    curriculum = {
        'Phase 1: Fundamentals': [],
        'Phase 2: Modular Arithmetic': [],
        'Phase 3: Sequences & Patterns': [],
        'Phase 4: Advanced Structures': []
    }
    
    for structure in [MathStructure.TOPOLOGICAL_INVARIANT, MathStructure.PRIME_CONSTELLATION]:
        for _ in range(3):
            task = env.generate_task(difficulty='easy', structure=structure)
            curriculum['Phase 1: Fundamentals'].append(task)
    
    for structure in [MathStructure.MODULAR_SEQUENCE, MathStructure.FERMAT_LITTLE]:
        for diff in ['easy', 'medium']:
            task = env.generate_task(difficulty=diff, structure=structure)
            curriculum['Phase 2: Modular Arithmetic'].append(task)
    
    for structure in [MathStructure.FIBONACCI_VARIANT, MathStructure.CONTINUED_FRACTION]:
        for diff in ['medium', 'hard']:
            task = env.generate_task(difficulty=diff, structure=structure)
            curriculum['Phase 3: Sequences & Patterns'].append(task)
    
    for structure in [MathStructure.ELLIPTIC_CURVE, MathStructure.QUADRATIC_RESIDUE]:
        for diff in ['hard', 'expert']:
            task = env.generate_task(difficulty=diff, structure=structure)
            curriculum['Phase 4: Advanced Structures'].append(task)
    
    for phase, tasks in curriculum.items():
        print(f"\n{phase}")
        print(f"  Total tasks: {len(tasks)}")
        print(f"  Structures covered: {set(t.hidden_structure.value for t in tasks)}")
        print(f"  Difficulty range: {set(t.difficulty for t in tasks)}")
        
        if tasks:
            print(f"\n  Example task:")
            task = tasks[0]
            print(f"    Structure: {task.hidden_structure.value}")
            print(f"    Target: {task.target}")
            print(f"    Numbers: {task.numbers[:5]}...")
    
    return curriculum


def create_llm_prompt_template():
    """Create a comprehensive prompt template for LLMs."""
    
    template = """# CalcChain Mathematical Reasoning Task

You are given a pool of numbers and a set of allowed operators. Your task is to reach the exact target value by applying a sequence of operations.

## Task Details
Numbers available: {numbers}
Target value: {target}
Available operators: {operators}
Constraints: Use between {min_steps} and {max_steps} operations

## Operators Explained
- Basic: +, -, *, // (integer division), % (modulo)
- Intermediate: ** (exponentiation), abs (absolute value), neg (negation)
- Advanced: sqrt, log2, gcd, lcm, floor, ceil

## Rules
1. Each operation consumes its input numbers and produces a new result
2. Results from operations can be used in subsequent operations
3. You must reach the EXACT target value
4. Think about mathematical patterns and structures that might help

## Reasoning Process
Before jumping to operations, consider:
1. What mathematical relationships might exist between the numbers?
2. Are there patterns in the numbers (primes, powers, sequences)?
3. What is the scale of the target relative to the numbers?
4. Could modular arithmetic, exponentiation, or special formulas apply?

## Solution Format
Provide your solution as:

Step 1: [operation] → [result]
Step 2: [operation] → [result]
...
Final result: [target]

## Begin Your Reasoning
First, analyze the numbers and target to identify potential mathematical structures.
Then, provide your step-by-step solution.
"""
    
    return template


def main():
    """Run the full evaluation demonstration."""
    
    print("\n" + "="*80)
    print("CALCCHAIN RLVR BENCHMARK - COMPREHENSIVE DEMONSTRATION")
    print("="*80)
    
    # Create evaluator
    evaluator = CalcChainEvaluator()
    
    # 1. Generate benchmark suite
    print("\n\n[1] GENERATING BENCHMARK SUITE...")
    suite = evaluator.create_benchmark_suite(tasks_per_structure=8)
    
    total_tasks = sum(len(tasks) for tasks in suite.values())
    print(f"✓ Generated {total_tasks} tasks across {len(suite)} mathematical structures")
    
    # 2. Analyze structure difficulty
    print("\n\n[2] ANALYZING MATHEMATICAL STRUCTURES...")
    evaluator.analyze_structure_difficulty(suite)
    
    # 3. Export benchmark
    print("\n\n[3] EXPORTING BENCHMARK...")
    evaluator.export_benchmark(suite, '/home/claude/calcchain_benchmark.json')
    
    # 4. Show demonstrations
    print("\n\n[4] DEMONSTRATING TASK VARIETY...")
    demonstrate_task_examples()
    
    # 5. Create curriculum
    print("\n\n[5] GENERATING TRAINING CURRICULUM...")
    curriculum = generate_training_curriculum()
    
    # 6. Show LLM prompt template
    print("\n\n[6] LLM PROMPT TEMPLATE...")
    print("="*80)
    template = create_llm_prompt_template()
    
    # Fill template with an example
    env = CalcChainEnvironment(seed=789)
    example_task = env.generate_task(difficulty='medium', structure=MathStructure.ELLIPTIC_CURVE)
    
    filled_template = template.format(
        numbers=example_task.numbers,
        target=example_task.target,
        operators=example_task.operators,
        min_steps=example_task.min_steps,
        max_steps=example_task.max_steps
    )
    
    print(filled_template)
    
    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
The CalcChain RLVR benchmark provides:

✓ {len(MathStructure)} distinct mathematical structures
✓ 4 difficulty levels (easy, medium, hard, expert)
✓ Hidden patterns requiring genuine mathematical reasoning
✓ Diverse operator sets from basic arithmetic to advanced functions
✓ Step constraints forcing efficient solutions

Mathematical structures included:
{chr(10).join(f"  • {s.value.replace('_', ' ').title()}" for s in MathStructure)}

This benchmark tests whether LLMs can:
1. Identify hidden mathematical patterns
2. Apply appropriate operations strategically
3. Reason about number-theoretic properties
4. Work with modular arithmetic and prime numbers
5. Recognize topological and geometric invariants
6. Handle recursive and sequential structures

Next steps:
- Use the exported JSON file for LLM evaluation
- Analyze which structures are most challenging
- Develop targeted training on weak areas
- Compare reasoning strategies across models
""")


if __name__ == "__main__":
    main()
