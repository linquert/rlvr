"""
CalcChain PRACTICAL - RL-Friendly Arithmetic Reasoning Benchmark

This is a practical benchmark for RL agents learning multi-step arithmetic reasoning.
Focus: Simple patterns that agents can learn, not obscure mathematical theorems.

Key principles:
- Targets are reachable with provided numbers and operators
- Patterns are learnable through experience
- Difficulty scales smoothly
- Solutions are verifiable
"""

import random
import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    """Types of arithmetic patterns (simple and learnable)"""
    BASIC_ARITHMETIC = "basic_arithmetic"           # (a + b) * c
    COMPOUND_OPERATIONS = "compound_operations"     # a * b + c * d
    NESTED_OPERATIONS = "nested_operations"         # (a + b) * (c - d)
    MODULAR_ARITHMETIC = "modular_arithmetic"       # (a * b) % m
    POWER_COMBINATIONS = "power_combinations"       # a^b + c
    AVERAGE_AND_SUM = "average_and_sum"            # sum/count, mean*variance
    DIGIT_OPERATIONS = "digit_operations"           # sum of digits, digit reversal
    FACTORIAL_COMBO = "factorial_combo"             # n! + m or n! % m
    SEQUENTIAL_APPLY = "sequential_apply"           # ((a op b) op c) op d
    TARGET_DECOMPOSITION = "target_decomposition"   # Break target into parts


@dataclass
class CalcTask:
    """A single arithmetic reasoning task"""
    numbers: List[int]          # Available numbers
    operators: List[str]        # Available operators
    target: int                 # Target to reach
    max_steps: int              # Max operations allowed
    min_steps: int              # Min operations required
    pattern_type: PatternType   # Pattern used (for analysis)
    difficulty: str             # easy, medium, hard
    hint: str                   # Optional hint for the pattern
    ground_truth_steps: List[str]  # One valid solution path


class PracticalCalcChain:
    """
    Practical CalcChain environment for RL training.
    
    Philosophy: Agents should learn arithmetic reasoning through patterns,
    not memorize obscure mathematical facts.
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.operator_arity = {
            '+': 2, '-': 2, '*': 2, '//': 2, '%': 2, '**': 2,
            'abs': 1, 'neg': 1
        }
    
    # =================================================================
    # PATTERN 1: BASIC ARITHMETIC COMBINATIONS
    # =================================================================
    
    def _basic_arithmetic(self, difficulty: str) -> CalcTask:
        """
        Simple combinations: (a op b) op c
        Example: (5 + 3) * 2 = 16
        """
        if difficulty == 'easy':
            a, b, c = random.randint(1, 10), random.randint(1, 10), random.randint(2, 5)
            ops = ['+', '-', '*']
        elif difficulty == 'medium':
            a, b, c = random.randint(5, 20), random.randint(5, 20), random.randint(2, 10)
            ops = ['+', '-', '*', '//']
        else:  # hard
            a, b, c = random.randint(10, 50), random.randint(10, 50), random.randint(2, 15)
            ops = ['+', '-', '*', '//', '%']
        
        # Generate target: (a + b) * c
        target = (a + b) * c
        
        return CalcTask(
            numbers=[a, b, c, 1, 0],
            operators=ops,
            target=target,
            max_steps=3,
            min_steps=2,
            pattern_type=PatternType.BASIC_ARITHMETIC,
            difficulty=difficulty,
            hint="Try combining numbers step by step",
            ground_truth_steps=[
                f"{a} + {b} = {a+b}",
                f"{a+b} * {c} = {target}"
            ]
        )
    
    # =================================================================
    # PATTERN 2: COMPOUND OPERATIONS (Parallel then Combine)
    # =================================================================
    
    def _compound_operations(self, difficulty: str) -> CalcTask:
        """
        Parallel operations: a*b + c*d
        Example: 3*4 + 5*2 = 12 + 10 = 22
        """
        if difficulty == 'easy':
            a, b = random.randint(2, 5), random.randint(2, 5)
            c, d = random.randint(2, 5), random.randint(2, 5)
        elif difficulty == 'medium':
            a, b = random.randint(3, 10), random.randint(3, 10)
            c, d = random.randint(3, 10), random.randint(3, 10)
        else:
            a, b = random.randint(5, 15), random.randint(5, 15)
            c, d = random.randint(5, 15), random.randint(5, 15)
        
        target = a * b + c * d
        
        return CalcTask(
            numbers=[a, b, c, d],
            operators=['+', '-', '*'],
            target=target,
            max_steps=4,
            min_steps=3,
            pattern_type=PatternType.COMPOUND_OPERATIONS,
            difficulty=difficulty,
            hint="Compute two products, then combine them",
            ground_truth_steps=[
                f"{a} * {b} = {a*b}",
                f"{c} * {d} = {c*d}",
                f"{a*b} + {c*d} = {target}"
            ]
        )
    
    # =================================================================
    # PATTERN 3: NESTED OPERATIONS
    # =================================================================
    
    def _nested_operations(self, difficulty: str) -> CalcTask:
        """
        Nested: (a + b) * (c - d)
        Example: (7 + 3) * (8 - 2) = 10 * 6 = 60
        """
        if difficulty == 'easy':
            a, b = random.randint(2, 8), random.randint(2, 8)
            c, d = random.randint(5, 10), random.randint(1, 4)
        elif difficulty == 'medium':
            a, b = random.randint(5, 15), random.randint(5, 15)
            c, d = random.randint(10, 20), random.randint(2, 8)
        else:
            a, b = random.randint(10, 30), random.randint(10, 30)
            c, d = random.randint(15, 40), random.randint(5, 15)
        
        target = (a + b) * (c - d)
        
        return CalcTask(
            numbers=[a, b, c, d],
            operators=['+', '-', '*'],
            target=target,
            max_steps=4,
            min_steps=3,
            pattern_type=PatternType.NESTED_OPERATIONS,
            difficulty=difficulty,
            hint="Create two intermediate results, then combine",
            ground_truth_steps=[
                f"{a} + {b} = {a+b}",
                f"{c} - {d} = {c-d}",
                f"{a+b} * {c-d} = {target}"
            ]
        )
    
    # =================================================================
    # PATTERN 4: MODULAR ARITHMETIC (Practical)
    # =================================================================
    
    def _modular_arithmetic(self, difficulty: str) -> CalcTask:
        """
        Modular: (a * b) % m or (a + b + c) % m
        Example: (17 * 5) % 13 = 85 % 13 = 7
        """
        if difficulty == 'easy':
            a, b = random.randint(2, 10), random.randint(2, 10)
            m = random.choice([7, 11, 13])
        elif difficulty == 'medium':
            a, b = random.randint(5, 20), random.randint(5, 20)
            m = random.choice([11, 13, 17, 19])
        else:
            a, b = random.randint(10, 40), random.randint(10, 40)
            m = random.choice([17, 19, 23, 29])
        
        target = (a * b) % m
        
        return CalcTask(
            numbers=[a, b, m],
            operators=['*', '%'],
            target=target,
            max_steps=3,
            min_steps=2,
            pattern_type=PatternType.MODULAR_ARITHMETIC,
            difficulty=difficulty,
            hint="Multiply first, then take modulo",
            ground_truth_steps=[
                f"{a} * {b} = {a*b}",
                f"{a*b} % {m} = {target}"
            ]
        )
    
    # =================================================================
    # PATTERN 5: POWER COMBINATIONS
    # =================================================================
    
    def _power_combinations(self, difficulty: str) -> CalcTask:
        """
        Powers: a^b + c or a^b - c
        Example: 2^5 + 7 = 32 + 7 = 39
        """
        if difficulty == 'easy':
            a, b = random.randint(2, 4), random.randint(2, 4)
            c = random.randint(1, 10)
        elif difficulty == 'medium':
            a, b = random.randint(2, 6), random.randint(2, 5)
            c = random.randint(5, 20)
        else:
            a, b = random.randint(3, 8), random.randint(2, 5)
            c = random.randint(10, 50)
        
        power = a ** b
        target = power + c
        
        return CalcTask(
            numbers=[a, b, c],
            operators=['**', '+', '-'],
            target=target,
            max_steps=3,
            min_steps=2,
            pattern_type=PatternType.POWER_COMBINATIONS,
            difficulty=difficulty,
            hint="Calculate power first, then add/subtract",
            ground_truth_steps=[
                f"{a} ** {b} = {power}",
                f"{power} + {c} = {target}"
            ]
        )
    
    # =================================================================
    # PATTERN 6: AVERAGE AND SUM PATTERNS
    # =================================================================
    
    def _average_and_sum(self, difficulty: str) -> CalcTask:
        """
        Statistical: (a + b + c) // 3 * d or similar
        Example: Sum of numbers, then multiply by count
        """
        if difficulty == 'easy':
            nums = [random.randint(2, 10) for _ in range(3)]
            multiplier = random.randint(2, 5)
        elif difficulty == 'medium':
            nums = [random.randint(5, 20) for _ in range(4)]
            multiplier = random.randint(2, 8)
        else:
            nums = [random.randint(10, 30) for _ in range(5)]
            multiplier = random.randint(3, 10)
        
        total = sum(nums)
        target = (total // len(nums)) * multiplier
        
        steps = [f"{nums[i]} + {nums[i+1]} = {nums[i] + nums[i+1]}" for i in range(len(nums)-1)]
        steps.append(f"{total} // {len(nums)} = {total // len(nums)}")
        steps.append(f"{total // len(nums)} * {multiplier} = {target}")
        
        return CalcTask(
            numbers=nums + [multiplier, len(nums)],
            operators=['+', '//', '*'],
            target=target,
            max_steps=len(nums) + 2,
            min_steps=len(nums),
            pattern_type=PatternType.AVERAGE_AND_SUM,
            difficulty=difficulty,
            hint="Find average (sum/count), then scale",
            ground_truth_steps=steps
        )
    
    # =================================================================
    # PATTERN 7: DIGIT OPERATIONS
    # =================================================================
    
    def _digit_operations(self, difficulty: str) -> CalcTask:
        """
        Digit-based: sum of digits, product of digits
        Example: 123 -> 1+2+3 = 6, then 6*5 = 30
        """
        if difficulty == 'easy':
            num = random.randint(10, 99)  # 2 digits
            multiplier = random.randint(2, 5)
        elif difficulty == 'medium':
            num = random.randint(100, 999)  # 3 digits
            multiplier = random.randint(3, 8)
        else:
            num = random.randint(1000, 9999)  # 4 digits
            multiplier = random.randint(5, 12)
        
        digits = [int(d) for d in str(num)]
        digit_sum = sum(digits)
        target = digit_sum * multiplier
        
        return CalcTask(
            numbers=digits + [multiplier, 10],
            operators=['+', '*', '//', '%'],
            target=target,
            max_steps=len(digits) + 2,
            min_steps=len(digits),
            pattern_type=PatternType.DIGIT_OPERATIONS,
            difficulty=difficulty,
            hint="Extract digits, sum them, then multiply",
            ground_truth_steps=[
                f"Digits of {num}: {digits}",
                f"Sum: {digit_sum}",
                f"{digit_sum} * {multiplier} = {target}"
            ]
        )
    
    # =================================================================
    # PATTERN 8: FACTORIAL COMBINATIONS (Small n only)
    # =================================================================
    
    def _factorial_combo(self, difficulty: str) -> CalcTask:
        """
        Small factorials: n! + m or n! % m
        Example: 5! + 8 = 120 + 8 = 128
        """
        if difficulty == 'easy':
            n = random.randint(3, 5)
            m = random.randint(5, 20)
        elif difficulty == 'medium':
            n = random.randint(4, 6)
            m = random.randint(10, 50)
        else:
            n = random.randint(5, 7)
            m = random.randint(20, 100)
        
        factorial = math.factorial(n)
        target = factorial + m
        
        # Provide numbers to build factorial
        numbers = list(range(1, n+1)) + [m]
        
        return CalcTask(
            numbers=numbers,
            operators=['*', '+'],
            target=target,
            max_steps=n + 1,
            min_steps=n,
            pattern_type=PatternType.FACTORIAL_COMBO,
            difficulty=difficulty,
            hint="Multiply consecutive numbers, then add",
            ground_truth_steps=[
                f"{n}! = {factorial}",
                f"{factorial} + {m} = {target}"
            ]
        )
    
    # =================================================================
    # PATTERN 9: SEQUENTIAL APPLICATION
    # =================================================================
    
    def _sequential_apply(self, difficulty: str) -> CalcTask:
        """
        Chain operations: ((a + b) * c) - d
        Example: ((5 + 3) * 2) - 4 = (8 * 2) - 4 = 16 - 4 = 12
        """
        if difficulty == 'easy':
            a, b, c, d = [random.randint(2, 10) for _ in range(4)]
        elif difficulty == 'medium':
            a, b, c, d = [random.randint(5, 20) for _ in range(4)]
        else:
            a, b, c, d = [random.randint(10, 40) for _ in range(4)]
        
        step1 = a + b
        step2 = step1 * c
        target = step2 - d
        
        return CalcTask(
            numbers=[a, b, c, d],
            operators=['+', '*', '-'],
            target=target,
            max_steps=4,
            min_steps=3,
            pattern_type=PatternType.SEQUENTIAL_APPLY,
            difficulty=difficulty,
            hint="Apply operations left to right sequentially",
            ground_truth_steps=[
                f"{a} + {b} = {step1}",
                f"{step1} * {c} = {step2}",
                f"{step2} - {d} = {target}"
            ]
        )
    
    # =================================================================
    # PATTERN 10: TARGET DECOMPOSITION
    # =================================================================
    
    def _target_decomposition(self, difficulty: str) -> CalcTask:
        """
        Reverse engineering: Given target, find how to build it
        Example: Target 100 from [25, 4, 10, 2] -> 25*4 or 10*10
        """
        if difficulty == 'easy':
            # Easy: target = a * b exactly
            a, b = random.randint(5, 15), random.randint(2, 8)
            target = a * b
            numbers = [a, b, random.randint(1, 5), random.randint(1, 5)]
        elif difficulty == 'medium':
            # Medium: target = a*b + c
            a, b = random.randint(5, 15), random.randint(3, 10)
            c = random.randint(1, 20)
            target = a * b + c
            numbers = [a, b, c, random.randint(1, 5)]
        else:
            # Hard: target = (a+b)*(c-d) or similar
            a, b = random.randint(5, 20), random.randint(5, 20)
            c, d = random.randint(10, 30), random.randint(2, 10)
            target = (a + b) * (c - d)
            numbers = [a, b, c, d]
        
        return CalcTask(
            numbers=numbers,
            operators=['+', '-', '*', '//', '%'],
            target=target,
            max_steps=4,
            min_steps=1,
            pattern_type=PatternType.TARGET_DECOMPOSITION,
            difficulty=difficulty,
            hint="Think about how to factor or decompose the target",
            ground_truth_steps=[
                f"Find operations that produce {target}"
            ]
        )
    
    # =================================================================
    # MAIN INTERFACE
    # =================================================================
    
    def generate_task(self, difficulty: str = 'medium', pattern: Optional[PatternType] = None) -> CalcTask:
        """Generate a task with specified difficulty and optional pattern type."""
        
        if pattern is None:
            # Random pattern
            pattern = random.choice(list(PatternType))
        
        generators = {
            PatternType.BASIC_ARITHMETIC: self._basic_arithmetic,
            PatternType.COMPOUND_OPERATIONS: self._compound_operations,
            PatternType.NESTED_OPERATIONS: self._nested_operations,
            PatternType.MODULAR_ARITHMETIC: self._modular_arithmetic,
            PatternType.POWER_COMBINATIONS: self._power_combinations,
            PatternType.AVERAGE_AND_SUM: self._average_and_sum,
            PatternType.DIGIT_OPERATIONS: self._digit_operations,
            PatternType.FACTORIAL_COMBO: self._factorial_combo,
            PatternType.SEQUENTIAL_APPLY: self._sequential_apply,
            PatternType.TARGET_DECOMPOSITION: self._target_decomposition,
        }
        
        return generators[pattern](difficulty)
    
    def generate_dataset(self, num_tasks: int = 100, 
                        difficulty_dist: Optional[Dict[str, float]] = None) -> List[CalcTask]:
        """Generate a balanced dataset."""
        
        if difficulty_dist is None:
            difficulty_dist = {'easy': 0.3, 'medium': 0.4, 'hard': 0.3}
        
        tasks = []
        for _ in range(num_tasks):
            difficulty = random.choices(
                list(difficulty_dist.keys()),
                weights=list(difficulty_dist.values())
            )[0]
            task = self.generate_task(difficulty=difficulty)
            tasks.append(task)
        
        return tasks
    
    def verify_solution(self, task: CalcTask, operations: List[str]) -> Dict[str, Any]:
        """
        Verify if a sequence of operations solves the task.
        
        Args:
            task: The CalcTask
            operations: List of operation strings like "5 + 3 = 8"
        
        Returns:
            Dict with verification results
        """
        if len(operations) > task.max_steps:
            return {'success': False, 'reason': 'Too many steps'}
        
        if len(operations) < task.min_steps:
            return {'success': False, 'reason': 'Too few steps'}
        
        available = task.numbers.copy()

        try:
            for step in operations:
                parsed = self._parse_operation(step)
                if not parsed:
                    return {'success': False, 'reason': f'Could not parse step: {step}'}

                op, operands, expected_result = parsed
                if op not in task.operators:
                    return {'success': False, 'reason': f'Operator not allowed: {op}'}

                expected_arity = self.operator_arity.get(op)
                if expected_arity is None:
                    return {'success': False, 'reason': f'Unknown operator: {op}'}
                if len(operands) != expected_arity:
                    return {
                        'success': False,
                        'reason': f'Operator {op} expects {expected_arity} operands, got {len(operands)}'
                    }

                # Ensure each operand is currently available and consume it.
                consumed_indices = []
                for operand in operands:
                    found_idx = None
                    for idx, value in enumerate(available):
                        if idx in consumed_indices:
                            continue
                        if value == operand:
                            found_idx = idx
                            break
                    if found_idx is None:
                        return {'success': False, 'reason': f'Operand not available: {operand}'}
                    consumed_indices.append(found_idx)

                actual_result = self._apply_operation(op, operands)
                if actual_result != expected_result:
                    return {
                        'success': False,
                        'reason': f'Incorrect step result for "{step}": expected {actual_result}, got {expected_result}'
                    }

                for idx in sorted(consumed_indices, reverse=True):
                    available.pop(idx)
                available.append(actual_result)

            final_value = available[-1] if available else None
            return {
                'success': final_value == task.target,
                'final_value': final_value,
                'target': task.target,
                'steps_used': len(operations)
            }
        except Exception as e:
            return {'success': False, 'reason': str(e)}

    def _parse_operation(self, step: str):
        """Parse operation strings like '5 + 3 = 8' or 'abs -2 = 2'."""
        if '=' not in step:
            return None

        left, right = step.split('=', 1)
        left = left.strip()
        right = right.strip()

        try:
            expected_result = int(right)
        except ValueError:
            return None

        tokens = left.split()
        if len(tokens) == 3:
            # Binary: a op b
            try:
                a = int(tokens[0])
                op = tokens[1]
                b = int(tokens[2])
            except ValueError:
                return None
            return op, [a, b], expected_result

        if len(tokens) == 2:
            # Unary: op a
            op = tokens[0]
            try:
                a = int(tokens[1])
            except ValueError:
                return None
            return op, [a], expected_result

        return None

    def _apply_operation(self, op: str, operands: List[int]) -> int:
        """Apply an operation to parsed operands."""
        if op == '+':
            return operands[0] + operands[1]
        if op == '-':
            return operands[0] - operands[1]
        if op == '*':
            return operands[0] * operands[1]
        if op == '//':
            if operands[1] == 0:
                raise ValueError("Division by zero")
            return operands[0] // operands[1]
        if op == '%':
            if operands[1] == 0:
                raise ValueError("Modulo by zero")
            return operands[0] % operands[1]
        if op == '**':
            if operands[1] < 0 or operands[1] > 20:
                raise ValueError("Exponent out of allowed range [0, 20]")
            return operands[0] ** operands[1]
        if op == 'abs':
            return abs(operands[0])
        if op == 'neg':
            return -operands[0]
        raise ValueError(f"Unknown operator: {op}")
    
    def format_task_for_llm(self, task: CalcTask) -> str:
        """Format task as a prompt for LLM."""
        
        prompt = f"""CalcChain Task (Difficulty: {task.difficulty})

Target: {task.target}
Numbers available: {task.numbers}
Operators allowed: {task.operators}
Steps: {task.min_steps}-{task.max_steps}

Hint: {task.hint}

Find a sequence of operations that reaches the target exactly.

Example format:
Step 1: 5 + 3 = 8
Step 2: 8 * 2 = 16
...
Final: [target]
"""
        return prompt


def main():
    """Demonstrate the practical benchmark."""
    
    env = PracticalCalcChain(seed=42)
    
    print("="*80)
    print("CALCCHAIN PRACTICAL - RL-Friendly Arithmetic Benchmark")
    print("="*80)
    print()
    
    # Show one example from each pattern type
    print("Sample tasks from each pattern:\n")
    
    for pattern in list(PatternType)[:5]:  # Show first 5
        task = env.generate_task(difficulty='medium', pattern=pattern)
        
        print(f"{'─'*80}")
        print(f"Pattern: {pattern.value.replace('_', ' ').title()}")
        print(f"{'─'*80}")
        print(f"Target: {task.target}")
        print(f"Numbers: {task.numbers}")
        print(f"Operators: {task.operators}")
        print(f"Steps: {task.min_steps}-{task.max_steps}")
        print(f"\nHint: {task.hint}")
        print(f"\nGround truth solution:")
        for i, step in enumerate(task.ground_truth_steps, 1):
            print(f"  Step {i}: {step}")
        print()
    
    # Generate dataset
    print(f"\n{'='*80}")
    print("Generating dataset...")
    print(f"{'='*80}\n")
    
    dataset = env.generate_dataset(num_tasks=100)
    
    # Statistics
    from collections import Counter
    
    pattern_counts = Counter(task.pattern_type.value for task in dataset)
    difficulty_counts = Counter(task.difficulty for task in dataset)
    
    print(f"Generated {len(dataset)} tasks")
    print(f"\nPattern distribution:")
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern.replace('_', ' ').title()}: {count}")
    
    print(f"\nDifficulty distribution:")
    for diff in ['easy', 'medium', 'hard']:
        if diff in difficulty_counts:
            print(f"  {diff.title()}: {difficulty_counts[diff]}")


if __name__ == "__main__":
    main()
