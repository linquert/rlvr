"""
CalcChain RLVR Environment - Advanced Mathematical Reasoning Benchmark

This environment generates arithmetic reasoning tasks where models must reach target values
by synthesizing multi-step operations. The targets are generated using hidden mathematical
structures that the model must discover through reasoning.

Hidden Mathematical Structures Used:
1. Elliptic Curves: Points on curves like y² = x³ + ax + b
2. Modular Arithmetic: Patterns in mod n spaces
3. Fibonacci-like Recurrences: Hidden recursive patterns
4. Prime Constellations: Sequences related to prime gaps
5. Continued Fractions: Hidden rational approximations
6. Topological Invariants: Euler characteristic computations
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import math
# put near top of class as a static map or instance attribute
_operator_meta = {
    '+': {'arity': 2},
    '-': {'arity': 2},
    '*': {'arity': 2},
    '//': {'arity': 2},
    '%': {'arity': 2},
    '**': {'arity': 2},
    'abs': {'arity': 1},
    'neg': {'arity': 1},
    'sqrt': {'arity': 1},
    'log2': {'arity': 1},
    'gcd': {'arity': 2},
    'lcm': {'arity': 2},
    'floor': {'arity': 1},
    'ceil': {'arity': 1},
    'factorial_mod': {'arity': 2},
    'mod_inverse': {'arity': 2},
}

def _apply_operation(self, op: str, operands: List[int]) -> int:
    """Apply an operation to operands with robust checks."""
    # verify operator exists
    if op not in self._operator_meta:
        raise ValueError(f"Unknown operator: {op}")

    arity = self._operator_meta[op]['arity']
    if len(operands) != arity:
        raise ValueError(f"Operator {op} expects {arity} operands, got {len(operands)}")

    # now safe implementations
    if op == '+':
        return operands[0] + operands[1]
    if op == '-':
        return operands[0] - operands[1]
    if op == '*':
        return operands[0] * operands[1]
    if op == '//':
        if operands[1] == 0:
            raise ZeroDivisionError("Integer division by zero")
        return operands[0] // operands[1]
    if op == '%':
        if operands[1] == 0:
            raise ZeroDivisionError("Modulo by zero")
        return operands[0] % operands[1]
    if op == '**':
        base, exp = operands[0], operands[1]
        # disallow negative exponents (would produce float) and very large ones
        if exp < 0:
            raise ValueError("Negative exponent not allowed")
        if exp > 30 or abs(base) > 10**6:
            raise ValueError("Exponent/base too large - risk of overflow")
        return pow(base, exp)
    if op == 'abs':
        return abs(operands[0])
    if op == 'neg':
        return -operands[0]
    if op == 'sqrt':
        if operands[0] < 0:
            raise ValueError("sqrt of negative number")
        return math.isqrt(operands[0])   # exact integer sqrt floor
    if op == 'log2':
        if operands[0] <= 0:
            raise ValueError("log2 input must be positive")
        return operands[0].bit_length() - 1  # floor(log2(n))
    if op == 'gcd':
        return math.gcd(operands[0], operands[1])
    if op == 'lcm':
        a, b = operands[0], operands[1]
        g = math.gcd(a, b)
        return 0 if g == 0 else abs(a // g * b)
    if op == 'floor':
        return math.floor(operands[0])
    if op == 'ceil':
        return math.ceil(operands[0])
    if op == 'mod_inverse':
        a, m = operands[0], operands[1]
        if m <= 1:
            raise ValueError("modulus must be > 1")
        # using pow for modular inverse when gcd(a,m)==1
        g = math.gcd(a, m)
        if g != 1:
            raise ValueError("mod inverse does not exist (a not coprime with m)")
        return pow(a, -1, m)
    if op == 'factorial_mod':
        n, m = operands[0], operands[1]
        if n < 0:
            raise ValueError("factorial of negative")
        if n > 1000:
            raise ValueError("factorial argument too large")
        return math.factorial(n) % m
    # safety fallback
    raise ValueError(f"Operator {op} not implemented")


class MathStructure(Enum):
    """Types of hidden mathematical structures"""
    ELLIPTIC_CURVE = "elliptic_curve"
    MODULAR_SEQUENCE = "modular_sequence"
    FIBONACCI_VARIANT = "fibonacci_variant"
    PRIME_CONSTELLATION = "prime_constellation"
    CONTINUED_FRACTION = "continued_fraction"
    TOPOLOGICAL_INVARIANT = "topological_invariant"
    FERMAT_LITTLE = "fermat_little_theorem"
    QUADRATIC_RESIDUE = "quadratic_residue"


@dataclass
class CalcChainTask:
    """A single CalcChain task"""
    numbers: List[int]  # Pool of integers to use
    operators: List[str]  # Available operators
    target: int  # Target value to reach
    max_steps: int  # Maximum number of operations allowed
    min_steps: int  # Minimum number of operations required
    hidden_structure: MathStructure  # The mathematical structure used (for analysis)
    structure_params: Dict[str, Any]  # Parameters of the hidden structure
    difficulty: str  # easy, medium, hard, expert
    

class CalcChainEnvironment:
    """
    RLVR Environment for CalcChain arithmetic reasoning tasks.
    
    The environment generates targets using hidden mathematical structures,
    then provides a pool of numbers and operators that allow reaching the target.
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Available operator sets by difficulty
        self.operator_sets = {
            'basic': ['+', '-', '*', '//', '%'],
            'intermediate': ['+', '-', '*', '//', '%', '**', 'abs', 'neg'],
            'advanced': ['+', '-', '*', '//', '%', '**', 'abs', 'neg', 'sqrt', 'log2', 'gcd', 'lcm'],
            'expert': ['+', '-', '*', '//', '%', '**', 'abs', 'neg', 'sqrt', 'log2', 'gcd', 'lcm', 
                      'floor', 'ceil', 'factorial_mod', 'mod_inverse']
        }
    
    # ===========================================
    # MATHEMATICAL STRUCTURE GENERATORS
    # ===========================================
    
    def _generate_elliptic_curve_point(self, a: int, b: int, p: int) -> Tuple[int, int]:
        """
        Generate a point on an elliptic curve y² ≡ x³ + ax + b (mod p)
        Returns coordinates that when combined reach the target
        """
        for x in range(p):
            y_squared = (x**3 + a*x + b) % p
            # Check if y_squared is a quadratic residue
            for y in range(p):
                if (y*y) % p == y_squared:
                    return (x, y)
        return (0, 0)
    
    def _elliptic_curve_task(self, difficulty: str) -> CalcChainTask:
        """
        Create task based on elliptic curve arithmetic.
        Target = x³ + ax + b (mod p) for some hidden curve parameters.
        """
        if difficulty == 'easy':
            p = random.choice([11, 13, 17, 19, 23])
            a, b = random.randint(1, 5), random.randint(1, 5)
            x = random.randint(1, 10)
            max_steps = 6
        elif difficulty == 'medium':
            p = random.choice([29, 31, 37, 41, 43])
            a, b = random.randint(1, 10), random.randint(1, 10)
            x = random.randint(5, 20)
            max_steps = 8
        else:  # hard or expert
            p = random.choice([47, 53, 59, 61, 67])
            a, b = random.randint(5, 15), random.randint(5, 15)
            x = random.randint(10, 40)
            max_steps = 10
        
        # Target: (x³ + ax + b) mod p
        target = (x**3 + a*x + b) % p
        
        # Provide numbers that hint at the structure without revealing it
        numbers = [x, a, b, p, 3, 2, 1]
        random.shuffle(numbers)
        
        operators = self.operator_sets['intermediate' if difficulty in ['easy', 'medium'] else 'advanced']
        
        return CalcChainTask(
            numbers=numbers,
            operators=operators,
            target=target,
            max_steps=max_steps,
            min_steps=3,
            hidden_structure=MathStructure.ELLIPTIC_CURVE,
            structure_params={'a': a, 'b': b, 'p': p, 'x': x},
            difficulty=difficulty
        )
    
    def _modular_sequence_task(self, difficulty: str) -> CalcChainTask:
        """
        Create task based on modular arithmetic sequences.
        Target follows pattern: a*n^k + b*n^(k-1) + ... (mod m)
        """
        if difficulty == 'easy':
            n, k, m = random.randint(2, 5), 2, random.choice([7, 11, 13])
            coeffs = [random.randint(1, 3) for _ in range(k+1)]
            max_steps = 5
        elif difficulty == 'medium':
            n, k, m = random.randint(3, 8), 3, random.choice([17, 19, 23])
            coeffs = [random.randint(1, 5) for _ in range(k+1)]
            max_steps = 7
        else:
            n, k, m = random.randint(5, 12), 4, random.choice([29, 31, 37])
            coeffs = [random.randint(2, 8) for _ in range(k+1)]
            max_steps = 9
        
        # Target: polynomial evaluation mod m
        target = sum(c * (n**(k-i)) for i, c in enumerate(coeffs)) % m
        
        numbers = [n, m, k] + coeffs + [1, 0]
        random.shuffle(numbers)
        
        operators = self.operator_sets['intermediate' if difficulty == 'easy' else 'advanced']
        
        return CalcChainTask(
            numbers=numbers,
            operators=operators,
            target=target,
            max_steps=max_steps,
            min_steps=3,
            hidden_structure=MathStructure.MODULAR_SEQUENCE,
            structure_params={'n': n, 'k': k, 'm': m, 'coefficients': coeffs},
            difficulty=difficulty
        )
    
    def _fibonacci_variant_task(self, difficulty: str) -> CalcChainTask:
        """
        Create task based on generalized Fibonacci recurrences.
        F(n) = a*F(n-1) + b*F(n-2) + c
        """
        if difficulty == 'easy':
            a, b, c = random.randint(1, 2), random.randint(1, 2), random.randint(0, 2)
            f0, f1 = random.randint(0, 3), random.randint(1, 4)
            n = random.randint(4, 6)
            max_steps = 6
        elif difficulty == 'medium':
            a, b, c = random.randint(1, 3), random.randint(1, 3), random.randint(0, 5)
            f0, f1 = random.randint(1, 5), random.randint(2, 6)
            n = random.randint(5, 8)
            max_steps = 8
        else:
            a, b, c = random.randint(2, 5), random.randint(2, 5), random.randint(1, 8)
            f0, f1 = random.randint(2, 7), random.randint(3, 9)
            n = random.randint(6, 10)
            max_steps = 10
        
        # Compute F(n) using the recurrence
        fib = [f0, f1]
        for i in range(2, n+1):
            fib.append(a * fib[i-1] + b * fib[i-2] + c)
        
        target = fib[n]
        
        numbers = [a, b, c, f0, f1, n] + list(range(1, 4))
        random.shuffle(numbers)
        
        operators = self.operator_sets['intermediate' if difficulty == 'easy' else 'advanced']
        
        return CalcChainTask(
            numbers=numbers,
            operators=operators,
            target=target,
            max_steps=max_steps,
            min_steps=n,
            hidden_structure=MathStructure.FIBONACCI_VARIANT,
            structure_params={'a': a, 'b': b, 'c': c, 'f0': f0, 'f1': f1, 'n': n},
            difficulty=difficulty
        )
    
    def _prime_constellation_task(self, difficulty: str) -> CalcChainTask:
        """
        Create task based on prime gaps and constellations.
        Target relates to prime gap patterns or Sophie Germain primes.
        """
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        
        if difficulty == 'easy':
            idx = random.randint(2, 8)
            p1, p2, p3 = primes[idx], primes[idx+1], primes[idx+2]
            # Target: gap pattern
            target = (p2 - p1) * (p3 - p2) + p1
            max_steps = 5
            p4 = None
        elif difficulty == 'medium':
            idx = random.randint(3, 12)
            p1, p2, p3 = primes[idx], primes[idx+1], primes[idx+2]
            # Sophie Germain: 2p+1
            target = 2 * p1 + 1
            if target in primes:
                target = target + (p2 - p1)
            max_steps = 7
            p4 = None
        else:
            idx = random.randint(5, 18)
            p1, p2, p3, p4 = primes[idx], primes[idx+1], primes[idx+2], primes[idx+3]
            # Complex constellation pattern
            target = ((p2 - p1) ** 2 + (p3 - p2) ** 2) % (p4 - p1)
            max_steps = 9
        
        # Provide primes and small numbers
        numbers = [p1, p2, p3, 2, 1] + \
                  ([p4] if difficulty in ['hard', 'expert'] and p4 is not None else []) + \
                  list(range(1, 4))
        random.shuffle(numbers)
        
        operators = self.operator_sets['basic' if difficulty == 'easy' else 'intermediate']
        
        return CalcChainTask(
            numbers=numbers,
            operators=operators,
            target=target,
            max_steps=max_steps,
            min_steps=3,
            hidden_structure=MathStructure.PRIME_CONSTELLATION,
            structure_params={'primes_used': [p1, p2, p3] + ([p4] if p4 is not None else [])},
            difficulty=difficulty
        )
    
    def _continued_fraction_task(self, difficulty: str) -> CalcChainTask:
        """
        Create task based on continued fraction convergents.
        Target is a convergent of a continued fraction representation.
        """
        if difficulty == 'easy':
            # Simple continued fraction [a0; a1, a2]
            a = [random.randint(1, 5), random.randint(1, 4), random.randint(1, 4)]
            max_steps = 5
        elif difficulty == 'medium':
            a = [random.randint(2, 8)] + [random.randint(1, 6) for _ in range(3)]
            max_steps = 7
        else:
            a = [random.randint(3, 12)] + [random.randint(1, 8) for _ in range(4)]
            max_steps = 9
        
        # Compute convergent p_n / q_n
        p = [a[0], a[1] * a[0] + 1]
        q = [1, a[1]]
        
        for i in range(2, len(a)):
            p.append(a[i] * p[i-1] + p[i-2])
            q.append(a[i] * q[i-1] + q[i-2])
        
        target = p[-1]  # Use numerator as target
        
        numbers = a + [1] + list(range(1, 4))
        random.shuffle(numbers)
        
        operators = self.operator_sets['intermediate']
        
        return CalcChainTask(
            numbers=numbers,
            operators=operators,
            target=target,
            max_steps=max_steps,
            min_steps=len(a),
            hidden_structure=MathStructure.CONTINUED_FRACTION,
            structure_params={'cf_coefficients': a, 'convergent_num': p[-1], 'convergent_den': q[-1]},
            difficulty=difficulty
        )
    
    def _topological_invariant_task(self, difficulty: str) -> CalcChainTask:
        """
        Create task based on topological invariants (Euler characteristic).
        χ = V - E + F for polyhedra or χ = 2 - 2g for surfaces.
        """
        if difficulty == 'easy':
            # Simple polyhedron (tetrahedron, cube, etc.)
            polyhedra = [
                {'V': 4, 'E': 6, 'F': 4, 'name': 'tetrahedron'},  # χ = 2
                {'V': 8, 'E': 12, 'F': 6, 'name': 'cube'},  # χ = 2
                {'V': 6, 'E': 12, 'F': 8, 'name': 'octahedron'},  # χ = 2
            ]
            poly = random.choice(polyhedra)
            target = poly['V'] - poly['E'] + poly['F']
            numbers = [poly['V'], poly['E'], poly['F'], 1, 2]
            max_steps = 4
        elif difficulty == 'medium':
            # Surface with genus g: χ = 2 - 2g
            g = random.randint(1, 5)
            target = 2 - 2*g
            # Also provide V, E, F that satisfy V - E + F = χ
            V = random.randint(6, 20)
            F = random.randint(4, 15)
            E = V + F - target
            numbers = [V, E, F, g, 2, 1]
            max_steps = 6
        else:
            # Complex surface structure
            g = random.randint(2, 8)
            b = random.randint(1, 4)  # boundary components
            target = 2 - 2*g - b  # χ with boundary
            V = random.randint(10, 30)
            F = random.randint(8, 25)
            E = V + F - target
            numbers = [V, E, F, g, b, 2, 1]
            max_steps = 8
        
        random.shuffle(numbers)
        operators = self.operator_sets['basic' if difficulty == 'easy' else 'intermediate']
        
        return CalcChainTask(
            numbers=numbers,
            operators=operators,
            target=target,
            max_steps=max_steps,
            min_steps=2,
            hidden_structure=MathStructure.TOPOLOGICAL_INVARIANT,
            structure_params={'euler_char': target, 'description': f'difficulty={difficulty}'},
            difficulty=difficulty
        )
    
    def _fermat_little_task(self, difficulty: str) -> CalcChainTask:
        """
        Create task based on Fermat's Little Theorem: a^(p-1) ≡ 1 (mod p)
        Target involves modular exponentiation patterns.
        """
        primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        if difficulty == 'easy':
            p = random.choice(primes[:5])
            a = random.randint(2, 5)
            k = random.randint(1, 3)
            max_steps = 5
        elif difficulty == 'medium':
            p = random.choice(primes[3:9])
            a = random.randint(2, 8)
            k = random.randint(2, 5)
            max_steps = 7
        else:
            p = random.choice(primes[6:])
            a = random.randint(3, 12)
            k = random.randint(3, 7)
            max_steps = 9
        
        # Target: a^k mod p, where k might be ≥ p-1
        target = pow(a, k * (p - 1) + random.randint(1, p-2), p)
        
        numbers = [a, p, k, p-1, 1] + list(range(2, 5))
        random.shuffle(numbers)
        
        operators = self.operator_sets['intermediate' if difficulty != 'expert' else 'advanced']
        
        return CalcChainTask(
            numbers=numbers,
            operators=operators,
            target=target,
            max_steps=max_steps,
            min_steps=3,
            hidden_structure=MathStructure.FERMAT_LITTLE,
            structure_params={'a': a, 'p': p, 'k': k},
            difficulty=difficulty
        )
    
    def _quadratic_residue_task(self, difficulty: str) -> CalcChainTask:
        """
        Create task based on quadratic residues and Legendre symbols.
        Target relates to whether a is a quadratic residue modulo p.
        """
        primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        if difficulty == 'easy':
            p = random.choice(primes[:4])
            a = random.randint(2, 10)
            max_steps = 5
        elif difficulty == 'medium':
            p = random.choice(primes[3:8])
            a = random.randint(3, 15)
            max_steps = 7
        else:
            p = random.choice(primes[6:])
            a = random.randint(5, 25)
            max_steps = 9
        
        # Compute Legendre symbol: a^((p-1)/2) mod p ∈ {-1, 0, 1}
        legendre = pow(a, (p-1)//2, p)
        if legendre == p - 1:
            legendre = -1
        
        # Target combines the residue check with arithmetic
        squares_mod_p = {(i*i) % p for i in range(p)}
        is_residue = (a % p) in squares_mod_p
        
        target = a + legendre * (p // 2) if is_residue else (a * 2) % p
        
        numbers = [a, p, (p-1)//2, 2, 1, -1] + list(range(1, 4))
        random.shuffle(numbers)
        
        operators = self.operator_sets['intermediate' if difficulty != 'expert' else 'advanced']
        
        return CalcChainTask(
            numbers=numbers,
            operators=operators,
            target=target,
            max_steps=max_steps,
            min_steps=3,
            hidden_structure=MathStructure.QUADRATIC_RESIDUE,
            structure_params={'a': a, 'p': p, 'legendre_symbol': legendre, 'is_residue': is_residue},
            difficulty=difficulty
        )
    
    # ===========================================
    # ENVIRONMENT INTERFACE
    # ===========================================
    
    def generate_task(self, difficulty: str = 'medium', structure: Optional[MathStructure] = None) -> CalcChainTask:
        """
        Generate a CalcChain task with specified difficulty.
        
        Args:
            difficulty: 'easy', 'medium', 'hard', or 'expert'
            structure: Specific mathematical structure to use (random if None)
        
        Returns:
            CalcChainTask instance
        """
        if structure is None:
            # Randomly choose a structure
            structure = random.choice(list(MathStructure))
        
        generators = {
            MathStructure.ELLIPTIC_CURVE: self._elliptic_curve_task,
            MathStructure.MODULAR_SEQUENCE: self._modular_sequence_task,
            MathStructure.FIBONACCI_VARIANT: self._fibonacci_variant_task,
            MathStructure.PRIME_CONSTELLATION: self._prime_constellation_task,
            MathStructure.CONTINUED_FRACTION: self._continued_fraction_task,
            MathStructure.TOPOLOGICAL_INVARIANT: self._topological_invariant_task,
            MathStructure.FERMAT_LITTLE: self._fermat_little_task,
            MathStructure.QUADRATIC_RESIDUE: self._quadratic_residue_task,
        }
        
        return generators[structure](difficulty)
    
    def generate_dataset(self, num_tasks: int = 100, difficulty_dist: Optional[Dict[str, float]] = None) -> List[CalcChainTask]:
        """
        Generate a dataset of CalcChain tasks.
        
        Args:
            num_tasks: Number of tasks to generate
            difficulty_dist: Distribution of difficulties {'easy': 0.2, 'medium': 0.5, ...}
        
        Returns:
            List of CalcChainTask instances
        """
        if difficulty_dist is None:
            difficulty_dist = {'easy': 0.15, 'medium': 0.35, 'hard': 0.35, 'expert': 0.15}
        
        tasks = []
        difficulties = list(difficulty_dist.keys())
        probabilities = list(difficulty_dist.values())
        
        for _ in range(num_tasks):
            difficulty = random.choices(difficulties, weights=probabilities)[0]
            task = self.generate_task(difficulty=difficulty)
            tasks.append(task)
        
        return tasks
    
    def verify_solution(self, task: CalcChainTask, operations: List[Tuple[str, List[int]]]) -> Dict[str, Any]:
        """
        Verify if a sequence of operations reaches the target.
        
        Args:
            task: The CalcChainTask
            operations: List of (operator, operand_indices) tuples
        
        Returns:
            Dictionary with 'success', 'final_value', 'steps_used', 'valid'
        """
        if len(operations) > task.max_steps:
            return {'success': False, 'valid': False, 'reason': 'Exceeded max steps'}
        
        if len(operations) < task.min_steps:
            return {'success': False, 'valid': False, 'reason': 'Below min steps'}
        
        # Track available values
        available = task.numbers.copy()
        
        try:
            for op, operand_indices in operations:
                # Get operands
                operands = [available[i] for i in operand_indices]
                
                # Apply operation
                result = self._apply_operation(op, operands)
                
                # Update available values (remove used, add result)
                for idx in sorted(operand_indices, reverse=True):
                    available.pop(idx)
                available.append(result)
            
            final_value = available[-1] if available else None
            success = final_value == task.target
            
            return {
                'success': success,
                'valid': True,
                'final_value': final_value,
                'steps_used': len(operations),
                'target': task.target
            }
        
        except Exception as e:
            return {'success': False, 'valid': False, 'reason': str(e)}
    
    def _apply_operation(self, op: str, operands: List[int]) -> int:
        """Apply an operation to operands."""
        if op == '+':
            return operands[0] + operands[1]
        elif op == '-':
            return operands[0] - operands[1]
        elif op == '*':
            return operands[0] * operands[1]
        elif op == '//':
            return operands[0] // operands[1] if operands[1] != 0 else 0
        elif op == '%':
            return operands[0] % operands[1] if operands[1] != 0 else 0
        elif op == '**':
            return operands[0] ** operands[1] if operands[1] < 100 else 0
        elif op == 'abs':
            return abs(operands[0])
        elif op == 'neg':
            return -operands[0]
        elif op == 'sqrt':
            return int(operands[0] ** 0.5)
        elif op == 'log2':
            return int(math.log2(operands[0])) if operands[0] > 0 else 0
        elif op == 'gcd':
            return math.gcd(operands[0], operands[1])
        elif op == 'lcm':
            return abs(operands[0] * operands[1]) // math.gcd(operands[0], operands[1])
        elif op == 'floor':
            return math.floor(operands[0])
        elif op == 'ceil':
            return math.ceil(operands[0])
        else:
            raise ValueError(f"Unknown operator: {op}")
    
    def format_task_for_llm(self, task: CalcChainTask, include_structure: bool = False) -> str:
        """
        Format a task as a prompt for an LLM.
        
        Args:
            task: CalcChainTask to format
            include_structure: Whether to reveal the hidden structure (for analysis)
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""CalcChain Task (Difficulty: {task.difficulty})

Your goal: Reach the target value by applying operations to the given numbers.

Numbers available: {task.numbers}
Target value: {task.target}
Available operators: {task.operators}
Step constraints: minimum {task.min_steps} steps, maximum {task.max_steps} steps

Rules:
1. You must use operations from the available operators list
2. Each operation consumes its input numbers and produces a new number
3. You can use each number from the pool only once, but intermediate results can be reused
4. You must reach the exact target value
5. Your solution must use between {task.min_steps} and {task.max_steps} operations

Format your solution as a sequence of operations, where each operation specifies:
- The operator to use
- The operands (by their current index in the available numbers)
- The result

Example format:
Step 1: numbers[0] + numbers[1] = 15
Step 2: result * numbers[2] = 45
...
"""
        
        if include_structure:
            prompt += f"\n[HIDDEN STRUCTURE - FOR ANALYSIS ONLY]\n"
            prompt += f"Mathematical structure: {task.hidden_structure.value}\n"
            prompt += f"Structure parameters: {task.structure_params}\n"
        
        return prompt


def main():
    """Example usage and testing."""
    env = CalcChainEnvironment(seed=42)
    
    print("=" * 80)
    print("CalcChain RLVR Environment - Mathematical Reasoning Benchmark")
    print("=" * 80)
    print()
    
    # Generate examples from each mathematical structure
    structures = list(MathStructure)
    
    for structure in structures[:3]:  # Show first 3 as examples
        print(f"\n{'='*80}")
        print(f"Example: {structure.value.upper().replace('_', ' ')}")
        print(f"{'='*80}\n")
        
        task = env.generate_task(difficulty='medium', structure=structure)
        print(env.format_task_for_llm(task, include_structure=True))
        print()
    
    # Generate a dataset
    print(f"\n{'='*80}")
    print("Generating dataset...")
    print(f"{'='*80}\n")
    
    dataset = env.generate_dataset(num_tasks=20)
    
    # Statistics
    structure_counts = {}
    difficulty_counts = {}
    
    for task in dataset:
        structure_counts[task.hidden_structure] = structure_counts.get(task.hidden_structure, 0) + 1
        difficulty_counts[task.difficulty] = difficulty_counts.get(task.difficulty, 0) + 1
    
    print(f"Generated {len(dataset)} tasks")
    print(f"\nDifficulty distribution:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  {diff}: {count}")
    
    print(f"\nMathematical structure distribution:")
    for struct, count in sorted(structure_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {struct.value}: {count}")
    
    print(f"\n{'='*80}")
    print("Sample tasks for LLM evaluation:")
    print(f"{'='*80}\n")
    
    for i, task in enumerate(dataset[:3], 1):
        print(f"\nTask {i}:")
        print(env.format_task_for_llm(task, include_structure=False))
        print()


if __name__ == "__main__":
    main()
