# CalcChain RLVR: Mathematical Reasoning Benchmark

## Overview

CalcChain is an arithmetic reasoning benchmark that challenges language models to reach precise target values through multi-step mathematical operations. What makes this benchmark unique is that targets are generated using **hidden mathematical structures** from various branches of mathematics including number theory, topology, algebra, and discrete mathematics.

The model is given:
- A pool of integers
- A set of allowed operators  
- A target value
- Step constraints (min/max operations)

The model must discover the hidden mathematical pattern and synthesize the correct sequence of operations.

## Mathematical Structures

### 1. Elliptic Curves 🎯

**Theory**: Elliptic curves are algebraic structures defined by equations of the form:
```
y² = x³ + ax + b (mod p)
```

These curves have deep connections to number theory, cryptography, and the famous proof of Fermat's Last Theorem.

**How it works in CalcChain**:
- Target = (x³ + ax + b) mod p for hidden parameters a, b, p
- The model receives x, a, b, p, and small integers
- Must discover the cubic polynomial structure and modular arithmetic

**Example**:
```
Numbers: [7, 2, 5, 13, 3, 1]
Target: 6
Hidden formula: (7³ + 2·7 + 5) mod 13 = (343 + 14 + 5) mod 13 = 362 mod 13 = 6
```

**Why it's challenging**: The model must recognize that the target involves:
1. Cubic powers (x³)
2. Linear combination with coefficients
3. Modular reduction

### 2. Modular Sequences 🔢

**Theory**: Polynomial evaluation in modular arithmetic creates patterns that repeat cyclically. The general form is:
```
f(n) = a₀n^k + a₁n^(k-1) + ... + aₖ (mod m)
```

This connects to finite field theory and coding theory.

**How it works**:
- Target follows a polynomial pattern mod m
- Model must identify degree k and reconstruct the polynomial
- Requires understanding of modular arithmetic properties

**Example**:
```
Numbers: [3, 2, 5, 11, ...]
Target: 8
Hidden: 2·3² + 5·3 + 1 (mod 11) = 18 + 15 + 1 (mod 11) = 34 mod 11 = 1
```

### 3. Fibonacci Variants 📈

**Theory**: Generalized Fibonacci sequences follow the recurrence:
```
F(n) = a·F(n-1) + b·F(n-2) + c
```

Standard Fibonacci is the case a=1, b=1, c=0. These sequences appear throughout nature, computer science, and combinatorics.

**How it works**:
- Target is the n-th term of a hidden recurrence
- Model receives initial terms F(0), F(1) and parameters a, b, c
- Must discover and apply the recurrence relation

**Example**:
```
Numbers: [2, 3, 1, 5, ...] 
F(0)=1, F(1)=2, a=2, b=3, c=1
F(2) = 2·2 + 3·1 + 1 = 8
F(3) = 2·8 + 3·2 + 1 = 23
F(4) = 2·23 + 3·8 + 1 = 71
Target: 71
```

### 4. Prime Constellations ⭐

**Theory**: Prime numbers exhibit fascinating patterns in their distribution:
- **Prime gaps**: The difference between consecutive primes
- **Sophie Germain primes**: Primes p where 2p+1 is also prime
- **Twin primes**: Primes p and p+2

**How it works**:
- Target relates to prime gap patterns or special prime relationships
- Model receives several primes but must discover the relationship
- Tests knowledge of prime properties

**Example**:
```
Primes: [7, 11, 13, 17]
Gaps: [4, 2, 4]
Target: (11-7) * (13-11) + 7 = 4 * 2 + 7 = 15
```

### 5. Continued Fractions 🔀

**Theory**: Any real number can be represented as a continued fraction:
```
x = a₀ + 1/(a₁ + 1/(a₂ + 1/(a₃ + ...)))
```

The convergents (rational approximations) are computed via:
```
p₋₁ = 1, p₀ = a₀
pₙ = aₙpₙ₋₁ + pₙ₋₂

q₋₁ = 0, q₀ = 1
qₙ = aₙqₙ₋₁ + qₙ₋₂
```

**How it works**:
- Target is a convergent numerator pₙ
- Model receives the continued fraction coefficients [a₀; a₁, a₂, ...]
- Must apply the convergent recurrence

**Example**:
```
CF: [3; 7, 15, 1]
p₀ = 3, q₀ = 1
p₁ = 7·3 + 1 = 22, q₁ = 7·1 + 0 = 7
p₂ = 15·22 + 3 = 333, q₂ = 15·7 + 1 = 106
Target: 333
```

### 6. Topological Invariants 🔺

**Theory**: The Euler characteristic χ is a topological invariant:
- For polyhedra: χ = V - E + F (vertices - edges + faces)
- For surfaces: χ = 2 - 2g (g = genus, number of "holes")
- For surfaces with boundary: χ = 2 - 2g - b

**Key insight**: χ = 2 for all simple polyhedra (sphere-like), χ = 0 for torus, χ < 0 for higher genus.

**How it works**:
- Target is the Euler characteristic
- Model receives V, E, F or genus g
- Must discover and apply the Euler formula

**Example**:
```
Cube: V=8, E=12, F=6
Target: 8 - 12 + 6 = 2

Torus: g=1
Target: 2 - 2·1 = 0
```

### 7. Fermat's Little Theorem 🎓

**Theory**: For prime p and integer a not divisible by p:
```
a^(p-1) ≡ 1 (mod p)
```

This fundamental theorem underlies RSA cryptography and primality testing.

**How it works**:
- Target involves modular exponentiation: a^k (mod p)
- When k is a multiple of p-1, the result cycles
- Model must recognize the pattern and use Fermat's theorem

**Example**:
```
a=3, p=7, k=6
3^6 mod 7 = 729 mod 7 = 1 (since 6 = p-1)
3^8 mod 7 = 3^6 · 3^2 mod 7 = 1 · 9 mod 7 = 2
```

### 8. Quadratic Residues 🎲

**Theory**: An integer a is a quadratic residue mod p if:
```
∃x: x² ≡ a (mod p)
```

The Legendre symbol (a/p) = a^((p-1)/2) mod p ∈ {-1, 0, 1} indicates whether a is a residue.

**How it works**:
- Target involves checking if a is a square mod p
- Model must compute the Legendre symbol
- Requires understanding of modular squares

**Example**:
```
a=3, p=7
Squares mod 7: {0, 1, 2, 4}
3 is not in this set → not a residue
Legendre symbol: 3^3 mod 7 = 27 mod 7 = 6 ≡ -1
```

## Operators Available

### Basic Operators
- `+`, `-`, `*`: Standard arithmetic
- `//`: Integer division
- `%`: Modulo operation

### Intermediate Operators
- `**`: Exponentiation
- `abs`: Absolute value
- `neg`: Negation

### Advanced Operators
- `sqrt`: Integer square root
- `log2`: Binary logarithm (integer)
- `gcd`: Greatest common divisor
- `lcm`: Least common multiple
- `floor`, `ceil`: Rounding operations

## Difficulty Levels

### Easy
- Simple patterns
- 3-5 operations
- Small numbers (< 30)
- Basic operators

### Medium
- Moderate complexity
- 4-8 operations
- Medium numbers (< 100)
- Intermediate operators

### Hard
- Complex patterns
- 6-10 operations
- Larger numbers (< 500)
- Advanced operators

### Expert
- Very complex structures
- 8-12 operations
- Large numbers (< 1000)
- Full operator set

## Why This Benchmark Matters

### Tests Deep Mathematical Reasoning
Unlike simple arithmetic tasks, CalcChain requires:
1. **Pattern Recognition**: Identifying hidden mathematical structures
2. **Strategic Planning**: Choosing the right operations in the right order
3. **Mathematical Knowledge**: Understanding number theory, algebra, topology
4. **Multi-step Reasoning**: Chaining operations toward a precise target

### No Memorization
The hidden structures ensure models cannot simply memorize solutions. Each task requires genuine mathematical reasoning.

### Diverse Skills
Different structures test different capabilities:
- **Elliptic Curves**: Polynomial evaluation, modular arithmetic
- **Sequences**: Recursive thinking, pattern recognition
- **Primes**: Number theoretic properties
- **Topology**: Abstract mathematical thinking
- **Continued Fractions**: Rational approximation, recurrence relations

## Evaluation Metrics

### Success Rate
Percentage of tasks where the model reaches the exact target value.

### Step Efficiency
How many operations the model uses compared to the minimum required.

### Structure Recognition
Can the model identify which mathematical structure is being used?

### Generalization
Performance on novel combinations of structures and parameters.

## Usage

```python
from calcchain_rlvr import CalcChainEnvironment

# Create environment
env = CalcChainEnvironment(seed=42)

# Generate a task
task = env.generate_task(difficulty='medium')

# Format for LLM
prompt = env.format_task_for_llm(task)

# Verify solution
result = env.verify_solution(task, operations)
```

## Future Extensions

1. **Composition**: Tasks requiring multiple structures
2. **Dynamic Operators**: Allowing models to define custom operations
3. **Proof Requirements**: Asking models to explain their reasoning
4. **Interactive Mode**: Multi-turn refinement of solutions
5. **Transfer Learning**: Testing if models can learn structures from examples

## Citation

If you use CalcChain in your research, please cite:

```bibtex
@misc{calcchain2025,
  title={CalcChain: A Mathematical Reasoning Benchmark with Hidden Structures},
  author={Claude (Anthropic)},
  year={2025},
  note={RLVR Environment for testing LLM mathematical reasoning}
}
```

## License

MIT License - Free for research and educational use.

---

**Note**: This benchmark is designed to push the boundaries of mathematical reasoning in language models. The hidden structures ensure that success requires genuine understanding, not just pattern matching or memorization.
