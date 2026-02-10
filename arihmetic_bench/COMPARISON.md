# CalcChain: Two Benchmark Approaches

## Overview

This repository contains **two different CalcChain benchmarks**, each designed for different purposes:

1. **Theoretical Benchmark** - Tests deep mathematical reasoning
2. **Practical Benchmark** - RL-friendly arithmetic training

---

## 📚 Theoretical Benchmark (calcchain_rlvr.py)

### Purpose
Test whether LLMs understand advanced mathematical concepts from number theory, topology, and algebra.

### Mathematical Structures
- **Elliptic Curves**: y² = x³ + ax + b (mod p)
- **Fermat's Little Theorem**: a^(p-1) ≡ 1 (mod p)
- **Quadratic Residues**: Legendre symbols
- **Continued Fractions**: Convergent calculations
- **Topological Invariants**: Euler characteristic χ = V - E + F
- **Prime Constellations**: Sophie Germain primes, prime gaps
- **Modular Sequences**: Polynomial evaluation mod m
- **Fibonacci Variants**: Custom recurrence relations

### Example Task
```
Numbers: [7, 2, 5, 13, 3, 1]
Target: 6
Hidden formula: (7³ + 2·7 + 5) mod 13 = 6
```

### Pros
✅ Tests deep mathematical knowledge
✅ Requires genuine reasoning, not just computation
✅ Covers diverse mathematical fields
✅ Interesting for AI research

### Cons
❌ Requires mathematical background to solve
❌ Sparse reward signal for RL agents
❌ Hard to learn patterns without explicit knowledge
❌ Each structure is somewhat isolated

### Best For
- Testing LLMs with mathematical training
- Research on symbolic reasoning
- Evaluating mathematical knowledge depth
- Comparing models' mathematical capabilities

---

## 🎯 Practical Benchmark (calcchain_practical.py)

### Purpose
Train RL agents to perform multi-step arithmetic reasoning through learnable patterns.

### Arithmetic Patterns
1. **Basic Arithmetic**: (a + b) * c
2. **Compound Operations**: a*b + c*d
3. **Nested Operations**: (a + b) * (c - d)
4. **Modular Arithmetic**: (a * b) % m
5. **Power Combinations**: a^b + c
6. **Average and Sum**: (sum/count) * multiplier
7. **Digit Operations**: sum_of_digits(n) * m
8. **Factorial Combos**: n! + m (small n)
9. **Sequential Apply**: ((a + b) * c) - d
10. **Target Decomposition**: Factor/decompose target

### Example Tasks

**Easy:**
```
Target: 12
Numbers: [1, 5, 2]
Solution: 
  1 + 5 = 6
  6 * 2 = 12
```

**Medium:**
```
Target: 75
Numbers: [9, 7, 4, 3]
Solution:
  9 * 7 = 63
  4 * 3 = 12
  63 + 12 = 75
```

**Hard:**
```
Target: 1078
Numbers: [22, 27, 32, 10]
Solution:
  22 + 27 = 49
  32 - 10 = 22
  49 * 22 = 1078
```

### Pros
✅ Learnable through experience (no math background needed)
✅ Clear reward signal (did we reach target?)
✅ Patterns compose (skills transfer)
✅ Realistic operations (like a calculator)
✅ Ground truth solutions provided
✅ Smooth difficulty curve

### Cons
❌ Less mathematically sophisticated
❌ Doesn't test deep mathematical knowledge
❌ Simpler patterns

### Best For
- Training RL agents from scratch
- Multi-step reasoning research
- Curriculum learning experiments
- Real-world calculator-like tasks
- Testing basic arithmetic reasoning

---

## 🔍 Detailed Comparison

| Aspect | Theoretical | Practical |
|--------|-------------|-----------|
| **Target Audience** | LLMs with math knowledge | RL agents learning from scratch |
| **Difficulty Source** | Mathematical sophistication | Multi-step planning |
| **Learning Curve** | Steep (needs math background) | Gradual (learns from experience) |
| **Reward Signal** | Sparse (correct or wrong) | Dense (partial credit possible) |
| **Operators** | +, -, *, //, %, **, mod, gcd, lcm | +, -, *, //, %, ** |
| **Pattern Discovery** | Requires insight | Emerges from trials |
| **Skill Transfer** | Limited between structures | High between patterns |
| **Dataset Size** | 64 tasks (8 per structure) | 500 tasks (50 per pattern) |
| **Verification** | Hard to verify optimality | Easy to verify success |

---

## 📊 Complexity Analysis

### Theoretical Benchmark
**Average Complexity:**
- Max steps: 7.6
- Numbers per task: 8.2
- Target magnitude: ~1.5M (varies widely)
- **Complexity Score**: Mathematical knowledge required

### Practical Benchmark
**Average Complexity (by pattern):**

| Pattern | Steps | Numbers | Score |
|---------|-------|---------|-------|
| Modular Arithmetic | 3.0 | 3.0 | 9.0 ⭐ Easiest |
| Power Combinations | 3.0 | 3.0 | 9.0 ⭐ Easiest |
| Basic Arithmetic | 3.0 | 5.0 | 15.0 |
| Compound Operations | 4.0 | 4.0 | 16.0 |
| Nested Operations | 4.0 | 4.0 | 16.0 |
| Sequential Apply | 4.0 | 4.0 | 16.0 |
| Digit Operations | 5.0 | 5.0 | 25.0 |
| Average and Sum | 6.0 | 6.0 | 36.0 |
| Factorial Combo | 6.0 | 7.0 | 42.0 🔴 Hardest |

---

## 🎓 When to Use Which?

### Use Theoretical Benchmark When:
1. You want to test mathematical reasoning depth
2. You're evaluating LLMs with mathematical training
3. You're researching symbolic reasoning capabilities
4. You want to compare models' mathematical knowledge
5. You're studying how AI understands mathematical structures

**Example Use Cases:**
- Testing GPT-4 vs Claude on math problems
- Evaluating impact of mathematical training data
- Research on theorem discovery
- Mathematical competition preparation

### Use Practical Benchmark When:
1. You're training RL agents from scratch
2. You want learnable multi-step reasoning
3. You need curriculum learning (easy → hard)
4. You're testing calculator-like capabilities
5. You want high agent success rates

**Example Use Cases:**
- Training a calculator agent
- Multi-step planning research
- Curriculum learning experiments
- Testing arithmetic reasoning without math knowledge
- Building practical AI assistants

---

## 📁 Files Included

### Theoretical Benchmark
- `calcchain_rlvr.py` - Environment with 8 mathematical structures
- `calcchain_evaluation.py` - Evaluation and analysis tools
- `calcchain_benchmark.json` - 64 pre-generated tasks
- `README.md` - Mathematical theory documentation
- `EXAMPLES.md` - Worked solutions

### Practical Benchmark
- `calcchain_practical.py` - Environment with 10 learnable patterns
- `generate_practical_dataset.py` - Dataset generation and analysis
- `calcchain_practical_dataset.json` - 500 pre-generated tasks

---

## 🚀 Quick Start

### For Theoretical Benchmark (LLM Testing)
```python
from calcchain_rlvr import CalcChainEnvironment

env = CalcChainEnvironment(seed=42)

# Generate a task with elliptic curves
task = env.generate_task(
    difficulty='medium',
    structure=MathStructure.ELLIPTIC_CURVE
)

# Format for LLM
prompt = env.format_task_for_llm(task)
print(prompt)
```

### For Practical Benchmark (RL Training)
```python
from calcchain_practical import PracticalCalcChain

env = PracticalCalcChain(seed=42)

# Generate learnable task
task = env.generate_task(
    difficulty='easy',
    pattern=PatternType.BASIC_ARITHMETIC
)

# See ground truth solution
print(f"Target: {task.target}")
print(f"Numbers: {task.numbers}")
print("\nSolution:")
for step in task.ground_truth_steps:
    print(f"  {step}")
```

---

## 🎯 Recommendations

### For AI Research Labs:
- **Use both benchmarks** to test different capabilities
- Theoretical benchmark → Test mathematical reasoning
- Practical benchmark → Test learning and planning

### For RL Researchers:
- **Start with practical benchmark**
- Begin with easy tasks (Basic Arithmetic, Power Combinations)
- Gradually increase difficulty
- Track which patterns agent learns fastest

### For LLM Evaluation:
- **Use theoretical benchmark**
- Compare performance across mathematical structures
- Analyze which structures are most challenging
- Test impact of mathematical training data

### For Education/Training:
- **Use practical benchmark**
- Clear progression from simple to complex
- Ground truth solutions for learning
- Realistic operations students understand

---

## 📈 Expected Performance

### Theoretical Benchmark
- **Random Baseline**: ~0% (guessing won't work)
- **Strong LLM (GPT-4)**: 30-60% depending on structure
- **Math-trained LLM**: 50-80%
- **RL Agent (no math training)**: <5%

### Practical Benchmark
- **Random Baseline**: ~0%
- **RL Agent (after training)**: 60-90% on easy, 30-60% on hard
- **LLM (GPT-4)**: 80-95% (can reason about simple arithmetic)
- **Calculator Program**: 100% (if given correct formula)

---

## 🔮 Future Extensions

### Theoretical Benchmark
- Combine multiple structures in one task
- Add proof generation requirements
- Interactive multi-turn problem solving
- Transfer learning between structures

### Practical Benchmark
- Add more operator types (sqrt, log, etc.)
- Multi-variable patterns
- Conditional operations
- Time-constrained solving
- Partial credit scoring

---

## 🤝 Contributing

Both benchmarks are open for extension! Ideas:
- New mathematical structures (theoretical)
- New arithmetic patterns (practical)
- Better curriculum strategies
- Improved verification methods
- Visualization tools

---

## 📄 License

MIT License - Free for research and educational use.

---

## 🎓 Citation

```bibtex
@misc{calcchain2025,
  title={CalcChain: Dual-Track Arithmetic Reasoning Benchmarks},
  author={Claude (Anthropic)},
  year={2025},
  note={Theoretical and Practical Benchmarks for Mathematical Reasoning}
}
```

---

## 💡 Key Takeaway

**Choose your benchmark based on your goal:**

- Want to test **mathematical knowledge**? → Use **Theoretical**
- Want to train **learning agents**? → Use **Practical**
- Want to do **comprehensive evaluation**? → Use **Both**

Both benchmarks test multi-step reasoning, but through different lenses:
- **Theoretical**: "Can you apply mathematical knowledge?"
- **Practical**: "Can you learn arithmetic patterns?"

Each serves a valuable purpose in understanding AI capabilities! 🚀
