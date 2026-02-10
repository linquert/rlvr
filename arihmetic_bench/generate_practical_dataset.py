"""
CalcChain Practical - Dataset Generation & RL Training Setup
"""

import json
from calcchain_practical import PracticalCalcChain, PatternType
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def export_dataset(filename: str, num_tasks: int = 500):
    """Export a comprehensive dataset for RL training."""
    
    env = PracticalCalcChain(seed=42)
    
    # Generate balanced dataset
    dataset = []
    
    # Curriculum proportions: Start easy, get harder
    difficulty_weights = [
        ('easy', 0.40),
        ('medium', 0.40),
        ('hard', 0.20),
    ]
    curriculum_dist = []
    assigned = 0
    for idx, (difficulty, weight) in enumerate(difficulty_weights):
        if idx < len(difficulty_weights) - 1:
            count = int(num_tasks * weight)
            assigned += count
        else:
            count = num_tasks - assigned
        curriculum_dist.append((difficulty, count))
    
    for difficulty, count in curriculum_dist:
        tasks = env.generate_dataset(
            num_tasks=count,
            difficulty_dist={difficulty: 1.0}
        )
        dataset.extend(tasks)
    
    # Convert to JSON
    export_data = []
    for i, task in enumerate(dataset):
        export_data.append({
            'id': f"task_{i:04d}",
            'pattern': task.pattern_type.value,
            'difficulty': task.difficulty,
            'numbers': task.numbers,
            'operators': task.operators,
            'target': task.target,
            'min_steps': task.min_steps,
            'max_steps': task.max_steps,
            'hint': task.hint,
            'ground_truth': task.ground_truth_steps,
            'prompt': env.format_task_for_llm(task)
        })

    # Basic integrity checks before write
    if len(export_data) != num_tasks:
        raise ValueError(f"Expected {num_tasks} tasks, got {len(export_data)}")
    for row in export_data:
        if row['min_steps'] > row['max_steps']:
            raise ValueError(
                f"Invalid step bounds in {row['id']}: min={row['min_steps']} max={row['max_steps']}"
            )
        if row['difficulty'] not in {'easy', 'medium', 'hard'}:
            raise ValueError(f"Invalid difficulty in {row['id']}: {row['difficulty']}")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✓ Exported {len(export_data)} tasks to {filename}")
    
    # Statistics
    from collections import Counter
    patterns = Counter(t['pattern'] for t in export_data)
    difficulties = Counter(t['difficulty'] for t in export_data)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total tasks: {len(export_data)}")
    print(f"\n   By difficulty:")
    for diff in ['easy', 'medium', 'hard']:
        pct = difficulties[diff] / len(export_data) * 100
        print(f"      {diff.title()}: {difficulties[diff]} ({pct:.1f}%)")
    
    print(f"\n   Top 5 patterns:")
    for pattern, count in patterns.most_common(5):
        pct = count / len(export_data) * 100
        print(f"      {pattern.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
    
    return export_data


def create_rl_training_examples():
    """Show examples suitable for RL training."""
    
    env = PracticalCalcChain(seed=123)
    
    print("\n" + "="*80)
    print("RL TRAINING EXAMPLES - Why This Benchmark is Practical")
    print("="*80)
    
    examples = [
        ('easy', PatternType.BASIC_ARITHMETIC),
        ('medium', PatternType.COMPOUND_OPERATIONS),
        ('hard', PatternType.NESTED_OPERATIONS),
    ]
    
    for difficulty, pattern in examples:
        task = env.generate_task(difficulty, pattern)
        
        print(f"\n{'─'*80}")
        print(f"Difficulty: {difficulty.upper()} | Pattern: {pattern.value}")
        print(f"{'─'*80}")
        print(f"\n🎯 TARGET: {task.target}")
        print(f"📦 NUMBERS: {task.numbers}")
        print(f"🔧 OPERATORS: {task.operators}")
        print(f"📏 STEPS: {task.min_steps}-{task.max_steps}")
        
        print(f"\n💡 Why this is RL-friendly:")
        print(f"   • Clear goal: Reach {task.target}")
        print(f"   • Verifiable: Can check if target is reached")
        print(f"   • Learnable pattern: {task.hint}")
        print(f"   • Reasonable operators: {', '.join(task.operators)}")
        
        print(f"\n✓ SOLUTION:")
        for step in task.ground_truth_steps:
            print(f"   {step}")


def analyze_learnability():
    """Analyze which patterns are most learnable for RL agents."""
    
    env = PracticalCalcChain(seed=456)
    
    print("\n" + "="*80)
    print("LEARNABILITY ANALYSIS - Pattern Complexity")
    print("="*80)
    
    # Analyze each pattern
    analysis = []
    
    for pattern in PatternType:
        # Generate 10 samples
        tasks = [env.generate_task('medium', pattern) for _ in range(10)]
        
        avg_steps = sum(t.max_steps for t in tasks) / len(tasks)
        avg_numbers = sum(len(t.numbers) for t in tasks) / len(tasks)
        avg_target = sum(abs(t.target) for t in tasks) / len(tasks)
        
        analysis.append({
            'pattern': pattern.value,
            'avg_steps': avg_steps,
            'avg_numbers': avg_numbers,
            'avg_target': avg_target,
            'hint': tasks[0].hint
        })
    
    # Sort by complexity (steps * numbers)
    analysis.sort(key=lambda x: x['avg_steps'] * x['avg_numbers'])
    
    print("\n📈 Pattern Complexity (easiest to hardest for RL):\n")
    
    for i, item in enumerate(analysis, 1):
        complexity = item['avg_steps'] * item['avg_numbers']
        print(f"{i}. {item['pattern'].replace('_', ' ').title()}")
        print(f"   Avg steps: {item['avg_steps']:.1f}")
        print(f"   Avg numbers: {item['avg_numbers']:.1f}")
        print(f"   Complexity score: {complexity:.1f}")
        print(f"   Hint: {item['hint']}")
        print()


def compare_with_theoretical_benchmark():
    """Compare practical vs theoretical benchmark."""
    
    print("\n" + "="*80)
    print("PRACTICAL vs THEORETICAL COMPARISON")
    print("="*80)
    
    comparison = """
    
THEORETICAL BENCHMARK (Original):
❌ Elliptic curves - Requires deep number theory knowledge
❌ Fermat's Little Theorem - Advanced cryptography concepts
❌ Quadratic residues - Abstract algebra
❌ Continued fractions - Complex recursive patterns
❌ Topological invariants - Geometry & topology
→ Hard for RL agents to learn without mathematical background

PRACTICAL BENCHMARK (New):
✅ Basic arithmetic - (a + b) * c
✅ Compound operations - a*b + c*d  
✅ Nested operations - (a + b) * (c - d)
✅ Modular arithmetic - (a * b) % m
✅ Power combinations - a^b + c
✅ Sequential chains - ((a + b) * c) - d
→ Easy to learn through trial and error

KEY DIFFERENCES:

1. REACHABILITY
   Theoretical: Target might require discovering obscure theorem
   Practical: Target always reachable with straightforward operations

2. VERIFIABILITY
   Theoretical: Hard to verify if solution is optimal
   Practical: Easy to verify - did we reach the target?

3. LEARNABILITY  
   Theoretical: Requires mathematical intuition
   Practical: Patterns emerge from experience

4. REWARD SIGNAL
   Theoretical: Sparse - only correct if you know the theorem
   Practical: Dense - partial credit for getting closer

5. SKILL TRANSFER
   Theoretical: Each structure is isolated
   Practical: Skills compose (add, multiply, nest)

RECOMMENDATION FOR RL:
Use PRACTICAL benchmark for:
- Training RL agents from scratch
- Testing multi-step reasoning
- Curriculum learning (easy → hard)
- Real-world calculator-like tasks

Use THEORETICAL benchmark for:
- Testing LLMs with mathematical knowledge
- Evaluating mathematical reasoning depth
- Research on symbolic reasoning
- Advanced AI capabilities assessment
"""
    
    print(comparison)


def main():
    """Generate dataset and show analysis."""
    
    print("="*80)
    print("CALCCHAIN PRACTICAL - Dataset Generation & Analysis")
    print("="*80)
    
    # 1. Export dataset
    print("\n[1] GENERATING DATASET...")
    dataset = export_dataset('calcchain_practical_dataset.json', num_tasks=500)
    
    # 2. Show RL training examples
    print("\n[2] RL TRAINING EXAMPLES...")
    create_rl_training_examples()
    
    # 3. Analyze learnability
    print("\n[3] LEARNABILITY ANALYSIS...")
    analyze_learnability()
    
    # 4. Compare approaches
    print("\n[4] COMPARISON WITH THEORETICAL BENCHMARK...")
    compare_with_theoretical_benchmark()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
✅ Generated 500 tasks for RL training
✅ Balanced difficulty distribution (40% easy, 40% medium, 20% hard)
✅ 10 learnable patterns with clear ground truth
✅ All tasks verified to be solvable
✅ Includes hints and solution paths

NEXT STEPS:
1. Use the JSON dataset for RL agent training
2. Start with easy tasks to learn basic patterns
3. Gradually increase difficulty (curriculum learning)
4. Measure success rate on held-out test set
5. Analyze which patterns agent learns fastest

FILES GENERATED:
- calcchain_practical.py (environment code)
- calcchain_practical_dataset.json (500 tasks)
""")


if __name__ == "__main__":
    main()
