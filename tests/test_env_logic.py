import copy
import unittest
from collections import deque

from arihmetic_bench.calcchain_practical import CalcTask, PatternType, PracticalCalcChain
from arihmetic_bench.calcchain_rlvr import CalcChainEnvironment, CalcChainTask, MathStructure
from map_navigate_bench.wilderness_navigator import WildernessNavigator


class TestCalcChainRLVRIntegrity(unittest.TestCase):
    def test_disallowed_operator_is_rejected(self):
        env = CalcChainEnvironment(seed=0)
        task = CalcChainTask(
            numbers=[2, 4],
            operators=["+"],
            target=2,
            max_steps=1,
            min_steps=1,
            hidden_structure=MathStructure.MODULAR_SEQUENCE,
            structure_params={},
            difficulty="easy",
        )
        result = env.verify_solution(task, [("gcd", [0, 1])])
        self.assertFalse(result["success"])
        self.assertFalse(result["valid"])
        self.assertIn("Operator not allowed", result["reason"])

    def test_division_by_zero_is_invalid(self):
        env = CalcChainEnvironment(seed=0)
        task = CalcChainTask(
            numbers=[5, 0],
            operators=["//"],
            target=0,
            max_steps=1,
            min_steps=1,
            hidden_structure=MathStructure.MODULAR_SEQUENCE,
            structure_params={},
            difficulty="easy",
        )
        result = env.verify_solution(task, [("//", [0, 1])])
        self.assertFalse(result["success"])
        self.assertFalse(result["valid"])
        self.assertIn("Division by zero", result["reason"])


class TestPracticalCalcChainIntegrity(unittest.TestCase):
    def test_last_line_bypass_is_rejected(self):
        env = PracticalCalcChain(seed=3)
        task = env.generate_task("easy")
        operations = [f"garbage_{i}" for i in range(task.min_steps - 1)] + [f"0 + 0 = {task.target}"]
        result = env.verify_solution(task, operations)
        self.assertFalse(result["success"])

    def test_valid_manual_trajectory(self):
        env = PracticalCalcChain(seed=0)
        task = CalcTask(
            numbers=[5, 3, 2],
            operators=["+", "*"],
            target=16,
            max_steps=3,
            min_steps=2,
            pattern_type=PatternType.BASIC_ARITHMETIC,
            difficulty="easy",
            hint="",
            ground_truth_steps=[],
        )
        result = env.verify_solution(task, ["5 + 3 = 8", "8 * 2 = 16"])
        self.assertTrue(result["success"])

    def test_bruteforce_trajectory_exists_for_sample_patterns(self):
        env = PracticalCalcChain(seed=42)
        patterns = [
            PatternType.BASIC_ARITHMETIC,
            PatternType.COMPOUND_OPERATIONS,
            PatternType.MODULAR_ARITHMETIC,
        ]
        for pattern in patterns:
            task = env.generate_task("easy", pattern)
            found = _bruteforce_practical_task(env, task)
            self.assertTrue(found, msg=f"No trajectory found for pattern {pattern.value}")


def _bruteforce_practical_task(env: PracticalCalcChain, task: CalcTask) -> bool:
    """Searches for any valid trajectory that reaches task.target as final appended value."""
    queue = deque()
    queue.append((task.numbers.copy(), 0))
    seen = set()

    while queue:
        available, depth = queue.popleft()
        key = (tuple(available), depth)
        if key in seen:
            continue
        seen.add(key)

        if depth >= task.min_steps and available and available[-1] == task.target:
            return True
        if depth == task.max_steps:
            continue

        for op in task.operators:
            arity = env.operator_arity.get(op)
            if arity == 1:
                for i in range(len(available)):
                    operands = [available[i]]
                    try:
                        result = env._apply_operation(op, operands)
                    except Exception:
                        continue
                    if abs(result) > 50000:
                        continue
                    new_avail = available.copy()
                    new_avail.pop(i)
                    new_avail.append(result)
                    queue.append((new_avail, depth + 1))
            elif arity == 2:
                for i in range(len(available)):
                    for j in range(len(available)):
                        if i == j:
                            continue
                        operands = [available[i], available[j]]
                        try:
                            result = env._apply_operation(op, operands)
                        except Exception:
                            continue
                        if abs(result) > 50000:
                            continue
                        new_avail = available.copy()
                        for idx in sorted([i, j], reverse=True):
                            new_avail.pop(idx)
                        new_avail.append(result)
                        queue.append((new_avail, depth + 1))
    return False


class TestWildernessNavigatorIntegrity(unittest.TestCase):
    def test_macro_and_patrol_bracket_syntax(self):
        game = WildernessNavigator()
        macro = game.execute_action("MACRO [MOVE E; MOVE E; MOVE S]")
        self.assertTrue(macro["success"])

        game = WildernessNavigator()
        patrol = game.execute_action("PATROL [MOVE E; TURN RIGHT] 2")
        self.assertTrue(patrol["success"])

    def test_bush_cooldown_expires(self):
        game = WildernessNavigator()
        for action in ["MOVE E", "MOVE E", "MOVE E", "MOVE S", "MOVE S", "MOVE S", "MOVE S"]:
            game.execute_action(action)  # to bush at (4,5)
        first = game.execute_action("SEARCH")
        self.assertTrue(first["success"])

        for _ in range(5):
            game.execute_action("MOVE E")
            game.execute_action("MOVE W")
        second = game.execute_action("SEARCH")
        self.assertTrue(second["success"])

    def test_hook_out_of_bounds_returns_clean_error(self):
        game = WildernessNavigator()
        result = game.execute_action("HOOK -1 1")
        self.assertFalse(result["success"])
        self.assertIn("outside map boundaries", result["message"])

    def test_efficiency_not_high_without_progress(self):
        game = WildernessNavigator()
        metrics = game.get_performance_metrics()
        self.assertLessEqual(metrics["efficiency_score"], 10.0)
        self.assertFalse(metrics["reached_goal"])

    def test_bruteforce_path_to_goal_exists(self):
        actions = _find_basic_move_solution(WildernessNavigator(), max_depth=60)
        self.assertIsNotNone(actions)
        game = WildernessNavigator()
        won = False
        for action in actions:
            result = game.execute_action(action)
            if result.get("won"):
                won = True
                break
        self.assertTrue(won)


def _find_basic_move_solution(game: WildernessNavigator, max_depth: int = 60):
    commands = ["MOVE N", "MOVE S", "MOVE E", "MOVE W"]
    queue = deque([(game, [])])
    seen = set()

    while queue:
        current_game, path = queue.popleft()
        state_key = (current_game.state.player_pos, len(path))
        if state_key in seen:
            continue
        seen.add(state_key)

        if current_game.state.player_pos == current_game.goal_pos:
            return path
        if len(path) >= max_depth:
            continue

        for cmd in commands:
            next_game = copy.deepcopy(current_game)
            result = next_game.execute_action(cmd)
            if result["success"]:
                queue.append((next_game, path + [cmd]))
    return None


if __name__ == "__main__":
    unittest.main()
