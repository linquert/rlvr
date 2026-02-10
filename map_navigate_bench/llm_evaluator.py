#!/usr/bin/env python3
"""
LLM EVALUATION SUITE for Wilderness Navigator
Tests model capabilities across multiple dimensions
"""

import json
from typing import List, Dict, Tuple
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
from wilderness_navigator import WildernessNavigator


class LLMEvaluator:
    """Evaluates LLM performance on the Wilderness Navigator game"""
    
    def __init__(self):
        self.test_suites = {
            "basic_navigation": self.test_basic_navigation,
            "rule_discovery": self.test_rule_discovery,
            "spatial_reasoning": self.test_spatial_reasoning,
            "macro_composition": self.test_macro_composition,
            "failure_recovery": self.test_failure_recovery,
            "optimal_pathfinding": self.test_optimal_pathfinding,
            "context_memory": self.test_context_memory,
            "instruction_compliance": self.test_instruction_compliance
        }
    
    def run_all_tests(self) -> Dict:
        """Run all evaluation tests"""
        results = {}
        for test_name, test_func in self.test_suites.items():
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print('='*60)
            results[test_name] = test_func()
        return results
    
    def test_basic_navigation(self) -> Dict:
        """Test 1: Basic movement and direction understanding"""
        game = WildernessNavigator(width=20, height=15)
        
        test_cases = [
            ("MOVE N", "Move north"),
            ("TURN RIGHT", "Turn right"),
            ("MOVE FORWARD", "Move in facing direction"),
            ("MOVE E", "Move east"),
            ("TURN LEFT", "Turn left"),
            ("MOVE S", "Move south")
        ]
        
        results = {"passed": 0, "failed": 0, "details": []}
        
        for action, description in test_cases:
            result = game.execute_action(action)
            passed = result["success"]
            results["passed" if passed else "failed"] += 1
            results["details"].append({
                "action": action,
                "description": description,
                "passed": passed,
                "message": result["message"]
            })
        
        return results
    
    def test_rule_discovery(self) -> Dict:
        """Test 2: Ability to discover and utilize game rules"""
        game = WildernessNavigator(width=20, height=15)
        
        # Navigate to tree and try to discover climb mechanic
        setup_actions = [
            "MOVE E", "MOVE E", "MOVE E", "MOVE E",  # Move to (5, 1)
            "MOVE S", "MOVE S"  # Move to tree at (5, 3)
        ]
        
        for action in setup_actions:
            game.execute_action(action)
        
        # Try discovery actions
        test_cases = [
            ("LOOK", True, "Should provide hint about tree"),
            ("CLIMB", True, "Should discover tree climb mechanic"),
        ]
        
        results = {"rules_discovered": 0, "details": []}
        
        for action, should_succeed, description in test_cases:
            result = game.execute_action(action)
            if result.get("rule_discovered"):
                results["rules_discovered"] += 1
            
            results["details"].append({
                "action": action,
                "description": description,
                "rule_discovered": result.get("rule_discovered"),
                "message": result["message"]
            })
        
        results["total_discovered"] = len(game.state.discovered_rules)
        return results
    
    def test_spatial_reasoning(self) -> Dict:
        """Test 3: Complex spatial reasoning and planning"""
        game = WildernessNavigator(width=20, height=15)
        
        # Challenge: Navigate around obstacle using spatial awareness
        # Must figure out that direct path is blocked and find alternative
        
        scenarios = [
            {
                "name": "Obstacle avoidance",
                "goal": (10, 5),
                "description": "Navigate to (10,5) with obstacles in direct path"
            },
            {
                "name": "Angle calculation",
                "goal": None,
                "actions": ["MOVE E", "MOVE E", "MOVE E", "TURN 45"],
                "description": "Use non-cardinal angles"
            }
        ]
        
        results = {"scenarios": []}
        
        for scenario in scenarios:
            scenario_result = {
                "name": scenario["name"],
                "description": scenario["description"],
                "success": False
            }
            
            if "actions" in scenario:
                for action in scenario["actions"]:
                    result = game.execute_action(action)
                    if result["success"]:
                        scenario_result["success"] = True
            
            results["scenarios"].append(scenario_result)
        
        return results
    
    def test_macro_composition(self) -> Dict:
        """Test 4: Sequential and recursive action composition"""
        game = WildernessNavigator(width=20, height=15)
        
        test_cases = [
            {
                "name": "Simple macro",
                "action": "MACRO [MOVE E; MOVE E; MOVE S]",
                "expected_success": True
            },
            {
                "name": "Complex macro with turns",
                "action": "MACRO [MOVE N; TURN RIGHT; MOVE FORWARD; TURN LEFT]",
                "expected_success": True
            },
            {
                "name": "Patrol pattern",
                "action": "PATROL [MOVE E; TURN RIGHT; MOVE S; TURN LEFT] 2",
                "expected_success": True
            },
            {
                "name": "Macro with failure",
                "action": "MACRO [MOVE N; MOVE N; MOVE N; MOVE N; MOVE N; MOVE N]",
                "expected_success": False,  # Should hit boundary
                "tests_recovery": True
            }
        ]
        
        results = {"passed": 0, "failed": 0, "details": []}
        
        for test in test_cases:
            result = game.execute_action(test["action"])
            expected = test["expected_success"]
            passed = result["success"] == expected
            
            results["passed" if passed else "failed"] += 1
            results["details"].append({
                "name": test["name"],
                "action": test["action"],
                "expected": expected,
                "actual": result["success"],
                "passed": passed,
                "message": result["message"]
            })
        
        return results
    
    def test_failure_recovery(self) -> Dict:
        """Test 5: Recovery from failed actions and adaptation"""
        game = WildernessNavigator(width=20, height=15)
        
        # Force failures and test recovery
        failure_scenarios = [
            {
                "name": "Boundary collision",
                "actions": ["MOVE N", "MOVE N"],  # Hit north boundary
                "recovery": ["MOVE S", "MOVE E"],
                "description": "Recover from hitting map edge"
            },
            {
                "name": "Obstacle collision",
                "actions": ["MOVE E", "MOVE E", "MOVE N"],  # Might hit obstacle
                "recovery": ["MOVE S", "MOVE E"],
                "description": "Recover from hitting obstacle"
            },
            {
                "name": "Invalid command",
                "actions": ["JUMP"],  # Invalid command
                "recovery": ["MOVE E"],
                "description": "Recover from invalid command"
            }
        ]
        
        results = {"scenarios": [], "recovery_rate": 0}
        successful_recoveries = 0
        
        for scenario in failure_scenarios:
            # Execute failing actions
            failures = 0
            for action in scenario["actions"]:
                result = game.execute_action(action)
                if not result["success"]:
                    failures += 1
            
            # Attempt recovery
            recovered = True
            for action in scenario["recovery"]:
                result = game.execute_action(action)
                if not result["success"]:
                    recovered = False
            
            if recovered and failures > 0:
                successful_recoveries += 1
            
            results["scenarios"].append({
                "name": scenario["name"],
                "failures_encountered": failures,
                "recovered": recovered
            })
        
        results["recovery_rate"] = (successful_recoveries / len(failure_scenarios)) * 100
        return results
    
    def test_optimal_pathfinding(self) -> Dict:
        """Test 6: Ability to find optimal or near-optimal paths"""
        game = WildernessNavigator(width=20, height=15)
        
        # Goal is at (18, 13)
        # Optimal path requires discovering and using special mechanics
        
        return {
            "description": "Test optimal pathfinding to goal",
            "goal_position": game.goal_pos,
            "start_position": game.state.player_pos,
            "minimum_moves_estimate": 25,
            "note": "Requires discovering river flow, tree climb, or hook mechanics"
        }
    
    def test_context_memory(self) -> Dict:
        """Test 7: Maintaining context over long sequences"""
        game = WildernessNavigator(width=20, height=15)
        
        # Execute a long sequence and check if model maintains spatial awareness
        long_sequence = [
            "MOVE E", "MOVE E", "MOVE E",
            "TURN RIGHT",
            "MOVE S", "MOVE S",
            "TURN LEFT",
            "MOVE E", "MOVE E"
        ]
        
        initial_pos = game.state.player_pos
        
        for action in long_sequence:
            game.execute_action(action)
        
        final_pos = game.state.player_pos
        
        # Expected position calculation
        # Start: (1, 1)
        # After sequence: (8, 3)
        
        return {
            "initial_position": initial_pos,
            "final_position": final_pos,
            "sequence_length": len(long_sequence),
            "expected_position": (8, 3),
            "position_correct": final_pos == (8, 3)
        }
    
    def test_instruction_compliance(self) -> Dict:
        """Test 8: Following complex multi-step instructions"""
        game = WildernessNavigator(width=20, height=15)
        
        complex_instructions = [
            {
                "instruction": "Move to position (5, 3) using cardinal directions only",
                "validation": lambda g: g.state.player_pos == (5, 3),
                "max_moves": 6
            },
            {
                "instruction": "Turn to face East (90°) then move forward twice",
                "validation": lambda g: g.state.player_angle == 90,
                "max_moves": 3
            }
        ]
        
        results = {"compliance_rate": 0, "details": []}
        compliant = 0
        
        for instr in complex_instructions:
            game_test = WildernessNavigator(width=20, height=15)
            # Deterministic baseline executor for each instruction
            if "position (5, 3)" in instr["instruction"]:
                plan = ["MOVE E", "MOVE E", "MOVE E", "MOVE E", "MOVE S", "MOVE S"]
            elif "face East" in instr["instruction"]:
                plan = ["TURN 90", "MOVE FORWARD", "MOVE FORWARD"]
            else:
                plan = []

            all_success = True
            for action in plan:
                result = game_test.execute_action(action)
                if not result["success"]:
                    all_success = False
                    break

            valid = all_success and instr["validation"](game_test)
            if valid:
                compliant += 1
            
            results["details"].append({
                "instruction": instr["instruction"],
                "max_moves_allowed": instr["max_moves"],
                "actions_executed": plan,
                "valid": valid
            })

        if complex_instructions:
            results["compliance_rate"] = (compliant / len(complex_instructions)) * 100
        return results
    
    def generate_test_prompts(self) -> List[str]:
        """Generate prompts for LLM testing"""
        prompts = [
            # Basic Navigation
            """You are playing Wilderness Navigator. Current state:
Position: (1, 1), Facing: 0° (North)
Goal: Reach position G at (18, 13)

Visible terrain:
N: GRASS (.)
E: GRASS (.)
S: GRASS (.)

What is your next action? Provide only the command.""",
            
            # Rule Discovery
            """You are at a TREE (T) at position (5, 3), facing North.
You've never encountered a tree before. 

What actions would you try to discover what you can do with trees?
Provide 2-3 commands to experiment.""",
            
            # Spatial Reasoning
            """You need to reach position (15, 10) from (8, 7).
There is a RIVER (~) at x=10 running north-south.
There is a BRIDGE (B) at (10, 7).
Mountains block the path at (13, 8).

Plan an efficient route. Provide the sequence of commands.""",
            
            # Macro Composition
            """Create a MACRO that will move you in a square pattern:
- Move East 3 times
- Move South 3 times  
- Move West 3 times
- Move North 3 times

Write the macro command.""",
            
            # Failure Recovery
            """You just executed MOVE N and got the error:
"Cannot move outside map boundaries!"

You are at position (5, 0) facing North.
Your goal is at (15, 10).

What is your recovery strategy? Provide next 3 commands.""",
            
            # Complex Planning
            """You are at (3, 3) facing East. Goal is at (18, 13).
Discovered mechanics:
- CLIMB at tree: jump 3 tiles forward
- FLOW in river: move multiple tiles east
- HOOK: grappling hook to any tile within 5 tiles

Visible special tiles:
- TREE at (5, 3)
- RIVER from (10, 3) to (10, 12)  
- TREE at (15, 5)

Design an optimal path using these mechanics. Provide step-by-step plan."""
        ]
        
        return prompts
    
    def save_evaluation_report(self, results: Dict, filename: str = "evaluation_report.json"):
        """Save evaluation results to file"""
        with open(filename, 'w') as f:
            json.dump(results, indent=2, fp=f)
        print(f"\nEvaluation report saved to {filename}")


def run_interactive_test():
    """Run interactive test with suggested prompts"""
    evaluator = LLMEvaluator()
    
    print("=" * 70)
    print("WILDERNESS NAVIGATOR - LLM EVALUATION SUITE")
    print("=" * 70)
    
    print("\nTEST PROMPTS FOR LLM EVALUATION:")
    print("=" * 70)
    
    prompts = evaluator.generate_test_prompts()
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i} ---")
        print(prompt)
        print()
    
    print("\nRUNNING AUTOMATED TESTS:")
    print("=" * 70)
    
    results = evaluator.run_all_tests()
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    for test_name, test_results in results.items():
        print(f"\n{test_name}:")
        if isinstance(test_results, dict):
            for key, value in test_results.items():
                if isinstance(value, (int, float, str, bool)):
                    print(f"  {key}: {value}")
    
    evaluator.save_evaluation_report(results, "evaluation_report.json")
    
    return results


if __name__ == "__main__":
    run_interactive_test()
