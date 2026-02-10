#!/usr/bin/env python3
"""
Demo script showing example gameplay scenarios
"""

from wilderness_navigator import WildernessNavigator


def demo_basic_movement():
    """Demo 1: Basic movement"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Movement & Navigation")
    print("="*70)
    
    game = WildernessNavigator(width=20, height=15)
    print(game.render_map())
    
    actions = [
        ("MOVE E", "Moving east"),
        ("MOVE E", "Continue east"),
        ("TURN RIGHT", "Turn to face south"),
        ("MOVE FORWARD", "Move in facing direction (south)"),
        ("LOOK", "Inspect surroundings")
    ]
    
    for action, description in actions:
        print(f"\n➤ {description}: {action}")
        result = game.execute_action(action)
        print(f"   Result: {result['message']}")
    
    print("\n" + game.render_map())
    print(f"\nMoves made: {game.state.moves_made}")


def demo_rule_discovery():
    """Demo 2: Discovering tree climbing"""
    print("\n" + "="*70)
    print("DEMO 2: Rule Discovery - Tree Climbing")
    print("="*70)
    
    game = WildernessNavigator(width=20, height=15)
    
    # Navigate to tree
    setup = ["MOVE E"] * 4 + ["MOVE S"] * 2
    for action in setup:
        game.execute_action(action)
    
    print(game.render_map())
    print("\n🌲 Standing on a TREE tile at (5, 3)")
    
    # Try to discover mechanic
    print("\n➤ Using LOOK to get hints")
    result = game.execute_action("LOOK")
    print(result['message'])
    
    print("\n➤ Attempting to CLIMB")
    result = game.execute_action("CLIMB")
    print(result['message'])
    
    if result.get('rule_discovered'):
        print(f"\n🎯 DISCOVERED: {game.rules[result['rule_discovered']]['description']}")
    
    print("\n" + game.render_map())


def demo_river_mechanics():
    """Demo 3: River and flow mechanic"""
    print("\n" + "="*70)
    print("DEMO 3: River Flow Mechanic")
    print("="*70)
    
    game = WildernessNavigator(width=20, height=15)
    
    # Navigate to river
    setup = ["MOVE E"] * 9 + ["MOVE S"] * 6
    for action in setup:
        game.execute_action(action)
    
    print(game.render_map())
    print("\n🌊 At river/bridge at (10, 7)")
    
    print("\n➤ Attempting to move onto river")
    result = game.execute_action("MOVE E")
    print(result['message'])
    
    print("\n➤ Using FLOW to ride the current")
    result = game.execute_action("FLOW 5")
    print(result['message'])
    
    if result.get('rule_discovered'):
        print(f"\n🎯 DISCOVERED: {game.rules[result['rule_discovered']]['description']}")
    
    print("\n" + game.render_map())


def demo_macro_commands():
    """Demo 4: Using macros"""
    print("\n" + "="*70)
    print("DEMO 4: Macro Commands")
    print("="*70)
    
    game = WildernessNavigator(width=20, height=15)
    print(game.render_map())
    
    print("\n➤ Executing MACRO to move in a pattern")
    result = game.execute_action("MACRO [MOVE E; MOVE E; TURN RIGHT; MOVE S; MOVE S]")
    print(result['message'])
    
    print("\n" + game.render_map())
    
    print("\n➤ Using PATROL to repeat a pattern")
    result = game.execute_action("PATROL [MOVE E; TURN RIGHT] 2")
    print(result['message'])
    
    print("\n" + game.render_map())


def demo_grappling_hook():
    """Demo 5: Using the axe as a grappling hook"""
    print("\n" + "="*70)
    print("DEMO 5: Axe Grappling Hook")
    print("="*70)
    
    game = WildernessNavigator(width=20, height=15)
    
    # Move to a good position
    setup = ["MOVE E"] * 5 + ["MOVE S"] * 3
    for action in setup:
        game.execute_action(action)
    
    print(game.render_map())
    print(f"\n🪓 Current position: {game.state.player_pos}")
    print(f"   Inventory: {game.state.inventory}")
    
    print("\n➤ Using HOOK to grapple to a distant tile")
    result = game.execute_action("HOOK 10 5")
    print(result['message'])
    
    if result.get('rule_discovered'):
        print(f"\n🎯 DISCOVERED: {game.rules[result['rule_discovered']]['description']}")
    
    print("\n" + game.render_map())


def demo_complete_solution():
    """Demo 6: Complete efficient solution"""
    print("\n" + "="*70)
    print("DEMO 6: Complete Efficient Solution")
    print("="*70)
    
    game = WildernessNavigator(width=20, height=15)
    print("Starting position:", game.state.player_pos)
    print("Goal position:", game.goal_pos)
    print()
    
    # Optimal path using multiple mechanics
    solution = [
        ("MOVE E", "Start moving east"),
        ("MOVE E", ""),
        ("MOVE E", ""),
        ("MOVE E", "Get to tree area"),
        ("MOVE S", ""),
        ("MOVE S", "Now at tree (5, 3)"),
        ("TURN 90", "Face East"),
        ("CLIMB", "Jump 3 tiles forward to (8, 3)"),
        ("MOVE S", "Move toward bridge"),
        ("MOVE S", ""),
        ("MOVE S", ""),
        ("MOVE S", "Now at bridge (10, 7)"),
        ("MOVE E", "Cross bridge"),
        ("FLOW 7", "Ride river to (18, 7)"),
        ("MOVE S", "Move to goal"),
        ("MOVE S", ""),
        ("MOVE S", ""),
        ("MOVE S", ""),
        ("MOVE S", ""),
        ("MOVE S", "Reached goal!")
    ]
    
    for i, (action, desc) in enumerate(solution, 1):
        if desc:
            print(f"\n{i}. {desc}: {action}")
        result = game.execute_action(action)
        if not result['success']:
            print(f"   ❌ {result['message']}")
            break
        
        if result.get('rule_discovered'):
            print(f"   🎯 Discovered: {result['rule_discovered']}")
        
        if result.get('won'):
            print("\n" + "🎉"*35)
            print(game.render_map())
            print("\n" + "🎉"*35)
            print("\n✨ VICTORY! ✨")
            metrics = game.get_performance_metrics()
            print(f"\n📊 Final Statistics:")
            print(f"   Moves: {metrics['moves_made']}")
            print(f"   Failures: {metrics['failures']}")
            print(f"   Rules Discovered: {metrics['rules_discovered']}/10")
            print(f"   Efficiency Score: {metrics['efficiency_score']:.1f}/100")
            break


def run_all_demos():
    """Run all demo scenarios"""
    demos = [
        demo_basic_movement,
        demo_rule_discovery,
        demo_river_mechanics,
        demo_macro_commands,
        demo_grappling_hook,
        demo_complete_solution
    ]
    
    for demo in demos:
        demo()
        input("\nPress Enter to continue to next demo...")


if __name__ == "__main__":
    print("\n" + "🎮"*35)
    print(" "*20 + "WILDERNESS NAVIGATOR")
    print(" "*25 + "DEMO SUITE")
    print("🎮"*35)
    
    run_all_demos()
    
    print("\n" + "="*70)
    print("All demos complete! Try the game yourself:")
    print("  python3 wilderness_navigator.py")
    print("="*70)
