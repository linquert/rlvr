#!/usr/bin/env python3
"""
WILDERNESS NAVIGATOR
A text-based spatial reasoning game for evaluating LLM capabilities

Game Features:
- Emergent rule discovery through exploration
- Complex movement mechanics (trees, rivers, axe-hook)
- Spatial reasoning and path planning
- Macro/sequential action composition
- Failure recovery and rule compliance
"""

import json
import math
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass, asdict
import copy


class Terrain(Enum):
    GRASS = "."
    TREE = "T"
    RIVER = "~"
    ROCK = "R"
    MOUNTAIN = "M"
    GOAL = "G"
    PLAYER = "@"
    CLIFF = "C"
    BRIDGE = "B"
    BUSH = "b"
    MUD = "m"


class Direction(Enum):
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)
    NORTHEAST = (1, -1)
    NORTHWEST = (-1, -1)
    SOUTHEAST = (1, 1)
    SOUTHWEST = (-1, 1)


@dataclass
class GameState:
    player_pos: Tuple[int, int]
    player_angle: int  # 0=North, 90=East, 180=South, 270=West
    moves_made: int
    discovered_rules: Set[str]
    inventory: Dict[str, int]
    visited_tiles: Set[Tuple[int, int]]
    last_action: str
    action_history: List[str]
    failures: int
    
    def to_dict(self):
        return {
            'player_pos': self.player_pos,
            'player_angle': self.player_angle,
            'moves_made': self.moves_made,
            'discovered_rules': list(self.discovered_rules),
            'inventory': self.inventory,
            'visited_tiles': [list(t) for t in self.visited_tiles],
            'last_action': self.last_action,
            'action_history': self.action_history[-10:],  # Last 10 actions
            'failures': self.failures
        }


class WildernessNavigator:
    def __init__(self, width=20, height=15, difficulty="medium"):
        self.width = width
        self.height = height
        self.map = [[Terrain.GRASS for _ in range(width)] for _ in range(height)]
        self.difficulty = difficulty
        
        # Initialize game state
        self.state = GameState(
            player_pos=(1, 1),
            player_angle=0,  # Facing North
            moves_made=0,
            discovered_rules=set(),
            inventory={"axe": 1},
            visited_tiles={(1, 1)},
            last_action="",
            action_history=[],
            failures=0
        )
        
        # Generate procedural map
        self._generate_map()
        
        # Rule discovery system
        self.rules = {
            "tree_climb": {
                "discovered": False,
                "hint": "Trees seem sturdy...",
                "description": "CLIMB at tree: Jump 3 tiles forward (relative to your angle)"
            },
            "river_flow": {
                "discovered": False,
                "hint": "The river flows swiftly...",
                "description": "FLOW <n> in river: Move n tiles with current (river flows East)"
            },
            "axe_hook": {
                "discovered": False,
                "hint": "Your axe could be useful for more than chopping...",
                "description": "HOOK <x> <y>: Use axe as grappling hook (max 5 tiles, line of sight)"
            },
            "rotation": {
                "discovered": False,
                "hint": "Direction matters in the wilderness...",
                "description": "TURN <LEFT|RIGHT|angle>: Change facing direction"
            },
            "bridge_crossing": {
                "discovered": False,
                "hint": "Bridges connect what rivers divide...",
                "description": "Bridges allow crossing rivers without using FLOW"
            },
            "cliff_danger": {
                "discovered": False,
                "hint": "Mind your step near edges...",
                "description": "CLIFF tiles: Cannot walk on; must HOOK or CLIMB over"
            },
            "bush_search": {
                "discovered": False,
                "hint": "Bushes might hide something...",
                "description": "SEARCH at bush: May find items (has cooldown)"
            },
            "mud_slow": {
                "discovered": False,
                "hint": "Muddy ground feels heavy...",
                "description": "MUD tiles: Each step costs 2 moves instead of 1"
            },
            "macro_commands": {
                "discovered": False,
                "hint": "Complex journeys need planning...",
                "description": "MACRO [action1; action2; ...]: Execute sequence (stops on failure)"
            },
            "recursive_patrol": {
                "discovered": False,
                "hint": "Patterns repeat in nature...",
                "description": "PATROL [actions] <n>: Repeat action sequence n times"
            }
        }
        
        self.search_cooldown = {}  # Track bush search cooldowns
        
    def _generate_map(self):
        """Generate a procedural wilderness map"""
        import random
        random.seed(42)  # Consistent map for testing
        
        # Place goal
        self.goal_pos = (self.width - 2, self.height - 2)
        self.map[self.goal_pos[1]][self.goal_pos[0]] = Terrain.GOAL
        
        # Create river (vertical)
        river_x = self.width // 2
        for y in range(3, self.height - 3):
            self.map[y][river_x] = Terrain.RIVER
            
        # Add bridge over river
        bridge_y = self.height // 2
        self.map[bridge_y][river_x] = Terrain.BRIDGE
        
        # Scatter trees
        tree_positions = [(5, 3), (8, 7), (15, 5), (12, 10), (3, 9)]
        for x, y in tree_positions:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.map[y][x] = Terrain.TREE
                
        # Add mountains (obstacles)
        mountain_positions = [(7, 2), (13, 8), (6, 12)]
        for x, y in mountain_positions:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.map[y][x] = Terrain.MOUNTAIN
                
        # Add cliffs
        for x in range(self.width // 3, self.width // 3 + 3):
            if 0 <= x < self.width:
                self.map[6][x] = Terrain.CLIFF
                
        # Add bushes
        bush_positions = [(4, 5), (11, 4), (16, 9)]
        for x, y in bush_positions:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.map[y][x] = Terrain.BUSH
                
        # Add mud patches
        mud_positions = [(9, 9), (10, 9), (9, 10), (14, 12)]
        for x, y in mud_positions:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.map[y][x] = Terrain.MUD
                
        # Add rocks (obstacles)
        rock_positions = [(2, 4), (17, 7)]
        for x, y in rock_positions:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.map[y][x] = Terrain.ROCK
    
    def render_map(self, show_fog=True) -> str:
        """Render the current map state with optional fog of war"""
        output = []
        output.append("=" * (self.width + 2))
        
        # Add coordinate header
        header = " "
        for x in range(self.width):
            header += str(x % 10)
        output.append(header)
        
        for y in range(self.height):
            row = str(y % 10)
            for x in range(self.width):
                if (x, y) == self.state.player_pos:
                    # Show player with directional indicator
                    direction_symbols = {0: "↑", 90: "→", 180: "↓", 270: "←"}
                    row += direction_symbols.get(self.state.player_angle, "@")
                elif show_fog and (x, y) not in self.state.visited_tiles:
                    # Calculate if in vision range (3 tiles)
                    px, py = self.state.player_pos
                    if abs(x - px) <= 3 and abs(y - py) <= 3:
                        row += self.map[y][x].value
                        self.state.visited_tiles.add((x, y))
                    else:
                        row += " "
                else:
                    row += self.map[y][x].value
            output.append(row)
            
        output.append("=" * (self.width + 2))
        
        # Add status info
        output.append(f"\nPosition: ({self.state.player_pos[0]}, {self.state.player_pos[1]}) | "
                     f"Facing: {self.state.player_angle}° | Moves: {self.state.moves_made}")
        output.append(f"Inventory: {self.state.inventory}")
        output.append(f"Rules Discovered: {len(self.state.discovered_rules)}/10")
        
        if self.state.last_action:
            output.append(f"Last Action: {self.state.last_action}")
            
        return "\n".join(output)
    
    def get_terrain_at(self, x: int, y: int) -> Optional[Terrain]:
        """Get terrain type at position"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.map[y][x]
        return None
    
    def _angle_to_direction(self, angle: int) -> Tuple[int, int]:
        """Convert angle to direction vector"""
        # Normalize angle
        angle = angle % 360
        
        if angle == 0:
            return (0, -1)  # North
        elif angle == 45:
            return (1, -1)  # Northeast
        elif angle == 90:
            return (1, 0)   # East
        elif angle == 135:
            return (1, 1)   # Southeast
        elif angle == 180:
            return (0, 1)   # South
        elif angle == 225:
            return (-1, 1)  # Southwest
        elif angle == 270:
            return (-1, 0)  # West
        elif angle == 315:
            return (-1, -1) # Northwest
        else:
            # For non-standard angles, approximate
            rad = math.radians(angle)
            return (round(math.sin(rad)), round(-math.cos(rad)))
    
    def _is_passable(self, x: int, y: int) -> bool:
        """Check if a tile can be walked on"""
        terrain = self.get_terrain_at(x, y)
        if terrain is None:
            return False
        return terrain not in [Terrain.MOUNTAIN, Terrain.ROCK, Terrain.CLIFF, Terrain.RIVER]
    
    def _line_of_sight(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if there's a clear line of sight between two points"""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2
        
        for _ in range(n):
            if (x, y) != (x1, y1):  # Don't check starting position
                terrain = self.get_terrain_at(x, y)
                if terrain in [Terrain.MOUNTAIN, Terrain.ROCK]:
                    return False
            
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
                
        return True
    
    def execute_action(self, action: str) -> Dict:
        """Execute a game action and return result"""
        action = action.strip().upper()
        self.state.action_history.append(action)
        self.state.last_action = action
        
        result = {
            "success": False,
            "message": "",
            "rule_discovered": None,
            "game_over": False,
            "won": False
        }
        
        try:
            # Parse action
            parts = action.split()
            command = parts[0] if parts else ""
            
            # Handle different commands
            if command == "MOVE" or command == "M":
                result = self._handle_move(parts)
            elif command == "TURN" or command == "T":
                result = self._handle_turn(parts)
            elif command == "CLIMB":
                result = self._handle_climb(parts)
            elif command == "FLOW":
                result = self._handle_flow(parts)
            elif command == "HOOK":
                result = self._handle_hook(parts)
            elif command == "SEARCH":
                result = self._handle_search(parts)
            elif command == "MACRO":
                result = self._handle_macro(parts)
            elif command == "PATROL":
                result = self._handle_patrol(parts)
            elif command == "LOOK":
                result = self._handle_look(parts)
            elif command == "HELP":
                result = self._handle_help(parts)
            else:
                result["message"] = f"Unknown command: {command}. Try HELP for available commands."
                self.state.failures += 1
                
        except Exception as e:
            result["message"] = f"Error executing action: {str(e)}"
            result["success"] = False
            self.state.failures += 1
        
        # Check win condition
        if self.state.player_pos == self.goal_pos:
            result["won"] = True
            result["game_over"] = True
            result["message"] += "\n🎉 YOU REACHED THE GOAL! 🎉"
            
        return result
    
    def _handle_move(self, parts: List[str]) -> Dict:
        """Handle MOVE command"""
        if len(parts) < 2:
            return {"success": False, "message": "MOVE requires direction: MOVE <FORWARD|BACK|LEFT|RIGHT|N|S|E|W>"}
        
        direction = parts[1]
        dx, dy = 0, 0
        
        # Parse direction
        if direction in ["FORWARD", "F"]:
            dx, dy = self._angle_to_direction(self.state.player_angle)
        elif direction in ["BACK", "B"]:
            dx, dy = self._angle_to_direction((self.state.player_angle + 180) % 360)
        elif direction in ["LEFT", "L"]:
            dx, dy = self._angle_to_direction((self.state.player_angle - 90) % 360)
        elif direction in ["RIGHT", "R"]:
            dx, dy = self._angle_to_direction((self.state.player_angle + 90) % 360)
        elif direction == "N":
            dx, dy = 0, -1
        elif direction == "S":
            dx, dy = 0, 1
        elif direction == "E":
            dx, dy = 1, 0
        elif direction == "W":
            dx, dy = -1, 0
        else:
            return {"success": False, "message": f"Invalid direction: {direction}"}
        
        # Calculate new position
        new_x = self.state.player_pos[0] + dx
        new_y = self.state.player_pos[1] + dy
        
        # Check bounds
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            self.state.failures += 1
            return {"success": False, "message": "Cannot move outside map boundaries!"}
        
        # Check terrain
        terrain = self.get_terrain_at(new_x, new_y)
        
        if terrain == Terrain.RIVER:
            if self.get_terrain_at(*self.state.player_pos) != Terrain.BRIDGE:
                self.state.failures += 1
                return {"success": False, "message": "Cannot walk on river! Try FLOW or find a bridge.",
                       "rule_discovered": "river_flow"}
        
        if terrain in [Terrain.MOUNTAIN, Terrain.ROCK]:
            self.state.failures += 1
            return {"success": False, "message": f"Path blocked by {terrain.value}!"}
        
        if terrain == Terrain.CLIFF:
            self.state.failures += 1
            return {"success": False, "message": "Cannot walk on cliff! Try HOOK or CLIMB.",
                   "rule_discovered": "cliff_danger"}
        
        # Move player
        self.state.player_pos = (new_x, new_y)
        self.state.visited_tiles.add((new_x, new_y))
        
        # Calculate move cost
        move_cost = 2 if terrain == Terrain.MUD else 1
        self.state.moves_made += move_cost
        
        message = f"Moved to ({new_x}, {new_y})"
        if terrain == Terrain.MUD:
            message += " [Mud: cost 2 moves]"
            if "mud_slow" not in self.state.discovered_rules:
                self.state.discovered_rules.add("mud_slow")
                return {"success": True, "message": message, "rule_discovered": "mud_slow"}
        
        return {"success": True, "message": message}
    
    def _handle_turn(self, parts: List[str]) -> Dict:
        """Handle TURN command"""
        if len(parts) < 2:
            return {"success": False, "message": "TURN requires direction: TURN <LEFT|RIGHT|angle>"}
        
        direction = parts[1]
        
        if direction in ["LEFT", "L"]:
            self.state.player_angle = (self.state.player_angle - 90) % 360
        elif direction in ["RIGHT", "R"]:
            self.state.player_angle = (self.state.player_angle + 90) % 360
        else:
            try:
                angle = int(direction)
                self.state.player_angle = angle % 360
            except ValueError:
                return {"success": False, "message": f"Invalid turn direction: {direction}"}
        
        if "rotation" not in self.state.discovered_rules:
            self.state.discovered_rules.add("rotation")
            return {"success": True, "message": f"Now facing {self.state.player_angle}°",
                   "rule_discovered": "rotation"}
        
        return {"success": True, "message": f"Now facing {self.state.player_angle}°"}
    
    def _handle_climb(self, parts: List[str]) -> Dict:
        """Handle CLIMB command (from tree)"""
        px, py = self.state.player_pos
        terrain = self.get_terrain_at(px, py)
        
        if terrain != Terrain.TREE:
            self.state.failures += 1
            return {"success": False, "message": "You must be at a tree to CLIMB!",
                   "rule_discovered": "tree_climb"}
        
        # Jump 3 tiles forward based on angle
        dx, dy = self._angle_to_direction(self.state.player_angle)
        landing_x = px + dx * 3
        landing_y = py + dy * 3
        
        # Check bounds
        if not (0 <= landing_x < self.width and 0 <= landing_y < self.height):
            self.state.failures += 1
            return {"success": False, "message": "Would land outside map boundaries!"}
        
        # Check landing spot
        if not self._is_passable(landing_x, landing_y):
            landing_terrain = self.get_terrain_at(landing_x, landing_y)
            self.state.failures += 1
            return {"success": False, "message": f"Cannot land on {landing_terrain.value}!"}
        
        # Execute jump
        self.state.player_pos = (landing_x, landing_y)
        self.state.visited_tiles.add((landing_x, landing_y))
        self.state.moves_made += 1
        
        if "tree_climb" not in self.state.discovered_rules:
            self.state.discovered_rules.add("tree_climb")
            return {"success": True, 
                   "message": f"Climbed tree and jumped to ({landing_x}, {landing_y})!",
                   "rule_discovered": "tree_climb"}
        
        return {"success": True, "message": f"Jumped to ({landing_x}, {landing_y})!"}
    
    def _handle_flow(self, parts: List[str]) -> Dict:
        """Handle FLOW command (in river)"""
        px, py = self.state.player_pos
        terrain = self.get_terrain_at(px, py)
        
        if terrain not in [Terrain.RIVER, Terrain.BRIDGE]:
            self.state.failures += 1
            return {"success": False, "message": "You must be in/on river to use FLOW!",
                   "rule_discovered": "river_flow"}
        
        if len(parts) < 2:
            return {"success": False, "message": "FLOW requires distance: FLOW <n>"}
        
        try:
            distance = int(parts[1])
        except ValueError:
            return {"success": False, "message": "FLOW distance must be a number!"}
        
        if distance < 1 or distance > 10:
            return {"success": False, "message": "FLOW distance must be between 1 and 10!"}
        
        # River flows East
        new_x = px + distance
        new_y = py
        
        # Check if path is all river/bridge
        for check_x in range(px + 1, new_x + 1):
            check_terrain = self.get_terrain_at(check_x, new_y)
            if check_terrain not in [Terrain.RIVER, Terrain.BRIDGE]:
                self.state.failures += 1
                return {"success": False, 
                       "message": f"River ends before {distance} tiles! Would hit {check_terrain.value if check_terrain else 'edge'}."}
        
        # Execute flow
        self.state.player_pos = (new_x, new_y)
        for x in range(px, new_x + 1):
            self.state.visited_tiles.add((x, new_y))
        self.state.moves_made += 1
        
        if "river_flow" not in self.state.discovered_rules:
            self.state.discovered_rules.add("river_flow")
            return {"success": True, 
                   "message": f"Flowed {distance} tiles east to ({new_x}, {new_y})!",
                   "rule_discovered": "river_flow"}
        
        return {"success": True, "message": f"Flowed to ({new_x}, {new_y})!"}
    
    def _handle_hook(self, parts: List[str]) -> Dict:
        """Handle HOOK command (axe grappling hook)"""
        if "axe" not in self.state.inventory or self.state.inventory["axe"] < 1:
            return {"success": False, "message": "You don't have an axe!"}
        
        if len(parts) < 3:
            return {"success": False, "message": "HOOK requires target: HOOK <x> <y>"}
        
        try:
            target_x = int(parts[1])
            target_y = int(parts[2])
        except ValueError:
            return {"success": False, "message": "HOOK coordinates must be numbers!"}
        
        px, py = self.state.player_pos
        
        # Calculate distance
        distance = math.sqrt((target_x - px)**2 + (target_y - py)**2)
        
        if distance > 5:
            self.state.failures += 1
            return {"success": False, "message": "Target too far! HOOK max range is 5 tiles."}
        
        # Check line of sight
        if not self._line_of_sight(px, py, target_x, target_y):
            self.state.failures += 1
            return {"success": False, "message": "No line of sight to target!"}
        
        # Check target terrain
        if not self._is_passable(target_x, target_y):
            target_terrain = self.get_terrain_at(target_x, target_y)
            self.state.failures += 1
            return {"success": False, "message": f"Cannot hook to {target_terrain.value}!"}
        
        # Execute hook
        self.state.player_pos = (target_x, target_y)
        self.state.visited_tiles.add((target_x, target_y))
        self.state.moves_made += 1
        
        if "axe_hook" not in self.state.discovered_rules:
            self.state.discovered_rules.add("axe_hook")
            return {"success": True, 
                   "message": f"Hooked to ({target_x}, {target_y})!",
                   "rule_discovered": "axe_hook"}
        
        return {"success": True, "message": f"Hooked to ({target_x}, {target_y})!"}
    
    def _handle_search(self, parts: List[str]) -> Dict:
        """Handle SEARCH command (bushes)"""
        px, py = self.state.player_pos
        terrain = self.get_terrain_at(px, py)
        
        if terrain != Terrain.BUSH:
            self.state.failures += 1
            return {"success": False, "message": "You must be at a bush to SEARCH!",
                   "rule_discovered": "bush_search"}
        
        # Check cooldown
        if (px, py) in self.search_cooldown:
            self.state.failures += 1
            return {"success": False, "message": "This bush was recently searched (cooldown active)."}
        
        # Set cooldown (3 moves)
        self.search_cooldown[(px, py)] = self.state.moves_made + 3
        self.state.moves_made += 1
        
        # Random item find
        import random
        found_item = random.choice(["compass", "rope", "map_fragment", None])
        
        if found_item:
            self.state.inventory[found_item] = self.state.inventory.get(found_item, 0) + 1
            message = f"Found {found_item} in the bush!"
        else:
            message = "Searched but found nothing..."
        
        if "bush_search" not in self.state.discovered_rules:
            self.state.discovered_rules.add("bush_search")
            return {"success": True, "message": message, "rule_discovered": "bush_search"}
        
        return {"success": True, "message": message}
    
    def _handle_macro(self, parts: List[str]) -> Dict:
        """Handle MACRO command (sequential actions)"""
        if len(parts) < 2:
            return {"success": False, "message": "MACRO requires actions: MACRO [action1; action2; ...]"}
        
        # Join all parts and split by semicolon
        macro_str = " ".join(parts[1:])
        actions = [a.strip() for a in macro_str.split(";")]
        
        results = []
        for action in actions:
            if not action:
                continue
            result = self.execute_action(action)
            results.append(f"{action}: {result['message']}")
            
            if not result["success"]:
                # Stop on failure
                message = "MACRO stopped on failure:\n" + "\n".join(results)
                if "macro_commands" not in self.state.discovered_rules:
                    self.state.discovered_rules.add("macro_commands")
                    return {"success": False, "message": message, "rule_discovered": "macro_commands"}
                return {"success": False, "message": message}
        
        message = "MACRO completed:\n" + "\n".join(results)
        if "macro_commands" not in self.state.discovered_rules:
            self.state.discovered_rules.add("macro_commands")
            return {"success": True, "message": message, "rule_discovered": "macro_commands"}
        
        return {"success": True, "message": message}
    
    def _handle_patrol(self, parts: List[str]) -> Dict:
        """Handle PATROL command (recursive repetition)"""
        if len(parts) < 3:
            return {"success": False, "message": "PATROL requires: PATROL [actions] <n>"}
        
        # Find the repeat count (last part)
        try:
            repeat_count = int(parts[-1])
            action_parts = parts[1:-1]
        except ValueError:
            return {"success": False, "message": "PATROL requires a number for repeat count!"}
        
        if repeat_count < 1 or repeat_count > 20:
            return {"success": False, "message": "PATROL repeat count must be between 1 and 20!"}
        
        # Join actions and split by semicolon
        macro_str = " ".join(action_parts)
        actions = [a.strip() for a in macro_str.split(";")]
        
        results = []
        for i in range(repeat_count):
            iteration_results = []
            for action in actions:
                if not action:
                    continue
                result = self.execute_action(action)
                iteration_results.append(f"{action}: {result['message']}")
                
                if not result["success"]:
                    # Stop entire patrol on failure
                    message = f"PATROL stopped at iteration {i+1}:\n" + "\n".join(iteration_results)
                    if "recursive_patrol" not in self.state.discovered_rules:
                        self.state.discovered_rules.add("recursive_patrol")
                        return {"success": False, "message": message, "rule_discovered": "recursive_patrol"}
                    return {"success": False, "message": message}
            
            results.append(f"Iteration {i+1}: Success")
        
        message = f"PATROL completed {repeat_count} iterations"
        if "recursive_patrol" not in self.state.discovered_rules:
            self.state.discovered_rules.add("recursive_patrol")
            return {"success": True, "message": message, "rule_discovered": "recursive_patrol"}
        
        return {"success": True, "message": message}
    
    def _handle_look(self, parts: List[str]) -> Dict:
        """Handle LOOK command (inspect surroundings)"""
        px, py = self.state.player_pos
        terrain = self.get_terrain_at(px, py)
        
        message = f"You are standing on {terrain.name} at ({px}, {py}).\n"
        message += f"Facing: {self.state.player_angle}°\n\n"
        
        # Look at adjacent tiles
        message += "Adjacent tiles:\n"
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                check_x, check_y = px + dx, py + dy
                check_terrain = self.get_terrain_at(check_x, check_y)
                if check_terrain:
                    direction = ""
                    if dy == -1:
                        direction = "N"
                    elif dy == 1:
                        direction = "S"
                    if dx == -1:
                        direction += "W"
                    elif dx == 1:
                        direction += "E"
                    message += f"  {direction}: {check_terrain.name} ({check_terrain.value})\n"
        
        # Give hints based on current terrain
        for rule_name, rule_info in self.rules.items():
            if rule_name not in self.state.discovered_rules:
                if rule_name == "tree_climb" and terrain == Terrain.TREE:
                    message += f"\nHint: {rule_info['hint']}"
                elif rule_name == "river_flow" and terrain in [Terrain.RIVER, Terrain.BRIDGE]:
                    message += f"\nHint: {rule_info['hint']}"
                elif rule_name == "bush_search" and terrain == Terrain.BUSH:
                    message += f"\nHint: {rule_info['hint']}"
        
        return {"success": True, "message": message}
    
    def _handle_help(self, parts: List[str]) -> Dict:
        """Handle HELP command"""
        message = "WILDERNESS NAVIGATOR - Available Commands:\n\n"
        message += "Basic Movement:\n"
        message += "  MOVE <dir> - Move one tile (dir: FORWARD/BACK/LEFT/RIGHT/N/S/E/W)\n"
        message += "  TURN <dir> - Change facing (dir: LEFT/RIGHT/angle)\n\n"
        
        message += "Special Actions:\n"
        message += "  CLIMB - [Discovery required]\n"
        message += "  FLOW <n> - [Discovery required]\n"
        message += "  HOOK <x> <y> - [Discovery required]\n"
        message += "  SEARCH - [Discovery required]\n\n"
        
        message += "Advanced:\n"
        message += "  MACRO [action1; action2; ...] - [Discovery required]\n"
        message += "  PATROL [actions] <n> - [Discovery required]\n\n"
        
        message += "Utility:\n"
        message += "  LOOK - Inspect surroundings\n"
        message += "  HELP - Show this message\n\n"
        
        message += "Discovered Rules:\n"
        for rule_name in self.state.discovered_rules:
            if rule_name in self.rules:
                message += f"  ✓ {self.rules[rule_name]['description']}\n"
        
        return {"success": True, "message": message}
    
    def get_state_summary(self) -> str:
        """Get a JSON summary of current game state"""
        return json.dumps(self.state.to_dict(), indent=2)
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        return {
            "moves_made": self.state.moves_made,
            "failures": self.state.failures,
            "success_rate": (1 - self.state.failures / max(len(self.state.action_history), 1)) * 100,
            "rules_discovered": len(self.state.discovered_rules),
            "tiles_explored": len(self.state.visited_tiles),
            "exploration_percentage": (len(self.state.visited_tiles) / (self.width * self.height)) * 100,
            "efficiency_score": self._calculate_efficiency()
        }
    
    def _calculate_efficiency(self) -> float:
        """Calculate efficiency score (0-100)"""
        # Optimal path is approximately 25 moves
        optimal_moves = 25
        move_efficiency = max(0, (optimal_moves / max(self.state.moves_made, 1)) * 100)
        
        # Penalize failures
        failure_penalty = self.state.failures * 5
        
        # Reward rule discovery
        rule_bonus = len(self.state.discovered_rules) * 5
        
        return min(100, max(0, move_efficiency - failure_penalty + rule_bonus))


def main():
    """Main game loop for human testing"""
    print("=" * 60)
    print("WILDERNESS NAVIGATOR - Text-Based Spatial Reasoning Game")
    print("=" * 60)
    print("\nGoal: Navigate to the 'G' tile using discovered mechanics!")
    print("Type HELP for commands, LOOK to inspect surroundings.\n")
    
    game = WildernessNavigator(width=20, height=15)
    
    while True:
        print("\n" + game.render_map())
        
        action = input("\nEnter command: ").strip()
        
        if action.upper() == "QUIT":
            print("\nThanks for playing!")
            break
        
        if action.upper() == "STATE":
            print("\n" + game.get_state_summary())
            continue
        
        if action.upper() == "METRICS":
            metrics = game.get_performance_metrics()
            print("\nPerformance Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
            continue
        
        result = game.execute_action(action)
        print(f"\n{result['message']}")
        
        if result.get("rule_discovered"):
            rule_name = result["rule_discovered"]
            if rule_name in game.rules:
                print(f"\n🎯 NEW RULE DISCOVERED: {game.rules[rule_name]['description']}")
        
        if result.get("game_over"):
            print("\n" + game.render_map())
            metrics = game.get_performance_metrics()
            print("\nFinal Performance:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
            break


if __name__ == "__main__":
    main()
