# 🌲 Wilderness Navigator

A sophisticated text-based spatial reasoning game designed to evaluate LLM capabilities in instruction following, spatial reasoning, macro composition, and failure recovery.

## 🎯 Overview

Wilderness Navigator is a 2D grid-based exploration game where players must navigate from a starting position to a goal while discovering and utilizing emergent game mechanics. The game is specifically designed to test:

- **Instruction Following**: Correctly parsing and executing complex commands
- **Spatial Reasoning**: Planning routes, understanding coordinates and angles
- **Rule Discovery**: Learning mechanics through exploration and experimentation  
- **Macro Composition**: Creating sequential and recursive action chains
- **Failure Recovery**: Adapting strategy when actions fail
- **Context Maintenance**: Tracking state across long action sequences

## 🗺️ Game Mechanics

### Terrain Types

```
.  = GRASS     (walkable)
T  = TREE      (special: can CLIMB)
~  = RIVER     (special: use FLOW)
R  = ROCK      (obstacle)
M  = MOUNTAIN  (obstacle)
C  = CLIFF     (special: requires HOOK or CLIMB)
B  = BRIDGE    (walkable, crosses river)
b  = BUSH      (special: can SEARCH)
m  = MUD       (walkable, costs 2 moves)
G  = GOAL      (win condition)
@/↑/→/↓/← = PLAYER (with direction indicator)
```

### Basic Commands

#### Movement
```
MOVE <direction>
  - MOVE N/S/E/W          (cardinal directions)
  - MOVE FORWARD/BACK     (relative to facing angle)
  - MOVE LEFT/RIGHT       (strafe relative to facing)

TURN <direction>
  - TURN LEFT             (rotate -90°)
  - TURN RIGHT            (rotate +90°)
  - TURN <angle>          (set specific angle: 0°=N, 90°=E, 180°=S, 270°=W)
```

#### Utility
```
LOOK                      (inspect surroundings, get hints)
HELP                      (show available commands)
```

### 🔓 Discoverable Mechanics

These mechanics are revealed through exploration and experimentation:

#### 1. Tree Climbing
```
CLIMB                     (must be at TREE tile)
```
- Climb the tree and jump 3 tiles in your current facing direction
- Landing spot must be passable terrain
- Costs 1 move
- **Discovery trigger**: Standing on TREE tile + using LOOK

#### 2. River Flow
```
FLOW <n>                  (must be in/on RIVER or BRIDGE)
```
- Ride the river current for n tiles (river flows East)
- All tiles in path must be RIVER or BRIDGE
- Range: 1-10 tiles
- Costs 1 move (efficient for long distances)
- **Discovery trigger**: Attempting to walk on RIVER

#### 3. Axe Grappling Hook
```
HOOK <x> <y>              (requires axe in inventory)
```
- Throw axe as grappling hook to target coordinates
- Max range: 5 tiles
- Requires line of sight (not blocked by mountains/rocks)
- Target must be passable terrain
- Costs 1 move
- **Discovery trigger**: Having axe in inventory + using LOOK

#### 4. Bush Searching
```
SEARCH                    (must be at BUSH tile)
```
- Search bush for items (compass, rope, map fragments)
- Has cooldown (3 moves) per bush
- Random item drops
- Costs 1 move
- **Discovery trigger**: Standing on BUSH tile + using LOOK

#### 5. Macro Commands (Sequential)
```
MACRO [action1; action2; action3; ...]
```
- Execute a sequence of actions
- Stops immediately on first failure
- All standard commands can be used
- Example: `MACRO [MOVE E; MOVE E; TURN RIGHT; MOVE S]`
- **Discovery trigger**: Through experimentation or HELP

#### 6. Patrol Commands (Recursive)
```
PATROL [action1; action2; ...] <n>
```
- Repeat a sequence of actions n times
- Stops entire patrol on any failure
- Range: 1-20 repetitions
- Example: `PATROL [MOVE E; TURN RIGHT] 4`
- **Discovery trigger**: Through experimentation or HELP

### 📜 Game Rules

1. **Boundaries**: Cannot move outside map (typically 20x15 grid)
2. **Obstacles**: MOUNTAIN, ROCK, and CLIFF tiles are impassable by walking
3. **Rivers**: Cannot walk on RIVER tiles (must use FLOW or find BRIDGE)
4. **Mud Penalty**: Moving through MUD costs 2 moves instead of 1
5. **Angle System**: 0°=North, 90°=East, 180°=South, 270°=West
6. **Line of Sight**: HOOK requires unobstructed view (MOUNTAIN/ROCK block)
7. **Failure Counting**: Invalid actions increment failure counter

## 🎮 Usage

### Running the Game (Human Player)

```bash
python wilderness_navigator.py
```

**Interactive Commands:**
- Type any game command to play
- `STATE` - View full game state JSON
- `METRICS` - View performance statistics
- `QUIT` - Exit game

### Running LLM Evaluation

```bash
python llm_evaluator.py
```

This will:
1. Display test prompts for LLM evaluation
2. Run automated test suites
3. Generate evaluation report (`evaluation_report.json`)

### Programmatic Usage

```python
from wilderness_navigator import WildernessNavigator

# Create game instance
game = WildernessNavigator(width=20, height=15)

# Execute actions
result = game.execute_action("MOVE E")
print(result["message"])

# Render map
print(game.render_map())

# Get state
state = game.get_state_summary()

# Get metrics
metrics = game.get_performance_metrics()
```

## 📊 Evaluation Metrics

The game tracks comprehensive performance metrics:

### Efficiency Metrics
- **Moves Made**: Total number of moves executed
- **Failures**: Number of failed actions
- **Success Rate**: Percentage of successful actions
- **Efficiency Score**: 0-100 score based on optimal path, failures, and rule discovery

### Exploration Metrics
- **Rules Discovered**: Number of special mechanics found (0-10)
- **Tiles Explored**: Number of unique tiles visited
- **Exploration Percentage**: Percentage of map explored

### Optimal Performance Benchmark
- **Minimum Moves**: ~25 moves (requires discovering special mechanics)
- **Maximum Score**: 100 (efficient path + all rules discovered + no failures)

## 🧪 LLM Evaluation Test Suites

### 1. Basic Navigation
Tests understanding of:
- Cardinal directions (N/S/E/W)
- Relative directions (FORWARD/BACK/LEFT/RIGHT)
- Turn mechanics

### 2. Rule Discovery
Tests ability to:
- Use LOOK command for hints
- Experiment with new terrain types
- Discover hidden mechanics through trial

### 3. Spatial Reasoning
Tests ability to:
- Calculate distances and paths
- Navigate around obstacles
- Use coordinate system
- Plan multi-step routes

### 4. Macro Composition
Tests ability to:
- Create sequential action chains
- Use MACRO syntax correctly
- Compose recursive PATROL commands
- Handle macro failures gracefully

### 5. Failure Recovery
Tests ability to:
- Recognize failure conditions
- Adapt strategy after failures
- Find alternative paths
- Recover from invalid commands

### 6. Optimal Pathfinding
Tests ability to:
- Discover and utilize special mechanics
- Plan efficient routes
- Minimize move count
- Chain multiple mechanics

### 7. Context Memory
Tests ability to:
- Maintain position awareness over long sequences
- Track facing direction through turns
- Remember discovered rules
- Build on previous actions

### 8. Instruction Compliance
Tests ability to:
- Follow multi-step natural language instructions
- Achieve specific goals
- Stay within constraints
- Interpret complex requirements

## 🎯 Example Gameplay

### Beginner Path (No Special Mechanics)
```
Starting at (1, 1), goal at (18, 13)

MOVE E (x17)      # Move to x=18
MOVE S (x12)      # Move to y=13
Total: 29 moves
```

### Intermediate Path (Using River)
```
MOVE E (x9)       # Get to river
FLOW 8            # Ride river east
MOVE S (x12)      # Move to goal
Total: 22 moves
```

### Advanced Path (Using All Mechanics)
```
MOVE E (x4)       # Get to tree at (5, 3)
MOVE S (x2)
TURN 90           # Face East
CLIMB             # Jump to (8, 3)
MOVE S (x4)       # Get to bridge
FLOW 8            # Ride river
HOOK 18 13        # Grappling hook to goal
Total: ~17 moves
```

## 📝 Sample LLM Prompts

### Prompt 1: Basic Challenge
```
You are playing Wilderness Navigator. You start at (1,1) facing North.
The goal G is at (18,13). What are your first 5 moves?
```

### Prompt 2: Discovery Challenge
```
You are at a TREE at (5,3). You've never seen a tree before.
Use LOOK to get hints, then experiment to discover what you can do.
Provide your action sequence.
```

### Prompt 3: Planning Challenge
```
Plan the most efficient route from (1,1) to (18,13).
You know these mechanics:
- CLIMB: jump 3 tiles forward from tree
- FLOW n: ride river n tiles east
- HOOK x y: grapple within 5 tiles

Trees at: (5,3), (15,5)
River: vertical line at x=10 with bridge at y=7

Provide your complete action sequence.
```

### Prompt 4: Recovery Challenge
```
You tried MOVE N and got "Cannot move outside map boundaries!"
You are at (7,0) facing North. Goal is at (18,13).
What is your recovery strategy? Give next 3 commands.
```

### Prompt 5: Macro Challenge
```
Create a MACRO that efficiently explores the area in a spiral pattern:
Move East 3, South 3, West 3, North 3, then repeat moving outward.
Write the command.
```

## 🏆 Scoring System

### Performance Grades

**S Tier (90-100)**: 
- Optimal path found
- All mechanics discovered
- <5 failures
- High exploration

**A Tier (80-89)**:
- Near-optimal path
- Most mechanics discovered
- <10 failures
- Good exploration

**B Tier (70-79)**:
- Suboptimal but valid path
- Some mechanics discovered
- <15 failures
- Moderate exploration

**C Tier (60-69)**:
- Valid path completed
- Few mechanics discovered
- <25 failures
- Basic exploration

**F Tier (<60)**:
- Excessive failures
- Inefficient path
- Poor exploration

## 🔬 Research Applications

This game is designed to evaluate:

1. **Instruction Following Capability**
   - Parse complex syntax (MACRO, PATROL)
   - Execute multi-step commands
   - Handle command chaining

2. **Spatial Reasoning**
   - Mental map building
   - Coordinate system understanding
   - Angle/direction calculation
   - Pathfinding algorithms

3. **Learning and Adaptation**
   - Rule discovery through experimentation
   - Incorporating new knowledge
   - Strategy adjustment

4. **Planning and Optimization**
   - Multi-step planning
   - Resource optimization (move count)
   - Constraint satisfaction

5. **Error Handling**
   - Failure detection
   - Recovery strategies
   - Alternative path finding

## 📄 Files

- `wilderness_navigator.py` - Main game engine
- `llm_evaluator.py` - LLM testing suite
- `README.md` - This documentation
- `evaluation_report.json` - Generated test results

## 🚀 Future Enhancements

Potential additions:
- [ ] Procedurally generated maps
- [ ] Difficulty levels (more complex rules)
- [ ] Multi-agent gameplay
- [ ] Time-based challenges
- [ ] Inventory management system
- [ ] Dynamic weather/terrain changes
- [ ] Achievement system
- [ ] Replay/visualization system

## 📜 License

MIT License - Free to use for research and evaluation purposes.

## 🙏 Acknowledgments

Inspired by:
- Classic text adventures (Zork, Colossal Cave)
- DND exploration mechanics
- Chess tactical thinking
- Maze pathfinding algorithms
- Real-world orienteering

---

**Happy Navigating! 🧭**
