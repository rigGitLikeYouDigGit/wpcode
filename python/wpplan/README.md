# wpplan - Multi-Goal Task Planning and Exploration

A stochastic planning system for agents in state-based worlds with:
- Property-based action discovery
- Leverage/influence heuristics
- Mixed discrete/continuous state
- Emergent strategy learning
- Curiosity-driven exploration

## Architecture

### Core Components

**State** (`state.py`)
- `State`: Complete world state with objects and global variables
- `ObjectState`: Individual object with properties and attributes
- `Variable`: Discrete or continuous state variable
- Supports dependencies between variables (e.g., object location depends on container location)

**Actions** (`action.py`)
- `ActionTemplate`: Generic action definition with parameters
- Property requirements determine applicability (e.g., "PutIn" requires Container property)
- Preconditions, effects, and cost functions
- Actions instantiated dynamically based on available objects

**Goals** (`goal.py`)
- `Goal`: Target condition with satisfaction function (0.0-1.0)
- Supports priorities and deadlines
- Goals can be abstract (e.g., "≥3 letters at Work", not specific letters)

**Influence** (`influence.py`)
- `InfluenceGraph`: Tracks variable dependencies and goal relevance
- **Leverage**: How many goal-relevant variables does an action affect?
- **Empowerment**: How many future states become reachable?
- Used to prioritize actions that create "fulcrum" effects

**Knowledge** (`knowledge.py`)
- `BeliefState`: Agent's potentially incomplete/incorrect view of world
- `KnowledgeBase`: Known action templates and learned patterns
- Tracks uncertainty and exploration targets (e.g., unobserved containers)
- Records discovery history for learning

**Planner** (`planner.py`)
- Stochastic beam search with weighted sampling
- Combines multiple heuristics:
  - α: Goal progress (direct advancement)
  - β: Leverage (influence on future actions)
  - γ: Information gain (knowledge discovery)
  - δ: Discovery potential (learned patterns)
- Temperature parameter controls exploration vs exploitation
- Avoids deterministic local minima through stochastic selection

## Key Design Principles

### 1. Property-Based Actions
Actions are generic templates applicable to any objects with required properties:
```python
# PutIn template requires Container property
putin = ActionTemplate("PutIn", ["container", "object"])
putin.add_property_requirement("container", ["Container"])

# Automatically applicable to ANY container: Bag, Box, Locker...
```

### 2. Leverage as Search Heuristic
Actions that affect high-leverage variables are prioritized:
```python
# PutIn(Bag, Letter) creates dependency: Letter.location depends on Bag.location
# Now actions affecting Bag have 1x leverage

# After PutIn(Bag, Letter1), PutIn(Bag, Letter2), PutIn(Bag, Letter3)...
# Actions affecting Bag now have 3x leverage (affect 3 letters)
```

### 3. Stochastic Selection
Instead of always choosing best action, sample weighted by value:
```python
weights = softmax([action.value for action in actions], temperature)
selected = sample(actions, weights)

# Prevents getting stuck in local minima
# Higher temperature = more exploration
```

### 4. Emergent Strategy Learning
Agents don't have hardcoded strategies. They:
- Try different approaches
- Track which work in which contexts
- Adjust heuristic weights based on experience
- Develop specializations based on task distribution

### 5. Knowledge Discovery
Information gain is a first-class concern:
```python
# Unobserved container → curiosity goal
if obj.has_property("Container") and not observed_contents:
    exploration_targets.add(obj)

# Learn discovery patterns
# "Opening doors reveals more than drawers on average"
# "High elevation → high discovery rate"
```

## Example: Letter Transportation

**Setup:**
- Agent at Home
- 5 Letters at Home
- 1 Bag at Home (has Container property)
- Goal: 3 letters at Work
- Constraint: Agent can carry 1 object

**Naive Plan:**
1. PickUp(Letter1), GoToWork, PutDown(Letter1), GoToHome
2. PickUp(Letter2), GoToWork, PutDown(Letter2), GoToHome
3. PickUp(Letter3), GoToWork, PutDown(Letter3)
Cost: 6 trips × 10 = 60

**Discovered Plan (via leverage):**
1. PutIn(Bag, Letter1)  # Bag.leverage increases
2. PutIn(Bag, Letter2)  # Bag.leverage increases
3. PutIn(Bag, Letter3)  # Bag.leverage = 3
4. PickUp(Bag)          # Now affects 3 letters!
5. GoToWork             # High-leverage action
6. PutDown(Bag)
7. Extract letters
Cost: 1 trip × 10 + setup ≈ 20

The planner discovers the efficient solution because:
- PutIn actions increase Bag's leverage
- High-leverage actions (moving Bag) are prioritized
- Cost/benefit ratio favors the container strategy

## Usage

```python
from wpplan import State, ActionTemplate, Goal, Planner

# Create world state
state = State()
# ... add objects ...

# Define action templates
templates = [...]
kb = KnowledgeBase()
for t in templates:
    kb.add_action_template(t)

# Define goals
goal = Goal("description", satisfaction_fn)
goals = GoalSet()
goals.add_goal(goal)

# Plan
planner = Planner(kb, influence_graph)
plan = planner.plan(state, goals)
```

## Future Extensions

- **Hierarchical planning**: Discovered macro-actions
- **Multi-agent**: Social influence, delegation
- **Continuous planning**: Replanning as world changes
- **Meta-learning**: Transfer strategies across domains
- **Explanation**: Why did agent choose this plan?

## Status

🚧 **Early prototype** - Core architecture defined, not yet fully implemented.

Next steps:
1. Complete action instantiation logic
2. Implement influence propagation
3. Add more sophisticated state copying
4. Build out learning mechanisms
5. Test on letter example
