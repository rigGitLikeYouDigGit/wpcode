# wpplan Architecture Design

## Overview

A multi-goal task planning system that learns efficient strategies through experience, guided by leverage/influence heuristics and curiosity-driven exploration.

## Design Philosophy

### Emergent Over Prescriptive
- No hardcoded strategies (no HTN hierarchies)
- Agents discover efficient approaches through search
- Specialization emerges from experience with different task distributions

### Stochastic Over Deterministic
- All selection uses weighted sampling, not argmax
- Prevents getting stuck in local minima
- Temperature parameter controls exploration/exploitation

### Simple Over Complex
- Portable design (works in Python, C#, C++)
- No multiple inheritance
- No dynamic metaprogramming
- Simple objects and functions

## Core Abstractions

### 1. State as Variable Graph

```
State
├── Objects
│   └── ObjectState
│       ├── properties: Set[str]      # "Container", "Portable", etc.
│       └── attributes: Dict[str, Variable]
└── GlobalVars: Dict[str, Variable]

Variable
├── value: discrete | continuous | set
├── depends_on: Set[Variable]         # Dependency tracking
└── constraints: min/max for continuous
```

**Key insight**: Dependencies enable leverage calculation.

### 2. Actions as Property Templates

```
ActionTemplate(name, params)
├── PropertyRequirements              # "param X must have property Y"
├── Preconditions                     # What must be true
├── Effects                           # What changes
└── cost_fn                           # How expensive

instantiate(objects) → Action         # Bind to specific objects
```

**Key insight**: Actions discovered dynamically based on object properties, not enumerated.

### 3. Influence as Leverage

```
InfluenceGraph
├── nodes: Variable → InfluenceNode
│   ├── direct_goal_relevance         # How much do goals care?
│   ├── transitive_relevance          # Sum of influenced variables
│   ├── depends_on: Set[Variable]
│   └── influences: Set[Variable]
└── propagate_relevance()             # Backwards through dependencies

action.leverage = Σ(affected_variable.total_relevance)
```

**Key insight**: Actions that affect high-leverage variables are prioritized.

### 4. Search as Stochastic Beam

```
SearchNode
├── state: State
├── action: Action                    # How we got here
├── parent: SearchNode
├── value: float                      # Combined heuristic
└── children: List[SearchNode]

value = (α·goal_progress + β·leverage + γ·info_gain + δ·discovery) / cost

expand(node):
    actions = get_applicable_actions(node.state)
    scored = [(a, score(a)) for a in actions]
    selected = stochastic_select(scored, k)  # Not just top k!
    return [apply(a, node.state) for a in selected]

stochastic_select(items, k):
    weights = softmax([item.value for item in items], temperature)
    return sample_without_replacement(items, weights, k)
```

**Key insight**: Weighted sampling prevents deterministic traps.

## Computational Efficiency

### Problem
With N objects and M action templates, naive forward search is O((N^k·M)^d) for depth d.

### Solutions

**1. Property-based filtering**
```python
# Don't try all objects for "PutIn"
# Only try objects with Container property
containers = state.get_objects_with_property("Container")
# Reduces N to |containers| << N
```

**2. Leverage-based pruning**
```python
# Don't explore actions affecting irrelevant variables
if action.affects(variable) and variable.total_relevance < threshold:
    prune(action)
```

**3. Stochastic sampling**
```python
# Don't explore all children of a node
# Sample top K weighted by value
children = stochastic_select(all_children, K)
```

**4. Beam search**
```python
# Don't maintain all frontier nodes
# Keep top beam_width at each depth
current_beam = stochastic_select(all_nodes_at_depth, beam_width)
```

**5. Adaptive depth**
```python
# Stop exploring if:
# - Goal satisfied
# - Leverage not increasing
# - Information gain low
# - Learned stopping criteria triggered
```

## Heuristic Functions

### α: Goal Progress
```python
goal_progress = Σ(goal.satisfaction(after) - goal.satisfaction(before))
```
Direct advancement toward goal conditions.

### β: Leverage
```python
leverage = Σ(affected_var.total_relevance for var in action.effects)
```
Influence on future actions (the "fulcrum" effect).

### γ: Information Gain
```python
info_gain = Σ(var.uncertainty for var in action.affected_unknowns)
```
Expected knowledge discovery from action.

### δ: Discovery Potential
```python
discovery_potential = learned_model.predict(action, context)
```
Learned patterns: "Doors reveal more than drawers", "High ground → high discovery".

### Combined Value
```python
value = (α·goal_progress + β·leverage + γ·info_gain + δ·discovery) / cost
```

Weight adjustment per agent based on experience → specialization emerges.

## Learning Mechanisms

### 1. Discovery Pattern Learning
```python
record_discovery(action, num_discovered, context)

# After many observations:
# "OpenDoor + LargeRoom → high discovery"
# "ClimbMountain → high discovery"
# "OpenDrawer → low discovery"

estimate_discovery(action, context):
    return learned_model.predict(action_features, context_features)
```

### 2. Strategy Abstraction
```python
# After successful plan execution
pattern = extract_pattern(plan, state_before, state_after)

# Pattern:
# - Problem: Transport N objects from A to B
# - Constraint: Carry limit = 1
# - Solution: Use container with capacity >= N
# - Benefit: O(1) trips vs O(N) trips

# Store and match to future problems
```

### 3. Weight Adjustment
```python
# Track correlation between heuristics and success
if plan_succeeded:
    for heuristic in [goal_progress, leverage, info_gain, discovery]:
        if heuristic_was_high_in_plan:
            increase_weight(heuristic, context)

# Different agents develop different weights
# Agent A (logistics): β↑ (leverage)
# Agent B (explorer): γ↑, δ↑ (discovery)
# Agent C (reactive): α↑ (goal progress)
```

## Exploration vs Exploitation

Controlled by **temperature** parameter:

```python
T = 0.1  # Low temp: nearly deterministic, exploit known good actions
T = 1.0  # Medium: balanced exploration/exploitation
T = 10.0 # High temp: nearly uniform, maximum exploration

weights = softmax(values / T)
```

Adaptation:
- High uncertainty → increase T (explore)
- Near deadline → decrease T (exploit)
- Stuck in loop → increase T (break out)

## Knowledge Representation

### Belief State
```python
BeliefState
├── known_objects: Dict[id, ObjectState]
├── uncertain_properties: Dict[id, Set[property]]
└── unobserved_containers: Set[id]      # Curiosity targets
```

Agent reasons based on beliefs, not true state (partial observability).

### Uncertainty Tracking
```python
# See object, infer type
observe(Locker)
infer: Locker has property Container (high confidence)
uncertain: Locker.contents (unknown)
add_exploration_goal: Open(Locker), priority=low

# After opening
observe(Locker.contents)
resolve_uncertainty: Locker.contents = {Key, Note}
remove_exploration_goal: Open(Locker)
```

## Symmetry and Equivalence

### Problem
`PutIn(Bag, L1) → PutIn(Bag, L2)` ≡ `PutIn(Bag, L2) → PutIn(Bag, L1)`

Don't explore both!

### Solution
State hashing with canonical ordering:
```python
state.hash():
    # Sort sets before hashing
    bag_contents = sorted(bag.contents)
    return hash((agent.location, tuple(bag_contents), ...))
```

Visited states cache prevents re-exploration.

## Future Extensions

### Hierarchical Planning
After discovering "container batching" strategy:
```python
# Create macro-action
ContainerTransport(items, container, dest):
    for item in items:
        PutIn(container, item)
    PickUp(container)
    GoTo(dest)
    PutDown(container)
    for item in items:
        TakeOut(container, item)

# Use in future planning at higher abstraction level
```

### Multi-Agent
```python
# Social influence as leverage
convince(OtherAgent, goal):
    # If successful, OtherAgent's actions now serve your goal
    # Delegation = leverage multiplication

# Learn: "Agent B is good at transport tasks"
# → Delegate transport subgoals to Agent B
```

### Continuous Replanning
```python
# World changes during execution
plan = planner.plan(state, goals)
for action in plan:
    execute(action)
    new_state = observe()
    if new_state != expected_state:
        # Replan from current state
        plan = planner.plan(new_state, remaining_goals)
```

## Implementation Status

✅ Architecture defined
✅ Core abstractions designed
⏳ Action instantiation logic
⏳ Influence propagation
⏳ Learning mechanisms
⏳ Letter example working end-to-end

## Testing Strategy

1. **Towers of Hanoi**: Verify basic planning works
2. **Letter/Bag**: Verify leverage discovery
3. **Exploration**: Agent discovers room contents
4. **Multi-goal conflicts**: Detect action interference
5. **Specialization**: Train agents on different task distributions, verify weight divergence

## Performance Targets

- **Small problems** (10 objects, 5 actions, depth 10): < 1 second
- **Medium problems** (50 objects, 20 actions, depth 15): < 10 seconds
- **Large problems** (500 objects, 100 actions, depth 20): < 1 minute

Achieved through aggressive pruning and sampling, not exhaustive search.
