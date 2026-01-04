# Reversible State Implementation

## Overview

The reversible state system allows efficient exploration of action sequences without expensive state copying. Changes are tracked and can be undone in constant time.

## Core Concept

Instead of:
```python
# Expensive deep copy for every action tried
for action in possible_actions:
	new_state = state.copy()  # O(n) where n = state size
	action.apply(new_state)
	score = evaluate(new_state)
	# new_state discarded
```

We do:
```python
# Cheap transaction for every action tried
for action in possible_actions:
	state.begin_transaction()  # O(1)
	action.apply(state)         # Records changes
	score = evaluate(state)
	state.rollback()            # O(k) where k = changes made
```

**Performance gain**: 10-100x for typical exploration scenarios.

## How It Works

### Transaction Recording

When a transaction is active, all state modifications are recorded:

```python
state.begin_transaction()

# This records: ("Agent.location", old_value="Home")
state.set_variable("Agent.location", "Work")

# This records: ("add", "Bag.contents", "Letter1")
state.add_to_set("Bag.contents", "Letter1")

# This records: ("add_dependency", "Letter1.location", "Bag.location")
state.add_dependency("Letter1.location", "Bag.location")
```

### Rollback

Undo applies changes in reverse order:

```python
state.rollback()
# Restores: Agent.location = "Home"
# Removes: "Letter1" from Bag.contents
# Removes: dependency Letter1.location -> Bag.location
```

### Commit

Accept changes and stop tracking:

```python
state.commit()
# Changes are kept, undo buffer cleared
```

## Supported Operations

### Variable Changes
```python
state.set_variable("Agent.location", "Work")
# Records old value, can be undone
```

### Set Operations
```python
state.add_to_set("Bag.contents", "Letter1")
state.remove_from_set("Bag.contents", "Letter1")
# Both recordable and reversible
```

### Dependencies
```python
state.add_dependency("Letter1.location", "Bag.location")
state.remove_dependency("Letter1.location", "Bag.location")
# Dependency graph changes are reversible
```

## Nested Transactions

Transactions can be nested:

```python
state.begin_transaction()  # Outer
state.set_variable("A", value1)

	state.begin_transaction()  # Inner
	state.set_variable("B", value2)
	state.rollback()  # Only B undone

state.rollback()  # A undone
```

This enables hierarchical exploration:
- Outer transaction: trying high-level strategy
- Inner transactions: trying specific action sequences

## Usage Patterns

### Pattern 1: Try-Evaluate-Discard

```python
# Evaluate multiple actions without keeping state
for action in actions:
	state.begin_transaction()
	action.apply(state)
	score = evaluate(state)
	if score > best_score:
		best_action = action
		best_score = score
	state.rollback()

# Now apply best action permanently
best_action.apply(state)
```

### Pattern 2: Selective Commit

```python
# Only commit promising branches
state.begin_transaction()
action.apply(state)
score = evaluate(state)

if score > threshold:
	state.commit()  # Keep this
	return state.copy()  # Make permanent copy for child node
else:
	state.rollback()  # Discard
	return None
```

### Pattern 3: Snapshot-Restore

```python
# Save current state for later restore
snapshot = state.snapshot()

# Try many things...
explore_many_actions(state)

# Restore to snapshot point
snapshot.restore()
```

### Pattern 4: Helper Functions

```python
# Evaluate action without modifying state
result = try_action(state, action, lambda s: evaluate(s))

# Apply action permanently
commit_action(state, action)
```

## Performance Characteristics

### Time Complexity

| Operation | Reversible | Copy-based |
|-----------|-----------|------------|
| Begin transaction | O(1) | - |
| Modify variable | O(1) | - |
| Rollback | O(k) | - |
| Commit | O(1) | - |
| Copy state | O(n) | O(n) |

Where:
- k = number of changes in transaction (typically 1-10)
- n = total state size (can be 100s-1000s)

### Space Complexity

- Transaction overhead: O(k) per transaction
- Stack depth: O(d) where d = nesting depth
- Total: O(k * d), typically much smaller than O(n) for state copy

### Practical Performance

Test results with 100 objects, 1000 explorations:
- **Reversible**: ~0.05s
- **Copy-based**: ~2.5s
- **Speedup**: ~50x

## Implementation Details

### Transaction Stack

```python
transaction_stack: List[Transaction]

Transaction:
	variable_changes: List[(path, old_value)]
	dependency_changes: List[(op, dependent, depends_on)]
	set_changes: List[(path, op, element)]
```

### Undo Mechanics

All modifications check `_current_transaction()`:
- If active: record change before applying
- If none: apply directly (no undo available)

Rollback iterates changes in reverse:
```python
for change in reversed(transaction.changes):
	undo_change(change)
```

### Avoiding Recursion

When undoing, we temporarily disable transaction recording:
```python
def _undo_variable_change(self, change):
	saved_stack = self.transaction_stack
	self.transaction_stack = []  # Disable recording
	self.set_variable(change.path, change.old_value)
	self.transaction_stack = saved_stack
```

This prevents undo operations from being recorded.

## Limitations

### When to Still Copy

1. **Keeping multiple states**: If you need to maintain several states simultaneously
   ```python
   state1 = state.copy()  # Need permanent copy
   state2 = state.copy()  # Need permanent copy
   ```

2. **Long-term storage**: Nodes kept in search tree need copies
   ```python
   if keep_node:
   	child.state = state.copy()  # Permanent node
   ```

3. **Parallel exploration**: Reversible state is not thread-safe

### Not Reversible

Some operations can't be easily reversed:
- Object creation/deletion (would need to track full object state)
- Complex structural changes
- External side effects

For these, use copy() or implement custom undo logic.

## Integration with Planner

The planner uses reversible state for efficiency:

```python
def _expand_node(self, node):
	applicable_actions = self._get_applicable_actions(node.state)

	scored_actions = []
	for action in applicable_actions:
		# Use reversible state for scoring
		if isinstance(node.state, ReversibleState):
			node.state.begin_transaction()
			action.apply(node.state)
			score = self._score_action(action, node.state, goals)
			node.state.rollback()  # Undo immediately
		else:
			# Fallback to regular scoring
			score = self._score_action(action, node.state, goals)

		scored_actions.append((action, score))

	# Only copy state for kept nodes
	for action in selected_actions:
		new_state = action.apply(node.state)  # Makes copy
		child = SearchNode(new_state, ...)
```

**Key insight**: Score 100 actions with 1 state object, not 100 state copies.

## Testing

Run tests:
```bash
python -m wpplan.test_reversible
```

Tests verify:
- ✓ Basic rollback/commit
- ✓ Nested transactions
- ✓ Set operations
- ✓ Dependency tracking
- ✓ Performance gains (50x+)

## Future Enhancements

### Possible improvements:

1. **Lazy copying**: Only copy parts of state that changed
2. **Copy-on-write**: Share unchanged data between states
3. **Checkpointing**: Save/restore full state efficiently
4. **Undo history**: Keep full history for replay/debugging
5. **Differential state**: Store only deltas from parent

But start simple - current implementation is fast enough for most cases.
