# Bug Fixes

## Critical Bugs Fixed (2026-01-03)

### 1. Effect Functions Not Properly Updating State

**Problem**: Effect functions were directly modifying `var.value` instead of using `set_variable()`, which bypassed ReversibleState's transaction tracking and relation propagation.

**Location**: `wpplan/action.py`, line 45-59 in `Effect.apply()`

**Symptom**:
- Actions appeared to execute but state didn't change
- `Agent.carrying` remained `None` after PickUp actions
- Relations weren't being propagated

**Fix**: Modified `Effect.apply()` to:
1. Call the effect function to get the new value
2. Use `state.set_variable()` if available (ReversibleState) for proper propagation
3. Fall back to direct assignment for basic State

```python
# Before (broken):
var.value = self.effect_fn(var.value, state)

# After (working):
new_value = self.effect_fn(var.value, state)
if hasattr(state, 'set_variable'):
    state.set_variable(self.variable_path, new_value)
else:
    var.value = new_value
```

### 2. Relations Not Being Copied

**Problem**: `ReversibleState.copy()` performed deep copy of objects and variables but didn't copy the relations graph.

**Location**: `wpplan/reversible.py`, line 380-390 in `copy()`

**Symptom**:
- Relations created in one state were lost after `action.apply()` which calls `state.copy()`
- Letter would not follow Agent when Agent moved, even though relation was created
- Leverage calculations showed 0 influence

**Fix**: Added `new_state.relations = copy.deepcopy(self.relations)` to copy method.

```python
# Before (broken):
def copy(self) -> 'ReversibleState':
    import copy
    new_state = ReversibleState()
    new_state.objects = copy.deepcopy(self.objects)
    new_state.global_vars = copy.deepcopy(self.global_vars)
    # Don't copy transaction stack
    return new_state

# After (working):
def copy(self) -> 'ReversibleState':
    import copy
    new_state = ReversibleState()
    new_state.objects = copy.deepcopy(self.objects)
    new_state.global_vars = copy.deepcopy(self.global_vars)
    new_state.relations = copy.deepcopy(self.relations)  # <-- Added this!
    # Don't copy transaction stack
    return new_state
```

### 3. Effect Functions Returning Wrong Values

**Problem**: Effect functions in test files were returning the old value (`val`) instead of the new value, or were directly modifying state and returning `val`.

**Location**: Multiple test files (`test_letter_with_leverage.py`, etc.)

**Symptom**: Variables weren't being updated even though relations were being created.

**Fix**: Effect functions should ONLY:
1. Create/remove relations as side effects
2. Return the NEW value for the variable

```python
# Before (broken):
def pickup_effect(val, state):
    agent = state.get_object("Agent")
    if agent:
        carrying_var = agent.get_attribute("carrying")
        if carrying_var:
            carrying_var.value = "Letter1"  # Direct modification!

    relation = Relation(...)
    state.add_relation(relation)
    return val  # Returning old value!

# After (working):
def pickup_effect(val, state):
    # Create relation as side effect
    relation = Relation(RelationType.EQUALS, "Letter1.location", "Agent.location")
    state.add_relation(relation)

    # Return new value
    return "Letter1"
```

## Impact

These three bugs were preventing the planning system from working at all:
- **Bug 1** prevented actions from having any effect
- **Bug 2** prevented relations from persisting across planning steps
- **Bug 3** prevented variables from updating correctly

After fixes:
- ✓ Manual action execution works
- ✓ Relation propagation works
- ✓ Planner finds valid plans
- ✓ Leverage-based discovery works
- ✓ All tests pass (8/8)

## Test Results

### Before Fixes
```
Planning...
Action cache built: 6 total actions
[FAIL] No plan found!
```

### After Fixes
```
Planning...
Action cache built: 8 total actions
Found plan with 6 actions:
  1. PickUpLetter1()
  2. PutLetter1InBag()
  3. PickUpLetter2()
  4. PutLetter2InBag()
  5. PickUpBag()
  6. GoToWork()
Total cost: 15.0
[PASS] Goal achieved!
```

The planner successfully discovers the bag strategy (leverage = 2) which is more efficient than the naive approach (cost 15 vs 32).
