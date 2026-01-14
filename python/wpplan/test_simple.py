"""
Simple test to verify basic planning works
"""

from wpplan.state import State, ObjectState, Variable, ValueType
from wpplan.action import ActionTemplate
from wpplan.goal import Goal, GoalSet
from wpplan.knowledge import KnowledgeBase
from wpplan.influence import InfluenceGraph
from wpplan.planner import Planner
from wpplan.reversible import ReversibleState


def test_simple_movement():
	"""Test basic agent movement"""
	print("\n=== Test: Simple Movement ===")

	# Create state
	state = ReversibleState()
	agent = ObjectState(
		id="Agent",
		object_type="Agent",
		traits=set(),
		attributes={
			"location": Variable("Agent.location", ValueType.DISCRETE, "A")
		}
	)
	state.add_object(agent)

	# Create action: Go from A to B
	go_to_b = ActionTemplate("GoToB", [])
	go_to_b.add_precondition("Agent.location", lambda loc: loc == "A", "At A")
	go_to_b.add_effect("Agent.location", lambda val, state: "B", "Move to B")
	go_to_b.set_cost_function(lambda s, p: 1.0)

	# Create goal: Be at B
	def at_b(state: State) -> float:
		agent = state.get_object("Agent")
		if agent:
			loc = agent.get_attribute("location")
			if loc and loc.value == "B":
				return 1.0
		return 0.0

	goal = Goal("Be at B", at_b)
	goals = GoalSet()
	goals.add_goal(goal)

	# Create knowledge base
	kb = KnowledgeBase()
	kb.add_action_template(go_to_b)

	# Create planner
	influence = InfluenceGraph()
	planner = Planner(kb, influence)

	print(f"Initial: Agent at {state.get_variable('Agent.location')}")
	print(f"Goal satisfaction: {goal.satisfaction(state)}")

	# Plan
	plan = planner.plan(state, goals, max_depth=5, max_samples=100)

	if plan:
		print(f"\nFound plan with {len(plan)} actions:")
		for i, action in enumerate(plan, 1):
			print(f"  {i}. {action}")

		# Execute plan
		current_state = state
		for action in plan:
			current_state = action.apply(current_state)

		print(f"\nFinal: Agent at {current_state.get_variable('Agent.location')}")
		print(f"Goal satisfaction: {goal.satisfaction(current_state)}")
		assert goal.satisfaction(current_state) >= 0.99, "Goal not achieved!"
		print("[PASS] Test passed")
	else:
		print("[FAIL] No plan found!")
		return False

	return True


def test_reversible_state():
	"""Test that reversible state works correctly"""
	print("\n=== Test: Reversible State ===")

	state = ReversibleState()
	agent = ObjectState(
		id="Agent",
		object_type="Agent",
		traits=set(),
		attributes={
			"location": Variable("Agent.location", ValueType.DISCRETE, "A")
		}
	)
	state.add_object(agent)

	print(f"Initial: {state.get_variable('Agent.location')}")

	# Test transaction
	state.begin_transaction()
	state.set_variable("Agent.location", "B")
	print(f"After change: {state.get_variable('Agent.location')}")
	assert state.get_variable("Agent.location") == "B"

	state.rollback()
	print(f"After rollback: {state.get_variable('Agent.location')}")
	assert state.get_variable("Agent.location") == "A", "Rollback failed!"

	print("[PASS] Test passed")
	return True


def test_action_application():
	"""Test that actions apply correctly"""
	print("\n=== Test: Action Application ===")

	state = ReversibleState()
	agent = ObjectState(
		id="Agent",
		object_type="Agent",
		traits=set(),
		attributes={
			"location": Variable("Agent.location", ValueType.DISCRETE, "A")
		}
	)
	state.add_object(agent)

	# Create action
	go_to_b = ActionTemplate("GoToB", [])
	go_to_b.add_precondition("Agent.location", lambda loc: loc == "A", "At A")
	go_to_b.add_effect("Agent.location", lambda val, state: "B", "Move to B")
	go_to_b.set_cost_function(lambda s, p: 1.0)

	# Instantiate
	action = go_to_b.instantiate(state, {})
	assert action is not None, "Failed to instantiate action"

	print(f"Initial: {state.get_variable('Agent.location')}")
	print(f"Action applicable: {action.is_applicable(state)}")
	assert action.is_applicable(state), "Action should be applicable!"

	# Apply
	new_state = action.apply(state)
	print(f"After apply: {new_state.get_variable('Agent.location')}")
	assert new_state.get_variable("Agent.location") == "B", "Action didn't change state!"

	# Original state unchanged
	print(f"Original state: {state.get_variable('Agent.location')}")
	assert state.get_variable("Agent.location") == "A", "Original state was modified!"

	print("[PASS] Test passed")
	return True


def run_all_tests():
	"""Run all simple tests"""
	print("=" * 60)
	print("Running Simple Tests")
	print("=" * 60)

	tests = [
		test_reversible_state,
		test_action_application,
		test_simple_movement,
	]

	passed = 0
	for test in tests:
		try:
			if test():
				passed += 1
		except Exception as e:
			print(f"[FAIL] Test failed with exception: {e}")
			import traceback
			traceback.print_exc()

	print("\n" + "=" * 60)
	print(f"Results: {passed}/{len(tests)} tests passed")
	print("=" * 60)


if __name__ == "__main__":
	run_all_tests()
