"""
Debug version of letter example to identify planning issues
"""

from wpplan.state import ObjectState, Variable, ValueType
from wpplan.reversible import ReversibleState
from wpplan.relations import Relation, RelationType
from wpplan.action import ActionTemplate
from wpplan.goal import Goal, GoalSet
from wpplan.knowledge import KnowledgeBase
from wpplan.influence import InfluenceGraph
from wpplan.planner import Planner


def create_world():
	"""Create world with Agent, 3 Letters, and 1 Bag"""
	state = ReversibleState()

	# Agent
	agent = ObjectState("Agent", "Agent", set(), {
		"location": Variable("Agent.location", ValueType.DISCRETE, "Home"),
		"carrying": Variable("Agent.carrying", ValueType.DISCRETE, None)
	})
	state.add_object(agent)

	# 3 Letters
	for i in range(1, 4):
		letter = ObjectState(f"Letter{i}", "Letter", {"Portable"}, {
			"location": Variable(f"Letter{i}.location", ValueType.DISCRETE, "Home")
		})
		state.add_object(letter)

	# Bag
	bag = ObjectState("Bag", "Bag", {"Portable", "Container"}, {
		"location": Variable("Bag.location", ValueType.DISCRETE, "Home")
	})
	state.add_object(bag)

	return state


def create_actions():
	"""Create action templates that use relations"""
	templates = []

	# GoToWork
	go_work = ActionTemplate("GoToWork", [])
	go_work.add_precondition("Agent.location", lambda loc: loc == "Home", "At Home")
	go_work.add_effect("Agent.location", lambda val, state: "Work", "Move to Work")
	go_work.set_cost_function(lambda s, p: 10.0)
	templates.append(go_work)

	# GoToHome
	go_home = ActionTemplate("GoToHome", [])
	go_home.add_precondition("Agent.location", lambda loc: loc == "Work", "At Work")
	go_home.add_effect("Agent.location", lambda val, state: "Home", "Move to Home")
	go_home.set_cost_function(lambda s, p: 10.0)
	templates.append(go_home)

	# PickUpLetter1
	pickup1 = ActionTemplate("PickUpLetter1", [])
	pickup1.add_precondition("Agent.carrying", lambda c: c is None, "Hands empty")
	pickup1.add_precondition("Agent.location", lambda loc: loc == "Home", "At Home")
	pickup1.add_precondition("Letter1.location", lambda loc: loc == "Home", "Letter at Home")

	def pickup1_effect(val, state):
		# Create carrying relation: Letter follows Agent
		relation = Relation(RelationType.EQUALS, "Letter1.location", "Agent.location")
		state.add_relation(relation)

		# Return new carrying value
		return "Letter1"

	pickup1.add_effect("Agent.carrying", pickup1_effect, "Pick up Letter1")
	pickup1.set_cost_function(lambda s, p: 1.0)
	templates.append(pickup1)

	# PickUpBag
	pickup_bag = ActionTemplate("PickUpBag", [])
	pickup_bag.add_precondition("Agent.carrying", lambda c: c is None, "Hands empty")
	pickup_bag.add_precondition("Agent.location", lambda loc: loc == "Home", "At Home")
	pickup_bag.add_precondition("Bag.location", lambda loc: loc == "Home", "Bag at Home")

	def pickup_bag_effect(val, state):
		# Create carrying relation: Bag follows Agent
		relation = Relation(RelationType.EQUALS, "Bag.location", "Agent.location")
		state.add_relation(relation)

		# Return new carrying value
		return "Bag"

	pickup_bag.add_effect("Agent.carrying", pickup_bag_effect, "Pick up Bag")
	pickup_bag.set_cost_function(lambda s, p: 1.0)
	templates.append(pickup_bag)

	# PutDown
	putdown = ActionTemplate("PutDown", [])
	putdown.add_precondition("Agent.carrying", lambda c: c is not None, "Must be carrying")

	def putdown_effect(val, state):
		agent = state.get_object("Agent")
		if not agent:
			return None

		carrying_var = agent.get_attribute("carrying")
		if not carrying_var or not carrying_var.value:
			return None

		carried_id = carrying_var.value
		state.remove_relation(f"{carried_id}.location", "Agent.location")
		return None

	putdown.add_effect("Agent.carrying", putdown_effect, "Put down object")
	putdown.set_cost_function(lambda s, p: 1.0)
	templates.append(putdown)

	# PutLetter1InBag
	putin1 = ActionTemplate("PutLetter1InBag", [])
	putin1.add_precondition("Agent.location", lambda loc: loc == "Home", "At Home")
	putin1.add_precondition("Bag.location", lambda loc: loc == "Home", "Bag at Home")
	putin1.add_precondition("Agent.carrying", lambda c: c == "Letter1", "Carrying Letter1")

	def putin1_effect(val, state):
		agent = state.get_object("Agent")
		if not agent:
			return val

		state.remove_relation("Letter1.location", "Agent.location")
		relation = Relation(RelationType.EQUALS, "Letter1.location", "Bag.location")
		state.add_relation(relation)
		return None

	putin1.add_effect("Agent.carrying", putin1_effect, "Put Letter1 in Bag")
	putin1.set_cost_function(lambda s, p: 1.0)
	templates.append(putin1)

	return templates


def test_manual_execution():
	"""Test manually executing a simple plan"""
	print("\n" + "=" * 70)
	print("Manual Execution Test")
	print("=" * 70)

	state = create_world()
	templates = create_actions()

	print("\nInitial state:")
	print(f"  Agent: location={state.get_variable('Agent.location')}, carrying={state.get_variable('Agent.carrying')}")
	print(f"  Letter1: location={state.get_variable('Letter1.location')}")
	print(f"  Bag: location={state.get_variable('Bag.location')}")

	# Manually execute: PickUpLetter1
	print("\nStep 1: PickUpLetter1")
	pickup1 = templates[2]  # PickUpLetter1
	action1 = pickup1.instantiate(state, {})

	if action1:
		print(f"  Action created: {action1}")
		print(f"  Is applicable: {action1.is_applicable(state)}")

		if action1.is_applicable(state):
			state = action1.apply(state)
			print(f"  After: Agent.location={state.get_variable('Agent.location')}, Agent.carrying={state.get_variable('Agent.carrying')}")
			print(f"  After: Letter1.location={state.get_variable('Letter1.location')}")
			print(f"  Bag leverage: {state.get_influence_count('Bag.location')}")
	else:
		print("  [ERROR] Could not instantiate action!")

	# Manually execute: GoToWork
	print("\nStep 2: GoToWork")
	go_work = templates[0]
	action2 = go_work.instantiate(state, {})

	if action2:
		print(f"  Action created: {action2}")
		print(f"  Is applicable: {action2.is_applicable(state)}")

		if action2.is_applicable(state):
			state = action2.apply(state)
			print(f"  After: Agent.location={state.get_variable('Agent.location')}")
			print(f"  After: Letter1.location={state.get_variable('Letter1.location')}")
	else:
		print("  [ERROR] Could not instantiate action!")


def test_action_cache():
	"""Test if actions are being cached properly"""
	print("\n" + "=" * 70)
	print("Action Cache Test")
	print("=" * 70)

	state = create_world()
	kb = KnowledgeBase()

	templates = create_actions()
	for template in templates:
		kb.add_action_template(template)

	print(f"\nAction templates: {len(templates)}")
	for template in templates:
		print(f"  - {template.name}")

	# Build cache
	from wpplan.action_cache import ActionCache
	cache = ActionCache()
	cache.build_cache(state, templates)

	print(f"\nCached actions: {len(cache.cache)}")
	for action in cache.cache.values():
		print(f"  - {action}")

	# Check applicability
	print(f"\nApplicable actions in initial state:")
	applicable = cache.get_applicable_actions(state)
	print(f"  Count: {len(applicable)}")
	for action in applicable:
		print(f"  - {action}")


def test_planner_first_step():
	"""Test if planner can find any applicable actions"""
	print("\n" + "=" * 70)
	print("Planner First Step Test")
	print("=" * 70)

	state = create_world()
	kb = KnowledgeBase()

	templates = create_actions()
	for template in templates:
		kb.add_action_template(template)

	goal = Goal("Get Letter1 to Work",
				lambda s: 1.0 if s.get_variable('Letter1.location') == 'Work' else 0.0,
				priority=1.0)
	goals = GoalSet()
	goals.add_goal(goal)

	influence = InfluenceGraph()
	planner = Planner(kb, influence, alpha=1.0, beta=2.0, gamma=0.0, delta=0.0, temperature=0.5)

	print(f"\nInitial state:")
	print(f"  Goal satisfaction: {goal.satisfaction(state):.2f}")

	# Try planning with verbose output
	print("\nPlanning (max_depth=5, max_samples=100)...")
	plan = planner.plan(state, goals, max_depth=5, max_samples=100, beam_width=5)

	if plan:
		print(f"\n[SUCCESS] Found plan with {len(plan)} actions:")
		for i, action in enumerate(plan, 1):
			print(f"  {i}. {action}")
	else:
		print("\n[FAIL] No plan found")


if __name__ == "__main__":
	# Run tests
	test_manual_execution()
	test_action_cache()
	test_planner_first_step()
