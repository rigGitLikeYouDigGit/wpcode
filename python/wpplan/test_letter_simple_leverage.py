"""
Simplified letter example to demonstrate leverage with bag strategy
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
	"""Create world with Agent, 2 Letters, and 1 Bag"""
	state = ReversibleState()

	# Agent
	agent = ObjectState("Agent", "Agent", set(), {
		"location": Variable("Agent.location", ValueType.DISCRETE, "Home"),
		"carrying": Variable("Agent.carrying", ValueType.DISCRETE, None)
	})
	state.add_object(agent)

	# 2 Letters
	for i in range(1, 3):
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

	# PickUp actions for each letter
	for i in range(1, 3):
		pickup = ActionTemplate(f"PickUpLetter{i}", [])
		pickup.add_precondition("Agent.carrying", lambda c: c is None, "Hands empty")
		pickup.add_precondition("Agent.location", lambda loc: loc == "Home", "At Home")
		pickup.add_precondition(f"Letter{i}.location", lambda loc: loc == "Home", "Letter at Home")

		def make_pickup_effect(letter_id):
			def effect(val, state):
				relation = Relation(RelationType.EQUALS, f"{letter_id}.location", "Agent.location")
				state.add_relation(relation)
				return letter_id
			return effect

		pickup.add_effect("Agent.carrying", make_pickup_effect(f"Letter{i}"), f"Pick up Letter{i}")
		pickup.set_cost_function(lambda s, p: 1.0)
		templates.append(pickup)

	# PickUpBag
	pickup_bag = ActionTemplate("PickUpBag", [])
	pickup_bag.add_precondition("Agent.carrying", lambda c: c is None, "Hands empty")
	pickup_bag.add_precondition("Agent.location", lambda loc: loc == "Home", "At Home")
	pickup_bag.add_precondition("Bag.location", lambda loc: loc == "Home", "Bag at Home")

	def pickup_bag_effect(val, state):
		relation = Relation(RelationType.EQUALS, "Bag.location", "Agent.location")
		state.add_relation(relation)
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

	# PutInBag actions for each letter
	for i in range(1, 3):
		putin = ActionTemplate(f"PutLetter{i}InBag", [])
		putin.add_precondition("Agent.location", lambda loc: loc == "Home", "At Home")
		putin.add_precondition("Bag.location", lambda loc: loc == "Home", "Bag at Home")

		# Capture the letter_id in a closure
		def make_precondition(letter_id):
			return lambda c: c == letter_id
		putin.add_precondition("Agent.carrying", make_precondition(f"Letter{i}"), f"Carrying Letter{i}")

		def make_putin_effect(letter_id):
			def effect(val, state):
				# Remove Agent relation
				state.remove_relation(f"{letter_id}.location", "Agent.location")
				# Create Bag relation: Letter now follows Bag
				relation = Relation(RelationType.EQUALS, f"{letter_id}.location", "Bag.location")
				state.add_relation(relation)
				return None
			return effect

		putin.add_effect("Agent.carrying", make_putin_effect(f"Letter{i}"), f"Put Letter{i} in Bag")
		putin.set_cost_function(lambda s, p: 1.0)
		templates.append(putin)

	return templates


def create_goal():
	"""Goal: Both letters at Work"""
	def satisfaction(state):
		count = 0
		for i in range(1, 3):
			letter = state.get_object(f"Letter{i}")
			if letter:
				loc_var = letter.get_attribute("location")
				if loc_var and loc_var.value == "Work":
					count += 1
		return count / 2.0

	return Goal("Get 2 letters to Work", satisfaction, priority=1.0)


def run_test():
	"""Run the letter example with leverage discovery"""
	print("\n" + "=" * 70)
	print("Simplified Letter Transportation with Leverage Discovery")
	print("=" * 70)

	# Create world
	state = create_world()

	print("\nInitial state:")
	print(f"  Agent: location={state.get_variable('Agent.location')}, carrying={state.get_variable('Agent.carrying')}")
	for i in range(1, 3):
		print(f"  Letter{i}: location={state.get_variable(f'Letter{i}.location')}")
	print(f"  Bag: location={state.get_variable('Bag.location')}")

	# Setup planning
	kb = KnowledgeBase()
	for template in create_actions():
		kb.add_action_template(template)

	goal = create_goal()
	goals = GoalSet()
	goals.add_goal(goal)

	influence = InfluenceGraph()

	# Create planner with leverage enabled
	planner = Planner(
		kb,
		influence,
		alpha=1.0,   # Goal progress
		beta=2.0,    # Leverage (HIGH - we want to discover bag strategy!)
		gamma=0.0,   # Info gain (not used yet)
		delta=0.0,   # Discovery (not used yet)
		temperature=0.5  # Some exploration
	)

	print(f"\nGoal: {goal.description}")
	print(f"Initial satisfaction: {goal.satisfaction(state):.2f}")
	print(f"\nPlanner weights: alpha={planner.alpha}, beta={planner.beta}")

	# Plan
	print("\nPlanning...")
	plan = planner.plan(state, goals, max_depth=15, max_samples=2000, beam_width=10)

	if plan:
		print(f"\nFound plan with {len(plan)} actions:")
		total_cost = 0.0
		for i, action in enumerate(plan, 1):
			cost = action.cost(state)
			total_cost += cost
			print(f"  {i}. {action} (cost: {cost})")
		print(f"\nTotal cost: {total_cost}")

		# Analyze plan
		print("\nPlan analysis:")
		if "Bag" in str(plan):
			print("  [SUCCESS] Plan uses the bag! Leverage discovery worked!")
		else:
			print("  [PARTIAL] Plan doesn't use bag - may need tuning")

		# Execute plan and show result
		print("\nExecuting plan...")
		current_state = state
		for i, action in enumerate(plan, 1):
			print(f"\nStep {i}: {action}")
			print(f"  Before: Agent.location={current_state.get_variable('Agent.location')}, Agent.carrying={current_state.get_variable('Agent.carrying')}")
			current_state = action.apply(current_state)
			print(f"  After: Agent.location={current_state.get_variable('Agent.location')}, Agent.carrying={current_state.get_variable('Agent.carrying')}")
			for j in range(1, 3):
				print(f"  Letter{j}.location={current_state.get_variable(f'Letter{j}.location')}")
			print(f"  Bag.location={current_state.get_variable('Bag.location')}")
			bag_leverage = current_state.get_influence_count("Bag.location")
			if bag_leverage > 0:
				print(f"  >> Bag leverage: {bag_leverage}")

		print(f"\nFinal satisfaction: {goal.satisfaction(current_state):.2f}")

		if goal.satisfaction(current_state) >= 0.99:
			print("\n[PASS] Goal achieved!")
			return True
		else:
			print("\n[FAIL] Goal not fully achieved!")
			return False
	else:
		print("\n[FAIL] No plan found!")
		return False


if __name__ == "__main__":
	success = run_test()
	print("\n" + "=" * 70)
	if success:
		print("Test PASSED - Leverage discovery demonstrated!")
	else:
		print("Test FAILED")
	print("=" * 70)
