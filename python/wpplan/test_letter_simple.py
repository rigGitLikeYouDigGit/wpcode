"""
Simplified letter example that demonstrates leverage discovery
without complex parameterized actions.

Scenario:
- Agent at Home
- 3 Letters at Home
- 1 Bag at Home
- Goal: Get 1 letter to Work

Agent should discover that putting letter in bag and carrying bag
is more efficient than carrying letter directly.
"""

from wpplan.state import State, ObjectState, Variable, ValueType
from wpplan.action import ActionTemplate
from wpplan.goal import Goal, GoalSet
from wpplan.knowledge import KnowledgeBase
from wpplan.influence import InfluenceGraph
from wpplan.planner import Planner
from wpplan.reversible import ReversibleState


def create_world():
	"""Create the letter world"""
	state = ReversibleState()

	# Agent
	agent = ObjectState(
		id="Agent",
		object_type="Agent",
		properties=set(),
		attributes={
			"location": Variable("Agent.location", ValueType.DISCRETE, "Home"),
			"carrying": Variable("Agent.carrying", ValueType.DISCRETE, None)
		}
	)
	state.add_object(agent)

	# Letter
	letter = ObjectState(
		id="Letter1",
		object_type="Letter",
		properties={"Portable"},
		attributes={
			"location": Variable("Letter1.location", ValueType.DISCRETE, "Home"),
			"in_container": Variable("Letter1.in_container", ValueType.DISCRETE, None)
		}
	)
	state.add_object(letter)

	# Bag
	bag = ObjectState(
		id="Bag",
		object_type="Bag",
		properties={"Portable", "Container"},
		attributes={
			"location": Variable("Bag.location", ValueType.DISCRETE, "Home"),
			"contains_letter": Variable("Bag.contains_letter", ValueType.DISCRETE, False)
		}
	)
	state.add_object(bag)

	return state


def create_actions():
	"""Create action templates"""
	templates = []

	# GoToWork - moves agent and whatever they're carrying
	go_work = ActionTemplate("GoToWork", [])
	go_work.add_precondition("Agent.location", lambda loc: loc == "Home", "At Home")

	def move_to_work(val, state):
		# Move agent
		agent = state.get_object("Agent")
		if not agent:
			return val

		# Move what agent is carrying
		carrying_var = agent.get_attribute("carrying")
		if carrying_var and carrying_var.value:
			carried_id = carrying_var.value
			carried = state.get_object(carried_id)
			if carried:
				carried_loc = carried.get_attribute("location")
				if carried_loc:
					carried_loc.value = "Work"

				# If carrying bag, move letter inside bag too
				if carried_id == "Bag":
					bag_contains = carried.get_attribute("contains_letter")
					if bag_contains and bag_contains.value:
						letter = state.get_object("Letter1")
						if letter:
							letter_loc = letter.get_attribute("location")
							if letter_loc:
								letter_loc.value = "Work"

		return "Work"

	go_work.add_effect("Agent.location", move_to_work, "Move to Work")
	go_work.set_cost_function(lambda s, p: 10.0)  # Expensive!
	templates.append(go_work)

	# PickUpLetter
	pickup_letter = ActionTemplate("PickUpLetter", [])
	pickup_letter.add_precondition("Agent.carrying", lambda c: c is None, "Hands empty")
	pickup_letter.add_precondition("Agent.location", lambda loc: loc == "Home", "At Home")
	pickup_letter.add_precondition("Letter1.location", lambda loc: loc == "Home", "Letter at Home")
	pickup_letter.add_precondition("Letter1.in_container", lambda c: c is None, "Letter not in container")
	pickup_letter.add_effect("Agent.carrying", lambda val, state: "Letter1", "Carry letter")
	pickup_letter.set_cost_function(lambda s, p: 1.0)
	templates.append(pickup_letter)

	# PickUpBag
	pickup_bag = ActionTemplate("PickUpBag", [])
	pickup_bag.add_precondition("Agent.carrying", lambda c: c is None, "Hands empty")
	pickup_bag.add_precondition("Agent.location", lambda loc: loc == "Home", "At Home")
	pickup_bag.add_precondition("Bag.location", lambda loc: loc == "Home", "Bag at Home")
	pickup_bag.add_effect("Agent.carrying", lambda val, state: "Bag", "Carry bag")
	pickup_bag.set_cost_function(lambda s, p: 1.0)
	templates.append(pickup_bag)

	# PutLetterInBag
	put_in_bag = ActionTemplate("PutLetterInBag", [])
	put_in_bag.add_precondition("Agent.location", lambda loc: loc == "Home", "At Home")
	put_in_bag.add_precondition("Letter1.location", lambda loc: loc == "Home", "Letter at Home")
	put_in_bag.add_precondition("Bag.location", lambda loc: loc == "Home", "Bag at Home")
	put_in_bag.add_precondition("Agent.carrying", lambda c: c == "Letter1", "Carrying letter")

	def put_letter_in_bag(val, state):
		# Clear agent carrying
		agent = state.get_object("Agent")
		if agent:
			carrying_var = agent.get_attribute("carrying")
			if carrying_var:
				carrying_var.value = None

		# Set letter in container
		letter = state.get_object("Letter1")
		if letter:
			in_container_var = letter.get_attribute("in_container")
			if in_container_var:
				in_container_var.value = "Bag"

		# Set bag contains letter
		bag = state.get_object("Bag")
		if bag:
			contains_var = bag.get_attribute("contains_letter")
			if contains_var:
				contains_var.value = True

		return val

	put_in_bag.add_effect("Agent.carrying", put_letter_in_bag, "Put letter in bag")
	put_in_bag.set_cost_function(lambda s, p: 1.0)
	templates.append(put_in_bag)

	return templates


def create_goal():
	"""Goal: Letter at Work"""
	def satisfaction(state: State) -> float:
		letter = state.get_object("Letter1")
		if not letter:
			return 0.0
		loc_var = letter.get_attribute("location")
		if not loc_var:
			return 0.0
		return 1.0 if loc_var.value == "Work" else 0.0

	return Goal("Letter at Work", satisfaction, priority=1.0)


def run_test():
	"""Run the letter example"""
	print("\n" + "=" * 60)
	print("Letter Transportation Test")
	print("=" * 60)

	# Create world
	state = create_world()
	print("\nInitial state:")
	print(f"  Agent: {state.get_variable('Agent.location')}, carrying: {state.get_variable('Agent.carrying')}")
	print(f"  Letter1: {state.get_variable('Letter1.location')}, in_container: {state.get_variable('Letter1.in_container')}")
	print(f"  Bag: {state.get_variable('Bag.location')}, contains_letter: {state.get_variable('Bag.contains_letter')}")

	# Setup planning
	kb = KnowledgeBase()
	for template in create_actions():
		kb.add_action_template(template)

	goal = create_goal()
	goals = GoalSet()
	goals.add_goal(goal)

	influence = InfluenceGraph()
	planner = Planner(kb, influence, alpha=1.0, beta=0.5, temperature=0.8)

	print(f"\nGoal: {goal.description}")
	print(f"Initial satisfaction: {goal.satisfaction(state)}")

	# Plan
	print("\nPlanning...")
	plan = planner.plan(state, goals, max_depth=10, max_samples=500, beam_width=5)

	if plan:
		print(f"\nFound plan with {len(plan)} actions:")
		total_cost = 0.0
		for i, action in enumerate(plan, 1):
			cost = action.cost(state)
			total_cost += cost
			print(f"  {i}. {action} (cost: {cost})")
		print(f"\nTotal cost: {total_cost}")

		# Execute plan
		print("\nExecuting plan...")
		current_state = state
		for i, action in enumerate(plan, 1):
			print(f"\n  Step {i}: {action}")
			current_state = action.apply(current_state)
			print(f"    Agent: {current_state.get_variable('Agent.location')}, carrying: {current_state.get_variable('Agent.carrying')}")
			print(f"    Letter1: {current_state.get_variable('Letter1.location')}, in_container: {current_state.get_variable('Letter1.in_container')}")
			print(f"    Bag: {current_state.get_variable('Bag.location')}, contains_letter: {current_state.get_variable('Bag.contains_letter')}")

		print(f"\nFinal satisfaction: {goal.satisfaction(current_state)}")

		if goal.satisfaction(current_state) >= 0.99:
			print("\n[PASS] Goal achieved!")
			return True
		else:
			print("\n[FAIL] Goal not achieved!")
			return False
	else:
		print("\n[FAIL] No plan found!")
		return False


if __name__ == "__main__":
	success = run_test()
	print("\n" + "=" * 60)
	if success:
		print("Test PASSED")
	else:
		print("Test FAILED")
	print("=" * 60)
