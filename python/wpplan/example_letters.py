"""
Example: Letter transportation problem

Agent at Home with 5 Letters and 1 Bag.
Goal: Get 3 letters to Work.
Agent discovers that using the bag is more efficient than individual trips.
"""

from wpplan.state import State, ObjectState, Variable, ValueType
from wpplan.action import ActionTemplate
from wpplan.goal import Goal, GoalSet
from wpplan.knowledge import KnowledgeBase
from wpplan.influence import InfluenceGraph
from wpplan.planner import Planner


def create_letter_world():
	"""Create the initial state for the letter problem"""
	state = State()

	# Create Agent
	agent = ObjectState(
		id="Agent",
		object_type="Agent",
		traits=set(),
		attributes={
			"location": Variable("Agent.location", ValueType.DISCRETE, "Home"),
			"carrying": Variable("Agent.carrying", ValueType.DISCRETE, None)
		}
	)
	state.add_object(agent)

	# Create 5 Letters at Home
	for i in range(1, 6):
		letter = ObjectState(
			id=f"Letter{i}",
			object_type="Letter",
			traits={"Portable"},
			attributes={
				"location": Variable(f"Letter{i}.location", ValueType.DISCRETE, "Home")
			}
		)
		state.add_object(letter)

	# Create Bag with Container property
	bag = ObjectState(
		id="Bag",
		object_type="Bag",
		traits={"Portable", "Container"},
		attributes={
			"location": Variable("Bag.location", ValueType.DISCRETE, "Home"),
			"contents": Variable("Bag.contents", ValueType.SET, set()),
			"capacity": Variable("Bag.capacity", ValueType.CONTINUOUS, 10.0)
		}
	)
	state.add_object(bag)

	return state


def create_action_templates():
	"""Create action templates for the letter world"""
	templates = []

	# GoToWork action
	go_to_work = ActionTemplate("GoToWork", [])
	go_to_work.add_precondition(
		"Agent.location",
		lambda loc: loc == "Home",
		"Agent must be at Home"
	)
	go_to_work.add_effect(
		"Agent.location",
		lambda val, state: "Work",
		"Agent moves to Work"
	)
	# When agent moves, anything they're carrying moves too
	def move_carrying_to_work(val, state):
		agent = state.get_object("Agent")
		if agent:
			carrying = agent.get_attribute("carrying")
			if carrying and carrying.value:
				carried_obj_id = carrying.value
				carried_obj = state.get_object(carried_obj_id)
				if carried_obj:
					loc_var = carried_obj.get_attribute("location")
					if loc_var:
						loc_var.value = "Work"
		return val
	go_to_work.add_effect("Agent.location", move_carrying_to_work, "Move carried object")
	go_to_work.set_cost_function(lambda s, p: 10.0)  # Expensive!
	templates.append(go_to_work)

	# GoToHome action
	go_to_home = ActionTemplate("GoToHome", [])
	go_to_home.add_precondition(
		"Agent.location",
		lambda loc: loc == "Work",
		"Agent must be at Work"
	)
	go_to_home.add_effect(
		"Agent.location",
		lambda val, state: "Home",
		"Agent moves to Home"
	)
	def move_carrying_to_home(val, state):
		agent = state.get_object("Agent")
		if agent:
			carrying = agent.get_attribute("carrying")
			if carrying and carrying.value:
				carried_obj_id = carrying.value
				carried_obj = state.get_object(carried_obj_id)
				if carried_obj:
					loc_var = carried_obj.get_attribute("location")
					if loc_var:
						loc_var.value = "Home"
		return val
	go_to_home.add_effect("Agent.location", move_carrying_to_home, "Move carried object")
	go_to_home.set_cost_function(lambda s, p: 10.0)  # Expensive!
	templates.append(go_to_home)

	# PickUp action (for Portable objects)
	pickup = ActionTemplate("PickUp", ["object"])
	pickup.add_property_requirement("object", ["Portable"])
	# Must not already be carrying something
	pickup.add_precondition("Agent.carrying", lambda c: c is None, "Hands must be empty")
	# Object must be at same location as agent
	def same_location_precondition(obj_id):
		def check(state):
			agent = state.get_object("Agent")
			obj = state.get_object(obj_id)
			if not agent or not obj:
				return False
			agent_loc = agent.get_attribute("location")
			obj_loc = obj.get_attribute("location")
			if not agent_loc or not obj_loc:
				return False
			return agent_loc.value == obj_loc.value
		return check
	# Note: This is a hack - we'll fix precondition instantiation properly
	pickup.add_precondition("_custom", lambda s: True, "Same location check")
	# Effect: Agent now carrying object
	def pickup_effect(obj_id):
		def effect(val, state):
			return obj_id
		return effect
	pickup.add_effect("Agent.carrying", lambda val, state: "PLACEHOLDER", "Pick up object")
	pickup.set_cost_function(lambda s, p: 1.0)
	templates.append(pickup)

	# PutDown action
	putdown = ActionTemplate("PutDown", ["object"])
	putdown.add_precondition("Agent.carrying", lambda c: c is not None, "Must be carrying something")
	putdown.add_effect("Agent.carrying", lambda val, state: None, "Put down object")
	putdown.set_cost_function(lambda s, p: 1.0)
	templates.append(putdown)

	# PutIn action (for Containers)
	putin = ActionTemplate("PutIn", ["container", "object"])
	putin.add_property_requirement("container", ["Container"])
	putin.add_property_requirement("object", ["Portable"])
	# Agent must be carrying the object
	putin.add_precondition("Agent.carrying", lambda c: c is not None, "Must be carrying object")
	# Container must be at same location
	putin.add_precondition("_custom", lambda s: True, "Same location check")
	# Effect: Add to container contents, remove from agent carrying
	putin.add_effect("Agent.carrying", lambda val, state: None, "Release object")
	putin.set_cost_function(lambda s, p: 1.0)
	templates.append(putin)

	# TakeOut action
	takeout = ActionTemplate("TakeOut", ["container", "object"])
	takeout.add_property_requirement("container", ["Container"])
	# Must not be carrying something
	takeout.add_precondition("Agent.carrying", lambda c: c is None, "Hands must be empty")
	# Object must be in container
	takeout.add_precondition("_custom", lambda s: True, "Object in container check")
	# Effect: Remove from container, add to agent carrying
	takeout.set_cost_function(lambda s, p: 1.0)
	templates.append(takeout)

	return templates


def create_goal():
	"""Create the goal: 3 letters at Work"""
	def satisfaction_fn(state: State) -> float:
		"""Check how many letters are at Work"""
		letters_at_work = 0
		for obj in state.objects.values():
			if obj.object_type == "Letter":
				loc = obj.get_attribute("location")
				if loc and loc.value == "Work":
					letters_at_work += 1

		# Goal is satisfied if 3+ letters at Work
		return min(1.0, letters_at_work / 3.0)

	return Goal(
		description="Get 3 letters to Work",
		satisfaction_fn=satisfaction_fn,
		priority=1.0
	)


def run_example():
	"""Run the letter transportation example"""
	print("=" * 60)
	print("Letter Transportation Problem")
	print("=" * 60)

	# Create world
	state = create_letter_world()
	print("\nInitial state:")
	print(f"  Agent at: {state.get_object('Agent').get_attribute('location').value}")
	print(f"  5 Letters at Home")
	print(f"  1 Bag at Home (Container, capacity=10)")

	# Create knowledge base
	kb = KnowledgeBase()
	for template in create_action_templates():
		kb.add_action_template(template)

	# Create influence graph
	influence = InfluenceGraph()

	# Create goal
	goal = create_goal()
	goals = GoalSet()
	goals.add_goal(goal)

	print("\nGoal: Get 3 letters to Work")

	# Create planner
	planner = Planner(
		knowledge=kb,
		influence_graph=influence,
		alpha=1.0,  # Goal progress
		beta=0.5,   # Leverage
		gamma=0.3,  # Info gain
		delta=0.2,  # Discovery
		temperature=0.5
	)

	print("\nPlanning...")
	plan = planner.plan(state, goals, max_depth=15, max_samples=500)

	if plan:
		print(f"\nFound plan with {len(plan)} actions:")
		total_cost = 0.0
		for i, action in enumerate(plan, 1):
			cost = action.cost(state)
			total_cost += cost
			print(f"  {i}. {action} (cost: {cost})")
		print(f"\nTotal cost: {total_cost}")
	else:
		print("\nNo plan found!")

	print("\n" + "=" * 60)


if __name__ == "__main__":
	run_example()
