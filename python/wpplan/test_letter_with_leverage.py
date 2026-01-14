"""
Letter example with relation-based leverage discovery.

This demonstrates the core innovation: the planner should discover that
using the bag is better because moving the bag affects multiple letters
(high leverage), even though it requires more initial setup steps.

consider traits to describe environments semantically if not physically -

knee-high, waist-high,
flat-top, thin-wall, opaque, hard?

TODO: refactor to assume object-local var names during solve
"""
from itertools import product

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

	# Agent is:
	# an actor (can take its own actions)
	# mobile (can move and generate GoTo actions) -> inherits from placed (
	# has a location)
	# bodied (has a body)
	agentName = "AgentA"
	agent = ObjectState(
		id=agentName,
		object_type="agent",
		traits={"Actor", "Motive", "Bodied"},
		attributes={
		# expected attributes for Motive schema
		"location": Variable(f"{agentName}.location", ValueType.DISCRETE, "Home"),
		"movementCostFn": lambda state, action, src, dst: 10.0 , # should be
			# saved per object type
		# expected attributes for Bodied schema
		"carrying": Variable(f"{agentName}.carrying", ValueType.DISCRETE, None)
	})
	state.add_object(agent)

	#TODO: proper object schema-ing, factory functions
	# is object_type just the exact leaf type of each object? should be
	def make_letter(obj_id, location):
		""" a letter (an envelope) is
		a Prop (maybe a compound type name?)
			is portable (can be carried) -> inherits from
			placed (has a location)
		Sized (has a physical size and weight)
		Shaped (has a shape wrt other things, eg flat)
		MadeOf (has a material?)
		Container (can be opened, can contain other things)
		Writable (can be written on, up to a limit, can be scratched out)

		"""
		return ObjectState(
			id=obj_id,
			object_type="letter",
			traits={"Portable", "Physical", "Sized", "Shaped",
			        "MadeOf", "Container", "Writable"},
			attributes={
				# Portable
				"location": Variable(f"{obj_id}.location",
				                     ValueType.DISCRETE, location),
				"heldBy" : Variable(f"{obj_id}.heldBy", ValueType.SET, None),
					# set because what if people fight over the letter
				# Sized?
				"size" : Variable("size", ValueType.DISCRETE, "leaf"),
				"weight" : Variable("weight", ValueType.DISCRETE,"leaf"),
				# Shaped
				"shape" : Variable("shape", ValueType.DISCRETE, "flat"), # no idea, maybe describe sides differently?
				# MadeOf
				"material" : Variable("material", ValueType.DISCRETE, "paper"),
				# Container
				"open" : Variable("open", ValueType.DISCRETE, False), "openActionT"
				: None, "closeActionT" : None,
				"capacity" : Variable("capacity", ValueType.DISCRETE, None),
				# prefer action conditions instead of predicates below
				# "willFitFn" : lambda state, obj, item: True,
				# "canExtractFn" : lambda state, obj, item: True,
				"contents" : Variable(f"{obj_id}.contents", ValueType.SET,
				                      None),
				# Writable
				"writing" : None # refer to some other writing struct
				# elsewhere in state
			}
		)

	# 3 Letters
	for i in range(1, 4):
		letter = make_letter(f"Letter{i}", "Home")
		state.add_object(letter)


	# Bag
	bag = ObjectState(
		id="bag",
		object_type="Bag",
		traits={"Portable", "Physical", "Sized", "Shaped",
			        "MadeOf", "Container"} ,
		attributes={
			"location": Variable("Bag.location", ValueType.DISCRETE, "Home"),
			"heldBy" : Variable("Bag.heldBy", ValueType.SET, set()),
			"size" : "medium", "weight" : "light",
			"shape" : "soft", # other things can't be stably put on top
			"material" : "fabric",
			"open" : Variable("Bag.open", ValueType.DISCRETE, False),
			"openActionT" : None, "closeActionT" : None,
			"capacity" : None,
			"contents" : Variable("Bag.contents", ValueType.SET, set()),
	})
	state.add_object(bag)

	return state


def create_actions():
	"""Create action templates that use relations
	 below action templates should be inherent based on
	 traits of objects
	 """
	templates = []

	# ===== Movement Actions =====
	# GoTo action - templated on Motive, source and destination
	goTo = ActionTemplate(
		name="GoToT",
		param_names=["mobile", "src", "dst"] # object and constant
		# values
	)
	goTo.add_precondition(
		var_path="mobile.location",
		predicate=lambda loc, action: loc == action.params["src"],
		description=lambda loc, action: f"At source {action.params["src"]}")
	goTo.add_effect(
		var_path="mobile.location",
		effect_fn=lambda loc, action: action.params["dst"],
		description=lambda loc, action: f"Moved to dst"
		                                f" {action.params["dst"]}")
	# calculate distance between src and dst, which may be different for each
	# actor
	goTo.set_cost_function(lambda state, action: state.get_object(
		action.params["mobile"]).attributes["movementCostFn"](
			state, action,
          action.params["src"], action.params["dst"])
	                       )
	templates.append(goTo)


	####### pickup is inherent to a bodied actor and a prop
	# don't overconstrain here to look for a free limb to pick up -
	# whichever limb action is tried on has to be free
	# manip inherits location from its body, which inherits from its actor?
	pickupT = ActionTemplate(
		name=f"PickUpT",
		param_names=["manip", "portable"]
	                                )
	pickupT.add_precondition(
		"manip.carrying",
		lambda v, a: v is None,
	    lambda loc, action: "Hands empty")
	# TODO:is there a better way to say a condition of equivalence than 2
	#  separate conditions?
	pickupT.add_precondition(
		"manip.location",
		lambda v, a: v == a.params["portable"].attributes["location"],
		lambda loc, action: f"At portable {action.params['portable'].id}"
	)
	pickupT.add_precondition(
		"portable.location",
		lambda v, a: v == a.params["manip"].attributes["location"],
		lambda loc, action: f"At manip {action.params['manip'].id}"
	)

	def pickupTEffect( state, action):
		manip_id=action.params['manip']
		portable_id=action.params['portable']
		relation = Relation(RelationType.EQUALS,
		                    f"{manip_id}.location",
		                    f"{portable_id}.location")
		state.add_relation(relation)
		return action.params['portable'].id
	pickupT.add_effect(
		"manip.carrying",
		pickupTEffect,
		lambda *_: "Pick up portable"
	)
	pickupT.set_cost_function(lambda state, action: 1.0)
	templates.append(pickupT)

	####### putDown template, should be inherent to a bodied actor and a prop
	# being
	# carried
	putdown = ActionTemplate(
		"PutDownT",
		param_names=["manip", "portable"]
	)
	putdown.add_precondition(
		"manip.carrying",
		lambda v, a: v is not None,
		lambda v, a : "Must be carrying"
	)

	def putdown_effect(state, action):
		manip = state.get_object(action.params['manip'])
		if not manip:
			return None

		carrying_var = manip.get_attribute("carrying")
		if not carrying_var or not carrying_var.value:
			return None

		carried_id = carrying_var.value

		# Remove the carrying relation
		state.remove_relation(f"{carried_id}.location", "Agent.location")

		# Clear carrying
		return None

	putdown.add_effect("Agent.carrying", putdown_effect, "Put down object")
	putdown.set_cost_function(lambda s, p: 0.1)
	templates.append(putdown)


	###### PutIn template
	# TODO: definitely need multi-input preconditions for equivalence,
	#   checking everything is in the same place here takes 9
	#   preconditions
	putInT = ActionTemplate(f"PutInT",
	                        ["manip", "portable", "container"]
	                        )
	for a, b in product(putInT.param_names, repeat=2):
		if a == b:
			continue
		putInT.add_precondition(
			f"{a}.location",
			lambda v, a: v == ,
			f"{a} == {b}")
	putInT.add_precondition("manip.location",
	                        lambda loc: loc == "Home",
	                       "At Home"
	                        )
	putInT.add_precondition("Bag.location", lambda loc: loc == "Home",
	                       "Bag at Home")
	putInT.add_precondition("Agent.carrying", lambda c: c == f"Letter{i}",
	                       f"Carrying Letter{i}")



	# # ===== PickUp Actions (one per object) =====
	#
	# for i in range(1, 4):
	# 	pickup = ActionTemplate(f"PickUpLetter{i}", [])
	# 	pickup.add_precondition("Agent.carrying", lambda c: c is None, "Hands empty")
	# 	pickup.add_precondition("Agent.location", lambda loc: loc == "Home", "At Home")
	# 	pickup.add_precondition(f"Letter{i}.location", lambda loc: loc == "Home", "Letter at Home")
	#
	# 	# Effect: Carry letter and create relation
	# 	def make_pickup_effect(letter_id):
	# 		def effect(val, state):
	# 			# Create carrying relation: Letter follows Agent
	# 			relation = Relation(
	# 				RelationType.EQUALS,
	# 				f"{letter_id}.location",
	# 				"Agent.location"
	# 			)
	# 			state.add_relation(relation)
	#
	# 			# Return new carrying value
	# 			return letter_id
	# 		return effect
	#
	# 	pickup.add_effect("Agent.carrying", make_pickup_effect(f"Letter{i}"), f"Pick up Letter{i}")
	# 	pickup.set_cost_function(lambda s, p: 1.0)
	# 	templates.append(pickup)
	#
	# # PickUpBag
	# pickup_bag = ActionTemplate("PickUpBag", [])
	# pickup_bag.add_precondition("Agent.carrying", lambda c: c is None, "Hands empty")
	# pickup_bag.add_precondition("Agent.location", lambda loc: loc == "Home", "At Home")
	# pickup_bag.add_precondition("Bag.location", lambda loc: loc == "Home", "Bag at Home")
	#
	# def pickup_bag_effect(val, state):
	# 	# Create carrying relation: Bag follows Agent
	# 	relation = Relation(
	# 		RelationType.EQUALS,
	# 		"Bag.location",
	# 		"Agent.location"
	# 	)
	# 	state.add_relation(relation)
	#
	# 	# Return new carrying value
	# 	return "Bag"
	#
	# pickup_bag.add_effect("Agent.carrying", pickup_bag_effect, "Pick up Bag")
	# pickup_bag.set_cost_function(lambda s, p: 1.0)
	# templates.append(pickup_bag)
	#
	# # ===== PutDown Action =====
	#
	# putdown = ActionTemplate("PutDown", [])
	# putdown.add_precondition("Agent.carrying", lambda c: c is not None, "Must be carrying")
	#
	# def putdown_effect(val, state):
	# 	agent = state.get_object("Agent")
	# 	if not agent:
	# 		return None
	#
	# 	carrying_var = agent.get_attribute("carrying")
	# 	if not carrying_var or not carrying_var.value:
	# 		return None
	#
	# 	carried_id = carrying_var.value
	#
	# 	# Remove the carrying relation
	# 	state.remove_relation(f"{carried_id}.location", "Agent.location")
	#
	# 	# Clear carrying
	# 	return None
	#
	# putdown.add_effect("Agent.carrying", putdown_effect, "Put down object")
	# putdown.set_cost_function(lambda s, p: 1.0)
	# templates.append(putdown)


	# ===== PutIn Actions (one per letter) =====

	for i in range(1, 4):
		putin = ActionTemplate(f"PutLetter{i}InBag", [])
		putin.add_precondition("Agent.location", lambda loc: loc == "Home", "At Home")
		putin.add_precondition("Bag.location", lambda loc: loc == "Home", "Bag at Home")
		putin.add_precondition("Agent.carrying", lambda c: c == f"Letter{i}", f"Carrying Letter{i}")

		def make_putin_effect(letter_id):
			def effect(val, state):
				agent = state.get_object("Agent")
				if not agent:
					return val

				# Remove carrying relation (Letter no longer follows Agent)
				state.remove_relation(f"{letter_id}.location", "Agent.location")

				# Create containment relation (Letter now follows Bag)
				relation = Relation(
					RelationType.EQUALS,
					f"{letter_id}.location",
					"Bag.location"
				)
				state.add_relation(relation)

				# Clear carrying
				return None

			return effect

		putin.add_effect("Agent.carrying", make_putin_effect(f"Letter{i}"), f"Put Letter{i} in Bag")
		putin.set_cost_function(lambda s, p: 1.0)
		templates.append(putin)

	return templates


def create_goal():
	"""Goal: All 3 letters at Work"""
	def satisfaction(state):
		count = 0
		for i in range(1, 4):
			letter = state.get_object(f"Letter{i}")
			if letter:
				loc_var = letter.get_attribute("location")
				if loc_var and loc_var.value == "Work":
					count += 1
		return count / 3.0

	return Goal("Get 3 letters to Work", satisfaction, priority=1.0)


def run_test():
	"""Run the letter example with leverage discovery"""
	print("\n" + "=" * 70)
	print("Letter Transportation with Leverage Discovery")
	print("=" * 70)

	# Create world
	state = create_world()

	print("\nInitial state:")
	print(f"  Agent: location={state.get_variable('Agent.location')}, carrying={state.get_variable('Agent.carrying')}")
	for i in range(1, 4):
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
	plan = planner.plan(state, goals, max_depth=20, max_samples=2000, beam_width=10)

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

		# Execute plan
		print("\nExecuting plan...")
		current_state = state
		for i, action in enumerate(plan, 1):
			current_state = action.apply(current_state)

			# Show state after key actions
			action_str = str(action)
			if "Bag" in action_str or "GoTo" in action_str or i == len(plan):
				print(f"\n  After step {i} ({action}):")
				print(f"    Agent: loc={current_state.get_variable('Agent.location')}, "
					  f"carrying={current_state.get_variable('Agent.carrying')}")
				for j in range(1, 4):
					print(f"    Letter{j}: loc={current_state.get_variable(f'Letter{j}.location')}")
				print(f"    Bag: loc={current_state.get_variable('Bag.location')}")

				# Show leverage
				bag_leverage = current_state.get_influence_count("Bag.location")
				if bag_leverage > 0:
					print(f"    >> Bag leverage: {bag_leverage} (affects {bag_leverage} letters!)")

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
