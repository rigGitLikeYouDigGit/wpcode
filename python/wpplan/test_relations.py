"""
Tests for the relation system
"""

from wpplan.reversible import ReversibleState
from wpplan.state import ObjectState, Variable, ValueType
from wpplan.relations import Relation, RelationType, create_containment_relation


def test_basic_propagation():
	"""Test that changes propagate through EQUALS relations"""
	print("\n=== Test: Basic Propagation ===")

	state = ReversibleState()

	# Create objects
	bag = ObjectState("Bag", "Bag", {"Container"}, {
		"location": Variable("Bag.location", ValueType.DISCRETE, "Home")
	})
	letter = ObjectState("Letter1", "Letter", {"Portable"}, {
		"location": Variable("Letter1.location", ValueType.DISCRETE, "Home")
	})

	state.add_object(bag)
	state.add_object(letter)

	print(f"Initial: Bag at {state.get_variable('Bag.location')}, Letter at {state.get_variable('Letter1.location')}")

	# Create containment relation
	relation = create_containment_relation("Bag.location", "Letter1.location")
	state.add_relation(relation)

	print(f"Added relation: Letter1.location == Bag.location")

	# Move bag
	state.set_variable("Bag.location", "Work")

	print(f"After moving Bag: Bag at {state.get_variable('Bag.location')}, Letter at {state.get_variable('Letter1.location')}")

	# Check propagation
	assert state.get_variable("Bag.location") == "Work"
	assert state.get_variable("Letter1.location") == "Work", "Propagation failed!"

	print("[PASS] Propagation works!")
	return True


def test_transitive_propagation():
	"""Test that propagation works transitively (bag in container in vehicle)"""
	print("\n=== Test: Transitive Propagation ===")

	state = ReversibleState()

	# Create chain: Letter in Bag in Car
	car = ObjectState("Car", "Car", set(), {
		"location": Variable("Car.location", ValueType.DISCRETE, "Home")
	})
	bag = ObjectState("Bag", "Bag", {"Container"}, {
		"location": Variable("Bag.location", ValueType.DISCRETE, "Home")
	})
	letter = ObjectState("Letter1", "Letter", {"Portable"}, {
		"location": Variable("Letter1.location", ValueType.DISCRETE, "Home")
	})

	state.add_object(car)
	state.add_object(bag)
	state.add_object(letter)

	# Create chain of relations
	state.add_relation(Relation(RelationType.EQUALS, "Bag.location", "Car.location"))
	state.add_relation(Relation(RelationType.EQUALS, "Letter1.location", "Bag.location"))

	print(f"Initial: Car={state.get_variable('Car.location')}, Bag={state.get_variable('Bag.location')}, Letter={state.get_variable('Letter1.location')}")
	print("Relations: Letter -> Bag -> Car")

	# Move car
	state.set_variable("Car.location", "Work")

	print(f"After moving Car: Car={state.get_variable('Car.location')}, Bag={state.get_variable('Bag.location')}, Letter={state.get_variable('Letter1.location')}")

	# Check transitive propagation
	assert state.get_variable("Car.location") == "Work"
	assert state.get_variable("Bag.location") == "Work", "First-level propagation failed!"
	assert state.get_variable("Letter1.location") == "Work", "Transitive propagation failed!"

	print("[PASS] Transitive propagation works!")
	return True


def test_reversible_relations():
	"""Test that relations can be undone with transactions"""
	print("\n=== Test: Reversible Relations ===")

	state = ReversibleState()

	bag = ObjectState("Bag", "Bag", {"Container"}, {
		"location": Variable("Bag.location", ValueType.DISCRETE, "Home")
	})
	letter = ObjectState("Letter1", "Letter", {"Portable"}, {
		"location": Variable("Letter1.location", ValueType.DISCRETE, "Home")
	})

	state.add_object(bag)
	state.add_object(letter)

	print(f"Initial: Bag at {state.get_variable('Bag.location')}, Letter at {state.get_variable('Letter1.location')}")

	# Test adding and removing relation
	state.begin_transaction()

	relation = create_containment_relation("Bag.location", "Letter1.location")
	state.add_relation(relation)
	print("Added relation in transaction")

	# Move bag - should propagate
	state.set_variable("Bag.location", "Work")
	print(f"After move: Bag at {state.get_variable('Bag.location')}, Letter at {state.get_variable('Letter1.location')}")
	assert state.get_variable("Letter1.location") == "Work", "Propagation failed!"

	# Rollback
	state.rollback()
	print("Rolled back transaction")

	print(f"After rollback: Bag at {state.get_variable('Bag.location')}, Letter at {state.get_variable('Letter1.location')}")

	# Check everything reverted
	assert state.get_variable("Bag.location") == "Home", "Bag location not reverted!"
	assert state.get_variable("Letter1.location") == "Home", "Letter location not reverted!"

	# Move bag again - should NOT propagate (relation removed)
	state.set_variable("Bag.location", "Work")
	print(f"After second move: Bag at {state.get_variable('Bag.location')}, Letter at {state.get_variable('Letter1.location')}")
	assert state.get_variable("Letter1.location") == "Home", "Relation not properly removed!"

	print("[PASS] Reversible relations work!")
	return True


def test_influence_counting():
	"""Test leverage/influence calculation"""
	print("\n=== Test: Influence Counting ===")

	state = ReversibleState()

	# Create bag with multiple letters
	bag = ObjectState("Bag", "Bag", {"Container"}, {
		"location": Variable("Bag.location", ValueType.DISCRETE, "Home")
	})
	state.add_object(bag)

	for i in range(1, 4):
		letter = ObjectState(f"Letter{i}", "Letter", {"Portable"}, {
			"location": Variable(f"Letter{i}.location", ValueType.DISCRETE, "Home")
		})
		state.add_object(letter)
		# Add containment relation
		state.add_relation(Relation(RelationType.EQUALS, f"Letter{i}.location", "Bag.location"))

	print(f"Created Bag with 3 letters inside")

	# Check influence count
	influence_count = state.get_influence_count("Bag.location")
	print(f"Bag.location directly influences {influence_count} variables")
	assert influence_count == 3, f"Expected 3, got {influence_count}"

	# Check transitive influence
	transitive = state.get_transitive_influence_count("Bag.location")
	print(f"Bag.location transitively influences {transitive} variables")
	assert transitive == 3, f"Expected 3, got {transitive}"

	print(f"[PASS] Influence counting works! (leverage = {influence_count})")
	return True


def test_loop_detection():
	"""Test that circular relations are detected"""
	print("\n=== Test: Loop Detection ===")

	state = ReversibleState()

	# Create circular dependency: A → B → A
	objA = ObjectState("A", "Object", set(), {
		"value": Variable("A.value", ValueType.DISCRETE, 1)
	})
	objB = ObjectState("B", "Object", set(), {
		"value": Variable("B.value", ValueType.DISCRETE, 1)
	})

	state.add_object(objA)
	state.add_object(objB)

	# Create circular relations
	state.add_relation(Relation(RelationType.EQUALS, "B.value", "A.value"))
	state.add_relation(Relation(RelationType.EQUALS, "A.value", "B.value"))

	print("Created circular relation: A.value -> B.value -> A.value")

	# Try to set - should raise error
	try:
		state.set_variable("A.value", 2)
		print("[FAIL] Loop not detected!")
		return False
	except RuntimeError as e:
		print(f"Caught error: {e}")
		print("[PASS] Loop detection works!")
		return True


def run_all_tests():
	"""Run all relation tests"""
	print("=" * 60)
	print("Relation System Tests")
	print("=" * 60)

	tests = [
		test_basic_propagation,
		test_transitive_propagation,
		test_reversible_relations,
		test_influence_counting,
		test_loop_detection,
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

	return passed == len(tests)


if __name__ == "__main__":
	success = run_all_tests()
	exit(0 if success else 1)
