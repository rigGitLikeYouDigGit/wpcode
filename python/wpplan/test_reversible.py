"""
Tests for reversible state operations
"""

from wpplan.reversible import ReversibleState, try_action
from wpplan.state import Variable, ValueType, ObjectState


def test_basic_transaction():
	"""Test basic transaction commit and rollback"""
	state = ReversibleState()

	# Add initial object
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

	initial_location = state.get_variable("Agent.location")
	print(f"Initial location: {initial_location}")
	assert initial_location == "Home"

	# Test rollback
	state.begin_transaction()
	state.set_variable("Agent.location", "Work")
	assert state.get_variable("Agent.location") == "Work"
	state.rollback()
	assert state.get_variable("Agent.location") == "Home", "Rollback failed!"
	print("✓ Rollback works")

	# Test commit
	state.begin_transaction()
	state.set_variable("Agent.location", "Work")
	state.commit()
	assert state.get_variable("Agent.location") == "Work", "Commit failed!"
	print("✓ Commit works")


def test_nested_transactions():
	"""Test nested transactions"""
	state = ReversibleState()

	agent = ObjectState(
		id="Agent",
		object_type="Agent",
		properties=set(),
		attributes={
			"location": Variable("Agent.location", ValueType.DISCRETE, "Home"),
			"fuel": Variable("Agent.fuel", ValueType.CONTINUOUS, 100.0)
		}
	)
	state.add_object(agent)

	# Outer transaction
	state.begin_transaction()
	state.set_variable("Agent.location", "Work")
	print(f"After outer change: location={state.get_variable('Agent.location')}")

	# Inner transaction
	state.begin_transaction()
	state.set_variable("Agent.fuel", 80.0)
	print(f"After inner change: fuel={state.get_variable('Agent.fuel')}")

	# Rollback inner
	state.rollback()
	assert state.get_variable("Agent.fuel") == 100.0, "Inner rollback failed!"
	assert state.get_variable("Agent.location") == "Work", "Outer transaction affected!"
	print("✓ Inner rollback works")

	# Rollback outer
	state.rollback()
	assert state.get_variable("Agent.location") == "Home", "Outer rollback failed!"
	print("✓ Outer rollback works")


def test_set_operations():
	"""Test set add/remove operations"""
	state = ReversibleState()

	bag = ObjectState(
		id="Bag",
		object_type="Bag",
		properties={"Container"},
		attributes={
			"contents": Variable("Bag.contents", ValueType.SET, set())
		}
	)
	state.add_object(bag)

	# Add to set
	state.begin_transaction()
	state.add_to_set("Bag.contents", "Letter1")
	state.add_to_set("Bag.contents", "Letter2")
	contents = state.get_variable("Bag.contents")
	print(f"After adds: {contents}")
	assert "Letter1" in contents and "Letter2" in contents

	# Rollback
	state.rollback()
	contents = state.get_variable("Bag.contents")
	print(f"After rollback: {contents}")
	assert len(contents) == 0, "Set rollback failed!"
	print("✓ Set operations rollback works")

	# Commit
	state.begin_transaction()
	state.add_to_set("Bag.contents", "Letter1")
	state.commit()
	contents = state.get_variable("Bag.contents")
	assert "Letter1" in contents
	print("✓ Set operations commit works")


def test_dependency_tracking():
	"""Test dependency add/remove with transactions"""
	state = ReversibleState()

	letter = ObjectState(
		id="Letter1",
		object_type="Letter",
		properties={"Portable"},
		attributes={
			"location": Variable("Letter1.location", ValueType.DISCRETE, "Home")
		}
	)
	state.add_object(letter)

	bag = ObjectState(
		id="Bag",
		object_type="Bag",
		properties={"Container"},
		attributes={
			"location": Variable("Bag.location", ValueType.DISCRETE, "Home")
		}
	)
	state.add_object(bag)

	# Add dependency
	state.begin_transaction()
	state.add_dependency("Letter1.location", "Bag.location")

	letter_var = state._get_variable_object("Letter1.location")
	assert "Bag.location" in letter_var.depends_on, "Dependency not added!"
	print(f"✓ Dependency added: {letter_var.depends_on}")

	# Rollback
	state.rollback()
	letter_var = state._get_variable_object("Letter1.location")
	assert "Bag.location" not in letter_var.depends_on, "Dependency rollback failed!"
	print("✓ Dependency rollback works")


def test_performance():
	"""Test performance of reversible vs copy"""
	import time

	state = ReversibleState()

	# Create many objects
	for i in range(100):
		obj = ObjectState(
			id=f"Object{i}",
			object_type="Item",
			properties={"Portable"},
			attributes={
				"location": Variable(f"Object{i}.location", ValueType.DISCRETE, "Home"),
				"value": Variable(f"Object{i}.value", ValueType.CONTINUOUS, float(i))
			}
		)
		state.add_object(obj)

	# Test reversible operations
	start = time.time()
	for i in range(1000):
		state.begin_transaction()
		state.set_variable(f"Object{i % 100}.location", "Work")
		state.set_variable(f"Object{i % 100}.value", float(i * 2))
		state.rollback()
	reversible_time = time.time() - start
	print(f"Reversible operations: {reversible_time:.3f}s")

	# Test copy operations
	start = time.time()
	for i in range(1000):
		new_state = state.copy()
		new_state.set_variable(f"Object{i % 100}.location", "Work")
		new_state.set_variable(f"Object{i % 100}.value", float(i * 2))
		# Discard new_state
	copy_time = time.time() - start
	print(f"Copy operations: {copy_time:.3f}s")

	speedup = copy_time / reversible_time
	print(f"✓ Speedup: {speedup:.1f}x")


def run_all_tests():
	"""Run all tests"""
	print("=" * 60)
	print("Testing Reversible State")
	print("=" * 60)

	print("\n1. Basic transactions:")
	test_basic_transaction()

	print("\n2. Nested transactions:")
	test_nested_transactions()

	print("\n3. Set operations:")
	test_set_operations()

	print("\n4. Dependency tracking:")
	test_dependency_tracking()

	print("\n5. Performance comparison:")
	test_performance()

	print("\n" + "=" * 60)
	print("All tests passed! ✓")
	print("=" * 60)


if __name__ == "__main__":
	run_all_tests()
