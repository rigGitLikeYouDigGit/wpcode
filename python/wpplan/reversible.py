"""
Reversible state operations for efficient search.

Instead of deep copying state for every action simulation,
we track changes and can efficiently undo them.
"""

from typing import Any, Optional, List, Tuple
from dataclasses import dataclass, field
from wpplan.state import State, Variable, ObjectState
from wpplan.relations import Relation, RelationGraph
from wpplan.state import Variable, ValueType


@dataclass
class VariableChange:
	"""Record of a variable value change"""
	path: str
	old_value: Any

	def __repr__(self):
		return f"VariableChange({self.path}: {self.old_value})"


@dataclass
class DependencyChange:
	"""Record of a dependency graph change"""
	operation: str  # "add" or "remove"
	dependent: str
	depends_on: str

	def __repr__(self):
		return f"DependencyChange({self.operation}: {self.dependent} -> {self.depends_on})"


@dataclass
class SetChange:
	"""Record of a set modification (for container contents, etc.)"""
	path: str
	operation: str  # "add" or "remove"
	element: Any

	def __repr__(self):
		return f"SetChange({self.path}.{self.operation}({self.element}))"


@dataclass
class RelationChange:
	"""Record of a relation graph change"""
	operation: str  # "add" or "remove"
	relation: Relation

	def __repr__(self):
		return f"RelationChange({self.operation}: {self.relation})"


class Transaction:
	"""
	Records all changes made during a transaction.
	Can be rolled back to restore original state.
	"""

	def __init__(self):
		self.variable_changes: List[VariableChange] = []
		self.dependency_changes: List[DependencyChange] = []
		self.set_changes: List[SetChange] = []
		self.relation_changes: List[RelationChange] = []

	def record_variable_change(self, path: str, old_value: Any):
		"""Record a variable value change"""
		self.variable_changes.append(VariableChange(path, old_value))

	def record_dependency_change(self, operation: str, dependent: str, depends_on: str):
		"""Record a dependency graph modification"""
		self.dependency_changes.append(DependencyChange(operation, dependent, depends_on))

	def record_set_change(self, path: str, operation: str, element: Any):
		"""Record a set add/remove operation"""
		self.set_changes.append(SetChange(path, operation, element))

	def record_relation_change(self, operation: str, relation: Relation):
		"""Record a relation graph modification"""
		self.relation_changes.append(RelationChange(operation, relation))

	def __len__(self):
		return (len(self.variable_changes) + len(self.dependency_changes) +
				len(self.set_changes) + len(self.relation_changes))


class ReversibleState(State):
	"""
	State that supports efficient undo of operations.

	Now includes relation graph for automatic propagation.

	Usage:
		state.begin_transaction()
		# Make changes...
		state.set_variable("Agent.location", "Work")
		# Automatically propagates through relations
		# Evaluate result...
		if good:
			state.commit()  # Keep changes
		else:
			state.rollback()  # Undo changes
	"""

	def __init__(self):
		super().__init__()
		self.transaction_stack: List[Transaction] = []
		self.relations = RelationGraph()
		self._last_propagation_source = {}  # For loop detection

	def begin_transaction(self) -> Transaction:
		"""
		Start a new transaction.
		All subsequent changes will be recorded until commit() or rollback().
		"""
		transaction = Transaction()
		self.transaction_stack.append(transaction)
		return transaction

	def commit(self):
		"""
		Commit the current transaction.
		Changes are kept, but no longer tracked for undo.
		"""
		if self.transaction_stack:
			self.transaction_stack.pop()

	def rollback(self):
		"""
		Roll back the current transaction.
		All changes since begin_transaction() are undone in reverse order.
		"""
		if not self.transaction_stack:
			return

		transaction = self.transaction_stack.pop()

		# Undo relation changes first (affects propagation)
		for change in reversed(transaction.relation_changes):
			self._undo_relation_change(change)

		# Undo set changes
		for change in reversed(transaction.set_changes):
			self._undo_set_change(change)

		# Undo dependency changes
		for change in reversed(transaction.dependency_changes):
			self._undo_dependency_change(change)

		# Undo variable changes
		for change in reversed(transaction.variable_changes):
			self._undo_variable_change(change)

	def _current_transaction(self) -> Optional[Transaction]:
		"""Get the current active transaction, if any"""
		return self.transaction_stack[-1] if self.transaction_stack else None

	# Override State methods to add transaction tracking

	def set_variable(self, path: str, value: Any, propagate: bool = True):
		"""
		Set a variable value, recording the change if in a transaction.
		Automatically propagates through relations if propagate=True.

		Path format: "object_id.attribute" or "global_var_name"
		"""
		transaction = self._current_transaction()

		# Record old value if in transaction
		if transaction is not None:
			old_value = self.get_variable(path)
			if old_value != value:  # Only record actual changes
				transaction.record_variable_change(path, old_value)

		# Apply change
		parts = path.split('.')
		if len(parts) == 1:
			# Global variable
			var = self.get_global(parts[0])
			if var is not None:
				var.value = value
			else:
				# Create new global variable if it doesn't exist
				self.set_global(parts[0], Variable(parts[0], ValueType.DISCRETE, value))
		else:
			# Object attribute
			obj = self.get_object(parts[0])
			if obj is not None:
				var = obj.get_attribute(parts[1])
				if var is not None:
					var.value = value

		# Propagate to related variables
		if propagate:
			self.relations.propagate(path, value, self)

	def get_variable(self, path: str) -> Any:
		"""Get a variable value by path"""
		parts = path.split('.')
		if len(parts) == 1:
			var = self.get_global(parts[0])
			return var.value if var else None
		else:
			obj = self.get_object(parts[0])
			if obj is not None:
				var = obj.get_attribute(parts[1])
				return var.value if var else None
		return None

	def add_dependency(self, dependent: str, depends_on: str):
		"""
		Add a dependency between variables.
		E.g., Letter1.location depends on Bag.location
		"""
		transaction = self._current_transaction()

		# Record for undo
		if transaction is not None:
			transaction.record_dependency_change("add", dependent, depends_on)

		# Get variables
		dep_var = self._get_variable_object(dependent)
		base_var = self._get_variable_object(depends_on)

		if dep_var is not None and base_var is not None:
			dep_var.depends_on.add(depends_on)

	def remove_dependency(self, dependent: str, depends_on: str):
		"""Remove a dependency between variables"""
		transaction = self._current_transaction()

		# Record for undo
		if transaction is not None:
			transaction.record_dependency_change("remove", dependent, depends_on)

		# Get variables
		dep_var = self._get_variable_object(dependent)

		if dep_var is not None:
			dep_var.depends_on.discard(depends_on)

	def add_to_set(self, path: str, element: Any):
		"""
		Add an element to a set variable (e.g., container contents).
		"""
		transaction = self._current_transaction()

		# Record for undo
		if transaction is not None:
			transaction.record_set_change(path, "add", element)

		# Apply change
		value = self.get_variable(path)
		if isinstance(value, set):
			value.add(element)

	def remove_from_set(self, path: str, element: Any):
		"""Remove an element from a set variable"""
		transaction = self._current_transaction()

		# Record for undo
		if transaction is not None:
			transaction.record_set_change(path, "remove", element)

		# Apply change
		value = self.get_variable(path)
		if isinstance(value, set):
			value.discard(element)

	# Relation management

	def add_relation(self, relation: Relation):
		"""Add a relation between variables"""
		transaction = self._current_transaction()

		# Record for undo
		if transaction is not None:
			transaction.record_relation_change("add", relation)

		# Apply change
		self.relations.add_relation(relation)

	def remove_relation(self, dependent: str, base: str):
		"""Remove a relation"""
		transaction = self._current_transaction()

		# Find the relation to record it
		if transaction is not None:
			for rel in self.relations.relations:
				if rel.dependent == dependent and rel.base == base:
					transaction.record_relation_change("remove", rel)
					break

		# Apply change
		self.relations.remove_relation(dependent, base)

	def get_influenced_variables(self, var_path: str) -> List[str]:
		"""Get variables that depend on var_path (for leverage calculation)"""
		return self.relations.get_influenced_variables(var_path)

	def get_influence_count(self, var_path: str) -> int:
		"""Get count of variables influenced by var_path"""
		return self.relations.get_influence_count(var_path)

	def get_transitive_influence_count(self, var_path: str) -> int:
		"""Get total transitive influence of var_path"""
		return self.relations.get_transitive_influence_count(var_path)

	# Undo operations

	def _undo_variable_change(self, change: VariableChange):
		"""Undo a variable value change"""
		# Temporarily disable transaction recording for undo
		saved_stack = self.transaction_stack
		self.transaction_stack = []

		self.set_variable(change.path, change.old_value)

		self.transaction_stack = saved_stack

	def _undo_dependency_change(self, change: DependencyChange):
		"""Undo a dependency graph change"""
		# Temporarily disable transaction recording
		saved_stack = self.transaction_stack
		self.transaction_stack = []

		if change.operation == "add":
			# Was added, so remove it
			self.remove_dependency(change.dependent, change.depends_on)
		else:
			# Was removed, so add it back
			self.add_dependency(change.dependent, change.depends_on)

		self.transaction_stack = saved_stack

	def _undo_relation_change(self, change: RelationChange):
		"""Undo a relation graph change"""
		# Temporarily disable transaction recording
		saved_stack = self.transaction_stack
		self.transaction_stack = []

		if change.operation == "add":
			# Was added, so remove it
			self.relations.remove_relation(change.relation.dependent, change.relation.base)
		else:
			# Was removed, so add it back
			self.relations.add_relation(change.relation)

		self.transaction_stack = saved_stack

	def _undo_set_change(self, change: SetChange):
		"""Undo a set modification"""
		# Temporarily disable transaction recording
		saved_stack = self.transaction_stack
		self.transaction_stack = []

		if change.operation == "add":
			# Was added, so remove it
			self.remove_from_set(change.path, change.element)
		else:
			# Was removed, so add it back
			self.add_to_set(change.path, change.element)

		self.transaction_stack = saved_stack

	def _get_variable_object(self, path: str) -> Optional[Variable]:
		"""Get the Variable object for a path"""
		parts = path.split('.')
		if len(parts) == 1:
			return self.get_global(parts[0])
		else:
			obj = self.get_object(parts[0])
			if obj is not None:
				return obj.get_attribute(parts[1])
		return None

	def copy(self) -> 'ReversibleState':
		"""
		Create a deep copy of this state.
		Should only be needed when committing to keep a state permanently.
		"""
		import copy
		new_state = ReversibleState()
		new_state.objects = copy.deepcopy(self.objects)
		new_state.global_vars = copy.deepcopy(self.global_vars)
		# Copy relations!
		new_state.relations = copy.deepcopy(self.relations)
		# Don't copy transaction stack
		return new_state

	def snapshot(self) -> 'StateSnapshot':
		"""
		Create a lightweight snapshot that can be restored.
		More efficient than copy() for temporary exploration.
		"""
		return StateSnapshot(self)


class StateSnapshot:
	"""
	Lightweight snapshot of state for quick restore.
	More efficient than full copy for temporary exploration.
	"""

	def __init__(self, state: ReversibleState):
		self.state = state
		# Just record current transaction depth
		self.transaction_depth = len(state.transaction_stack)

	def restore(self):
		"""
		Restore state to snapshot point.
		Rolls back all transactions created after snapshot.
		"""
		while len(self.state.transaction_stack) > self.transaction_depth:
			self.state.rollback()


# Helper functions for common patterns

def try_action(state: ReversibleState, action, evaluation_fn):
	"""
	Try an action and evaluate the result, then undo.

	Args:
		state: Reversible state
		action: Action to try
		evaluation_fn: Function that evaluates the resulting state

	Returns:
		Result of evaluation_fn
	"""
	state.begin_transaction()
	try:
		action.apply(state)
		result = evaluation_fn(state)
		return result
	finally:
		state.rollback()


def commit_action(state: ReversibleState, action):
	"""
	Apply an action and commit the changes.

	Args:
		state: Reversible state
		action: Action to apply permanently
	"""
	state.begin_transaction()
	action.apply(state)
	state.commit()
