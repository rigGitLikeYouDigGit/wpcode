"""
Relations between state variables.

Relations are first-class elements that define how variables affect each other.
Changes to one variable can automatically propagate to related variables.
"""

from typing import Optional, Callable, Dict, List, Set
from dataclasses import dataclass, field
from enum import Enum


class RelationType(Enum):
	"""Types of relations between variables"""
	EQUALS = "equals"           # dependent == base (containment, carrying)
	DEPENDS_ON = "depends_on"   # dependent = f(base) (general dependency)
	DERIVED = "derived"         # dependent computed from multiple sources


@dataclass
class Relation:
	"""
	A relationship between state variables.

	When base variable changes, dependent variable may need to update.
	"""
	relation_type: RelationType
	dependent: str              # Variable that depends (e.g., "Letter1.location")
	base: str                   # Variable depended on (e.g., "Bag.location")

	# For derived relations, function to compute new value
	compute_fn: Optional[Callable] = None

	# Metadata
	strength: float = 1.0       # Strength of relation (for influence calculation)

	def __hash__(self):
		return hash((self.relation_type, self.dependent, self.base))

	def __eq__(self, other):
		if not isinstance(other, Relation):
			return False
		return (self.relation_type == other.relation_type and
				self.dependent == other.dependent and
				self.base == other.base)

	def __repr__(self):
		return f"Relation({self.relation_type.value}: {self.dependent} <- {self.base})"


class RelationGraph:
	"""
	Manages relations between variables and handles propagation.

	Key features:
	- Detects infinite loops using propagation_source tracking
	- Greedy propagation (simple, no lazy evaluation)
	- Integrated with reversible state
	"""

	def __init__(self):
		self.relations: List[Relation] = []

		# Forward index: base -> List[Relation]
		# "What depends on this variable?"
		self.influenced_by: Dict[str, List[Relation]] = {}

		# Reverse index: dependent -> Relation
		# "What does this variable depend on?"
		self.depends_on: Dict[str, Relation] = {}

	def add_relation(self, relation: Relation):
		"""Add a relation to the graph"""
		if relation in self.relations:
			return  # Already exists

		self.relations.append(relation)

		# Update forward index
		if relation.base not in self.influenced_by:
			self.influenced_by[relation.base] = []
		self.influenced_by[relation.base].append(relation)

		# Update reverse index
		self.depends_on[relation.dependent] = relation

	def remove_relation(self, dependent: str, base: str):
		"""Remove a relation by dependent and base variable"""
		# Find the relation
		relation = None
		for rel in self.relations:
			if rel.dependent == dependent and rel.base == base:
				relation = rel
				break

		if not relation:
			return

		# Remove from list
		self.relations.remove(relation)

		# Update forward index
		if base in self.influenced_by:
			self.influenced_by[base] = [r for r in self.influenced_by[base] if r != relation]
			if not self.influenced_by[base]:
				del self.influenced_by[base]

		# Update reverse index
		if dependent in self.depends_on:
			del self.depends_on[dependent]

	def remove_all_for_variable(self, var_path: str):
		"""Remove all relations involving a variable (as base or dependent)"""
		# Remove where var is dependent
		if var_path in self.depends_on:
			rel = self.depends_on[var_path]
			self.remove_relation(rel.dependent, rel.base)

		# Remove where var is base
		if var_path in self.influenced_by:
			# Copy list since we're modifying during iteration
			for rel in list(self.influenced_by[var_path]):
				self.remove_relation(rel.dependent, rel.base)

	def get_influenced_variables(self, base: str) -> List[str]:
		"""Get list of variables that depend on base"""
		if base not in self.influenced_by:
			return []
		return [rel.dependent for rel in self.influenced_by[base]]

	def get_base_variable(self, dependent: str) -> Optional[str]:
		"""Get the base variable that dependent depends on"""
		if dependent not in self.depends_on:
			return None
		return self.depends_on[dependent].base

	def has_relation(self, dependent: str, base: str) -> bool:
		"""Check if a relation exists"""
		return any(r.dependent == dependent and r.base == base for r in self.relations)

	def propagate(self, var_path: str, new_value, state, propagation_source: int = None):
		"""
		Propagate a variable change to all dependent variables.

		Args:
			var_path: Variable that changed
			new_value: New value
			state: State object (needs get_variable/set_variable methods)
			propagation_source: Hash to detect infinite loops

		Raises:
			RuntimeError: If infinite loop detected
		"""
		# Generate propagation source if not provided
		if propagation_source is None:
			propagation_source = hash((var_path, new_value, id(self)))

		# Check for infinite loop
		if hasattr(state, '_last_propagation_source'):
			last_source_var = state._last_propagation_source.get(var_path)
			if last_source_var == propagation_source:
				raise RuntimeError(
					f"Infinite propagation loop detected for variable {var_path}! "
					f"This suggests a circular dependency in relations."
				)

		# Track this propagation
		if not hasattr(state, '_last_propagation_source'):
			state._last_propagation_source = {}
		state._last_propagation_source[var_path] = propagation_source

		# Get all relations where this variable is the base
		if var_path not in self.influenced_by:
			return  # Nothing depends on this variable

		# Propagate to each dependent
		for relation in self.influenced_by[var_path]:
			if relation.relation_type == RelationType.EQUALS:
				# Simple equality: dependent gets same value as base
				# Set the dependent variable (propagate=False to avoid re-entering propagation)
				state.set_variable(relation.dependent, new_value, propagate=False)
				# Recursive propagation to dependents of this dependent
				self.propagate(relation.dependent, new_value, state, propagation_source)

			elif relation.relation_type == RelationType.DERIVED:
				# Computed value
				if relation.compute_fn:
					derived_value = relation.compute_fn(state)
					state.set_variable(relation.dependent, derived_value, propagate=False)
					self.propagate(relation.dependent, derived_value, state, propagation_source)

			elif relation.relation_type == RelationType.DEPENDS_ON:
				# Generic dependency - use compute_fn if available
				if relation.compute_fn:
					new_dep_value = relation.compute_fn(state, new_value)
					state.set_variable(relation.dependent, new_dep_value, propagate=False)
					self.propagate(relation.dependent, new_dep_value, state, propagation_source)

	def clear(self):
		"""Clear all relations"""
		self.relations.clear()
		self.influenced_by.clear()
		self.depends_on.clear()

	def get_influence_count(self, var_path: str) -> int:
		"""
		Get number of variables directly influenced by var_path.
		Used for leverage calculation.
		"""
		return len(self.influenced_by.get(var_path, []))

	def get_transitive_influence_count(self, var_path: str, visited: Optional[Set[str]] = None) -> int:
		"""
		Get total number of variables transitively influenced by var_path.
		Used for full leverage calculation.
		"""
		if visited is None:
			visited = set()

		if var_path in visited:
			return 0

		visited.add(var_path)
		count = 0

		# Add direct influences
		direct = self.influenced_by.get(var_path, [])
		count += len(direct)

		# Add transitive influences
		for rel in direct:
			count += self.get_transitive_influence_count(rel.dependent, visited)

		return count


# Helper functions for common relation patterns

def create_containment_relation(container_var: str, contained_var: str) -> Relation:
	"""
	Create a relation for containment (object in container).
	contained.location == container.location
	"""
	return Relation(
		RelationType.EQUALS,
		dependent=contained_var,
		base=container_var,
		strength=1.0
	)


def create_carrying_relation(agent_carrying_var: str, object_location_var: str, agent_location_var: str) -> Relation:
	"""
	Create a relation for carrying (agent carries object).
	object.location == agent.location
	"""
	return Relation(
		RelationType.EQUALS,
		dependent=object_location_var,
		base=agent_location_var,
		strength=1.0
	)


def create_derived_relation(dependent_var: str, source_vars: List[str], compute_fn: Callable) -> Relation:
	"""
	Create a derived relation where dependent is computed from multiple sources.
	Note: Only one base variable for now, compute_fn can access state for others.
	"""
	return Relation(
		RelationType.DERIVED,
		dependent=dependent_var,
		base=source_vars[0] if source_vars else "",
		compute_fn=compute_fn,
		strength=1.0
	)
