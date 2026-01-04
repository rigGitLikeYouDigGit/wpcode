"""
State representation: discrete and continuous variables
"""

from typing import Any, Dict, Set, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class ValueType(Enum):
	"""Type of a state variable"""
	DISCRETE = "discrete"  # e.g., location = "Home"
	CONTINUOUS = "continuous"  # e.g., fuel = 75.5
	SET = "set"  # e.g., bag.contents = {Letter1, Letter2}


@dataclass
class Variable:
	"""A state variable that can change over time"""
	name: str
	value_type: ValueType
	value: Any

	# Constraints for continuous variables
	min_value: Optional[float] = None
	max_value: Optional[float] = None

	# Dependencies: which variables does this one depend on?
	# e.g., Letter1.location depends on Bag.location if Letter1 in Bag
	depends_on: Set[str] = field(default_factory=set)

	def __hash__(self):
		return hash(self.name)

	def satisfies_constraint(self, constraint) -> bool:
		"""Check if current value satisfies a constraint"""
		if self.value_type == ValueType.CONTINUOUS:
			if self.min_value is not None and self.value < self.min_value:
				return False
			if self.max_value is not None and self.value > self.max_value:
				return False
		return constraint(self.value) if callable(constraint) else True


@dataclass
class ObjectState:
	"""State of a single object in the world"""
	id: str
	object_type: str
	properties: Set[str] = field(default_factory=set)  # e.g., {"Container", "Portable"}
	attributes: Dict[str, Variable] = field(default_factory=dict)  # e.g., {"location": Variable(...)}

	def has_property(self, prop: str) -> bool:
		return prop in self.properties

	def get_attribute(self, name: str) -> Optional[Variable]:
		return self.attributes.get(name)

	def set_attribute(self, name: str, var: Variable):
		self.attributes[name] = var

	def __hash__(self):
		return hash(self.id)

class State:
	"""Complete world state at a point in time"""

	def __init__(self):
		self.objects: Dict[str, ObjectState] = {}
		self.global_vars: Dict[str, Variable] = {}  # e.g., current_time
		self.type_object_map: Dict[str, Set[ObjectState]] = {}
		self.property_type_object_map: Dict[str, Set[ObjectState]] = {}

	def add_object(self, obj: ObjectState):
		"""Add an object to the state"""
		self.objects[obj.id] = obj
		self.type_object_map.setdefault(obj.object_type, set()).add(obj)
		for prop in obj.properties:
			self.property_type_object_map.setdefault(prop, set()).add(obj)

	def get_object(self, obj_id: str) -> Optional[ObjectState]:
		"""Get object by ID"""
		return self.objects.get(obj_id)

	def get_objects_with_property(self, prop: str) -> list[ObjectState]:
		"""Get all objects that have a given property"""
		return [obj for obj in self.objects.values() if obj.has_property(prop)]

	def get_objects_of_type(self, obj_type: str) -> list[ObjectState]:
		"""Get all objects of a given type"""
		return [obj for obj in self.objects.values() if obj.object_type == obj_type]

	def set_global(self, name: str, var: Variable):
		"""Set a global variable"""
		self.global_vars[name] = var

	def get_global(self, name: str) -> Optional[Variable]:
		"""Get a global variable"""
		return self.global_vars.get(name)

	def copy(self) -> 'State':
		"""Create a copy of this state for simulation"""
		# TODO: Implement proper deep copy
		# For now, shallow copy for structure
		import copy
		return copy.deepcopy(self)

	def get_all_variables(self) -> Dict[str, Variable]:
		"""Get all variables in the state (for influence analysis)"""
		variables = dict(self.global_vars)
		for obj in self.objects.values():
			for attr_name, var in obj.attributes.items():
				variables[f"{obj.id}.{attr_name}"] = var
		return variables


@dataclass
class Value:
	"""A value that can be discrete, continuous, or a set"""
	value: Any
	value_type: ValueType

	def matches(self, other: 'Value') -> bool:
		"""Check if this value matches another (for pattern matching)"""
		if self.value_type != other.value_type:
			return False

		if self.value_type == ValueType.SET:
			# Set matching: check containment or overlap depending on context
			return bool(self.value & other.value)
		else:
			return self.value == other.value

	def in_range(self, min_val: Optional[float], max_val: Optional[float]) -> bool:
		"""Check if continuous value is in range"""
		if self.value_type != ValueType.CONTINUOUS:
			return True
		if min_val is not None and self.value < min_val:
			return False
		if max_val is not None and self.value > max_val:
			return False
		return True
