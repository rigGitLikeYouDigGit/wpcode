"""
Actions: property-based templates with preconditions and effects
"""

from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass, field
from wpplan.state import State, Variable, ObjectState


@dataclass
class Precondition:
	"""A condition that must be true before an action can execute"""
	variable_path: str  # e.g., "Agent.location" or "fuel"
	predicate: Callable[[Any], bool]  # Function that checks the condition
	description: str = ""

	def is_satisfied(self, state: State) -> bool:
		"""Check if precondition is satisfied in given state"""
		# Parse variable_path to get the actual value
		parts = self.variable_path.split('.')
		if len(parts) == 1:
			# Global variable
			var = state.get_global(parts[0])
			if var is None:
				return False
			return self.predicate(var.value)
		else:
			# Object attribute
			obj = state.get_object(parts[0])
			if obj is None:
				return False
			var = obj.get_attribute(parts[1])
			if var is None:
				return False
			return self.predicate(var.value)


@dataclass
class Effect:
	"""An effect that modifies state when action executes"""
	variable_path: str
	effect_fn: Callable[[Any, State], Any]  # Takes current value and state, returns new value
	description: str = ""

	def apply(self, state: State):
		"""Apply this effect to the state"""
		parts = self.variable_path.split('.')
		if len(parts) == 1:
			# Global variable
			var = state.get_global(parts[0])
			if var is not None:
				new_value = self.effect_fn(var.value, state)
				# Use set_variable if available (ReversibleState) for proper propagation
				if hasattr(state, 'set_variable'):
					state.set_variable(parts[0], new_value)
				else:
					var.value = new_value
		else:
			# Object attribute
			obj = state.get_object(parts[0])
			if obj is not None:
				var = obj.get_attribute(parts[1])
				if var is not None:
					new_value = self.effect_fn(var.value, state)
					# Use set_variable if available (ReversibleState) for proper propagation
					if hasattr(state, 'set_variable'):
						state.set_variable(self.variable_path, new_value)
					else:
						var.value = new_value


@dataclass
class PropertyRequirement:
	"""Requirement on object properties for action applicability"""
	param_name: str  # Which parameter this applies to
	required_properties: List[str]  # Properties the object must have
	forbidden_properties: List[str] = field(default_factory=list)  # Properties the object must NOT have


class ActionTemplate:
	"""A template for actions that can be instantiated with specific objects
	TODO: generalise to accept any number of participants, given local names
		within local scope of action
	"""

	def __init__(self, name: str, param_names: List[str]):
		self.name = name
		self.param_names = param_names
		self.property_requirements: List[PropertyRequirement] = []
		self.preconditions: List[Precondition] = []
		self.effects: List[Effect] = []
		self.cost_fn: Callable[[State, Dict[str, Any]], float] = lambda s, p: 1.0

	def add_property_requirement(self, param_name: str,
								 required: List[str],
								 forbidden: List[str] = None):
		"""Add property requirements for a parameter"""
		self.property_requirements.append(
			PropertyRequirement(param_name, required, forbidden or [])
		)

	def add_precondition(self, var_path: str, predicate: Callable, description: str = ""):
		"""Add a precondition"""
		self.preconditions.append(Precondition(var_path, predicate, description))

	def add_effect(self, var_path: str, effect_fn: Callable, description: str = ""):
		"""Add an effect"""
		self.effects.append(Effect(var_path, effect_fn, description))

	def set_cost_function(self, cost_fn: Callable[[State, Dict[str, Any]], float]):
		"""Set the cost function for this action"""
		self.cost_fn = cost_fn

	def instantiate(self, state: State, params: Dict[str, Any]) -> Optional['Action']:
		"""
		Try to instantiate this template with specific parameters.
		Returns None if parameters don't satisfy property requirements.
		"""
		# Check property requirements
		for req in self.property_requirements:
			param_value = params.get(req.param_name)
			if param_value is None:
				return None

			# If param is an object ID, get the object
			obj = state.get_object(param_value) if isinstance(param_value, str) else None
			if obj is None:
				continue

			# Check required properties
			for prop in req.required_properties:
				if not obj.has_property(prop):
					return None

			# Check forbidden properties
			for prop in req.forbidden_properties:
				if obj.has_property(prop):
					return None

		# Create instantiated action
		return Action(self, params, state)


class Action:
	"""An instantiated action with specific parameters"""

	def __init__(self, template: ActionTemplate, params: Dict[str, Any], state: State):
		self.template = template
		self.params = params
		self._preconditions_cache = None
		self._cost_cache = None
		self._instantiate_preconditions_and_effects(state)

	def _instantiate_preconditions_and_effects(self, state: State):
		"""Replace parameter placeholders in preconditions/effects"""
		# For now, store them directly
		# In full implementation, would need to substitute parameter values into paths
		self.preconditions = self.template.preconditions
		self.effects = self.template.effects

	def is_applicable(self, state: State) -> bool:
		"""Check if all preconditions are satisfied"""
		return all(pre.is_satisfied(state) for pre in self.preconditions)

	def apply(self, state: State) -> State:
		"""Apply this action to a state, returning new state"""
		new_state = state.copy()
		for effect in self.effects:
			effect.apply(new_state)
		return new_state

	def cost(self, state: State) -> float:
		"""Get the cost of executing this action"""
		if self._cost_cache is None:
			self._cost_cache = self.template.cost_fn(state, self.params)
		return self._cost_cache

	def __str__(self):
		param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
		return f"{self.template.name}({param_str})"

	def __repr__(self):
		return self.__str__()
