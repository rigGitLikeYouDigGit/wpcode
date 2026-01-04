"""
Parameterized actions - properly handles parameter substitution.

This module provides a cleaner way to define actions with parameters
that get substituted into preconditions and effects.
"""

from typing import Callable, Dict, Any
from wpplan.state import State
from wpplan.action import ActionTemplate, Precondition, Effect


class ParameterizedPrecondition:
	"""Precondition that references action parameters"""

	def __init__(self, var_path_fn: Callable[[Dict], str], predicate_fn: Callable[[Any, Dict, State], bool], description: str = ""):
		"""
		Args:
			var_path_fn: Function that takes params dict and returns variable path
			predicate_fn: Function that takes (value, params, state) and returns bool
			description: Human-readable description
		"""
		self.var_path_fn = var_path_fn
		self.predicate_fn = predicate_fn
		self.description = description

	def bind(self, params: Dict[str, Any]) -> Precondition:
		"""Bind parameters to create a concrete precondition"""
		var_path = self.var_path_fn(params)

		def bound_predicate(value):
			# Create a closure over params for the predicate
			return lambda state: self.predicate_fn(value, params, state)

		return Precondition(var_path, bound_predicate(None), self.description)


class ParameterizedEffect:
	"""Effect that references action parameters"""

	def __init__(self, var_path_fn: Callable[[Dict], str], effect_fn: Callable[[Any, Dict, State], Any], description: str = ""):
		"""
		Args:
			var_path_fn: Function that takes params dict and returns variable path
			effect_fn: Function that takes (current_value, params, state) and returns new value
			description: Human-readable description
		"""
		self.var_path_fn = var_path_fn
		self.effect_fn = effect_fn
		self.description = description

	def bind(self, params: Dict[str, Any]) -> Effect:
		"""Bind parameters to create a concrete effect"""
		var_path = self.var_path_fn(params)

		def bound_effect(value, state):
			return self.effect_fn(value, params, state)

		return Effect(var_path, bound_effect, self.description)


def make_simple_precondition(var_path: str, predicate: Callable[[Any], bool], description: str = "") -> ParameterizedPrecondition:
	"""Helper to create a simple precondition that doesn't reference parameters"""
	return ParameterizedPrecondition(
		lambda params: var_path,
		lambda val, params, state: predicate(val) if callable(predicate) else (val == predicate),
		description
	)


def make_simple_effect(var_path: str, new_value_fn: Callable[[Any, State], Any], description: str = "") -> ParameterizedEffect:
	"""Helper to create a simple effect that doesn't reference parameters"""
	return ParameterizedEffect(
		lambda params: var_path,
		lambda val, params, state: new_value_fn(val, state),
		description
	)


# Example builders for common patterns

def same_location_precondition(agent_id: str ="Agent", obj_param: str = "object") -> ParameterizedPrecondition:
	"""Precondition: agent and object must be at same location"""
	def check(val, params, state):
		agent = state.get_object(agent_id)
		obj_id = params.get(obj_param)
		if not obj_id:
			return False
		obj = state.get_object(obj_id)
		if not agent or not obj:
			return False
		agent_loc = agent.get_attribute("location")
		obj_loc = obj.get_attribute("location")
		if not agent_loc or not obj_loc:
			return False
		return agent_loc.value == obj_loc.value
	return ParameterizedPrecondition(
		lambda params: "_special",  # Special marker
		check,
		f"{agent_id} and {obj_param} at same location"
	)


def set_agent_carrying(agent_id: str = "Agent", obj_param: str = "object") -> ParameterizedEffect:
	"""Effect: set agent carrying to object"""
	def effect(val, params, state):
		return params.get(obj_param)
	return ParameterizedEffect(
		lambda params: f"{agent_id}.carrying",
		effect,
		f"Set {agent_id} carrying {obj_param}"
	)


def clear_agent_carrying(agent_id: str = "Agent") -> ParameterizedEffect:
	"""Effect: clear agent carrying"""
	return ParameterizedEffect(
		lambda params: f"{agent_id}.carrying",
		lambda val, params, state: None,
		f"Clear {agent_id} carrying"
	)


def add_to_container(container_param: str = "container", obj_param: str = "object") -> ParameterizedEffect:
	"""Effect: add object to container contents"""
	def effect(val, params, state):
		container_id = params.get(container_param)
		obj_id = params.get(obj_param)
		if not container_id or not obj_id:
			return val

		container = state.get_object(container_id)
		if container:
			contents_var = container.get_attribute("contents")
			if contents_var and isinstance(contents_var.value, set):
				contents_var.value.add(obj_id)

		return val

	return ParameterizedEffect(
		lambda params: f"{params.get(container_param, '')}.contents",
		effect,
		f"Add {obj_param} to {container_param}"
	)


def remove_from_container(container_param: str = "container", obj_param: str = "object") -> ParameterizedEffect:
	"""Effect: remove object from container contents"""
	def effect(val, params, state):
		container_id = params.get(container_param)
		obj_id = params.get(obj_param)
		if not container_id or not obj_id:
			return val

		container = state.get_object(container_id)
		if container:
			contents_var = container.get_attribute("contents")
			if contents_var and isinstance(contents_var.value, set):
				contents_var.value.discard(obj_id)

		return val

	return ParameterizedEffect(
		lambda params: f"{params.get(container_param, '')}.contents",
		effect,
		f"Remove {obj_param} from {container_param}"
	)
