"""
Knowledge base: agent's beliefs about the world (may be incomplete/incorrect)
"""

from typing import Dict, Set, Optional, List
from dataclasses import dataclass, field
from wpplan.state import State, ObjectState
from wpplan.action import ActionTemplate


@dataclass
class BeliefState:
	"""
	An agent's belief about the world state.
	May differ from true state (partial observability).
	"""
	# Known objects
	known_objects: Dict[str, ObjectState] = field(default_factory=dict)

	# Unknown/uncertain properties
	uncertain_properties: Dict[str, Set[str]] = field(default_factory=dict)  # obj_id -> uncertain properties

	# Known but unobserved containers (curiosity targets)
	unobserved_containers: Set[str] = field(default_factory=set)

	def add_object(self, obj: ObjectState):
		"""Add a known object to beliefs"""
		self.known_objects[obj.id] = obj

	def get_object(self, obj_id: str) -> Optional[ObjectState]:
		"""Get believed state of an object"""
		return self.known_objects.get(obj_id)

	def mark_uncertain(self, obj_id: str, property_name: str):
		"""Mark a property as uncertain"""
		if obj_id not in self.uncertain_properties:
			self.uncertain_properties[obj_id] = set()
		self.uncertain_properties[obj_id].add(property_name)

	def mark_unobserved_container(self, obj_id: str):
		"""Mark a container as unobserved (contents unknown)"""
		self.unobserved_containers.add(obj_id)

	def resolve_uncertainty(self, obj_id: str, property_name: str):
		"""Mark a property as now known (observed)"""
		if obj_id in self.uncertain_properties:
			self.uncertain_properties[obj_id].discard(property_name)

	def get_uncertainty_count(self) -> int:
		"""Get total number of uncertain properties"""
		return sum(len(props) for props in self.uncertain_properties.values())


class KnowledgeBase:
	"""
	Agent's knowledge about the world:
	- Current beliefs about state
	- Known action templates
	- Learned patterns and strategies
	"""

	def __init__(self):
		self.beliefs = BeliefState()
		self.action_templates: Dict[str, ActionTemplate] = {}

		# Historical data for learning
		self.action_history: List[tuple] = []  # (action, state_before, state_after, success)
		self.discovery_history: List[tuple] = []  # (action, num_objects_discovered, context)

	def add_action_template(self, template: ActionTemplate):
		"""Register an action template"""
		self.action_templates[template.name] = template

	def get_action_template(self, name: str) -> Optional[ActionTemplate]:
		"""Get an action template by name"""
		return self.action_templates.get(name)

	def get_all_templates(self) -> List[ActionTemplate]:
		"""Get all known action templates"""
		return list(self.action_templates.values())

	def observe(self, true_state: State, observable_objects: Set[str]):
		"""
		Update beliefs based on observation of true state.
		Only objects in observable_objects are observed.
		"""
		for obj_id in observable_objects:
			true_obj = true_state.get_object(obj_id)
			if true_obj:
				# Update belief with observed object
				self.beliefs.add_object(true_obj)

				# Resolve uncertainties
				if obj_id in self.beliefs.uncertain_properties:
					self.beliefs.uncertain_properties[obj_id].clear()

				# Check if it's a container we haven't looked inside
				if true_obj.has_property("Container"):
					# If we don't know contents, mark as unobserved
					contents_attr = true_obj.get_attribute("contents")
					if contents_attr and not self._has_observed_contents(obj_id):
						self.beliefs.mark_unobserved_container(obj_id)

	def _has_observed_contents(self, container_id: str) -> bool:
		"""Check if we've observed the contents of a container"""
		# Placeholder: in full implementation, track observation history
		return container_id not in self.beliefs.unobserved_containers

	def record_action_outcome(self, action, state_before: State, state_after: State, success: bool):
		"""Record the outcome of an action for learning"""
		self.action_history.append((action, state_before, state_after, success))

	def record_discovery(self, action, num_discovered: int, context: Dict):
		"""Record information gain from an action"""
		self.discovery_history.append((action, num_discovered, context))

	def estimate_discovery_potential(self, action, state: State) -> float:
		"""
		Estimate how much new information this action might reveal.
		Based on learned patterns from discovery_history.
		"""
		# Simple heuristic: average discovery rate for similar actions
		similar_actions = [
			(a, n) for a, n, ctx in self.discovery_history
			if a.template.name == action.template.name
		]

		if not similar_actions:
			# No history: default optimistic estimate
			return 1.0

		# Average discovery from similar actions
		avg_discovery = sum(n for a, n in similar_actions) / len(similar_actions)
		return avg_discovery

	def get_exploration_targets(self) -> List[str]:
		"""
		Get objects that should be explored (curiosity targets).
		E.g., unobserved containers.
		"""
		return list(self.beliefs.unobserved_containers)
