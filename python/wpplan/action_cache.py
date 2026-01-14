"""
Action instantiation and caching.

Actions are instantiated once and cached. For each state, we only check
which cached actions are applicable (preconditions satisfied).
"""

from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from wpplan.state import State, ObjectState
from wpplan.action import Action, ActionTemplate
import itertools


@dataclass
class ActionKey:
	"""
	Unique identifier for an instantiated action.
	Used as cache key.
	"""
	template_name: str
	params: Tuple[str, ...]  # Sorted tuple of parameter values

	def __hash__(self):
		return hash((self.template_name, self.params))

	def __eq__(self, other):
		return (self.template_name == other.template_name and
				self.params == other.params)

	def __repr__(self):
		return f"{self.template_name}({', '.join(self.params)})"


class ActionCache:
	"""
	Cache of instantiated actions.

	Actions are instantiated once based on available objects.
	For each state, we filter to applicable actions by checking preconditions.
	"""

	def __init__(self):
		self.cache: Dict[ActionKey, Action] = {}
		self.template_actions: Dict[str, List[ActionKey]] = {}  # template_name -> action keys
		self._dirty = True

	def build_cache(self, state: State, templates: List[ActionTemplate]):
		"""
		Build the action cache based on objects in state.

		This should be called once at the start of planning, or when
		new objects are discovered.
		"""
		self.cache.clear()
		self.template_actions.clear()

		for template in templates:
			# Get all valid instantiations for this template
			actions = self._instantiate_template(template, state)

			# Cache them
			template_actions = []
			for action in actions:
				key = self._make_key(action)
				self.cache[key] = action
				template_actions.append(key)

			self.template_actions[template.name] = template_actions

		self._dirty = False
		print(f"Action cache built: {len(self.cache)} total actions")

	def get_applicable_actions(self, state: State) -> List[Action]:
		"""
		Get all actions that are applicable in the given state.

		This is fast: just checks preconditions of cached actions.
		"""
		if self._dirty:
			raise RuntimeError("Action cache is dirty - rebuild it first")

		applicable = []
		for action in self.cache.values():
			if action.is_applicable(state):
				applicable.append(action)

		return applicable

	def get_actions_for_template(self, template_name: str) -> List[Action]:
		"""Get all instantiated actions for a specific template"""
		keys = self.template_actions.get(template_name, [])
		return [self.cache[key] for key in keys]

	def invalidate(self):
		"""Mark cache as dirty (needs rebuild)"""
		self._dirty = True

	def _make_key(self, action: Action) -> ActionKey:
		"""Create a cache key for an action"""
		# Sort parameters for consistent key
		param_values = tuple(sorted(str(v) for v in action.params.values()))
		return ActionKey(action.template.name, param_values)

	def _instantiate_template(self, template: ActionTemplate, state: State) -> List[Action]:
		"""
		Instantiate a template with all valid object combinations.

		This is where the expensive enumeration happens.

		TODO: cache by (actionTemplate, params)
		"""
		if not template.param_names:
			# No parameters - just instantiate once
			action = template.instantiate(state, {})
			return [action] if action else []

		# Get candidate objects for each parameter
		param_candidates = {}
		for param_name in template.param_names:
			candidates = self._get_candidates_for_param(
				param_name, template, state
			)
			param_candidates[param_name] = candidates

		# Generate all combinations
		param_names = template.param_names
		candidate_lists = [param_candidates[name] for name in param_names]

		# Limit combinations to avoid explosion
		max_combinations = 1000
		actions = []

		for i, combination in enumerate(itertools.product(*candidate_lists)):
			if i >= max_combinations:
				print(f"Warning: {template.name} hit combination limit ({max_combinations})")
				break

			params = dict(zip(param_names, combination))

			# Try to instantiate
			action = template.instantiate(state, params)
			if action:
				actions.append(action)

		return actions

	def _get_candidates_for_param(
		self,
		param_name: str,
		template: ActionTemplate,
		state: State
	) -> List[str]:
		"""
		Get candidate object IDs for a parameter.

		Filters by property requirements to reduce search space.
		"""
		# Find property requirements for this parameter
		requirements = None
		for req in template.property_requirements:
			if req.param_name == param_name:
				requirements = req
				break

		if requirements is None:
			# No requirements - all objects are candidates
			return list(state.objects.keys())

		# Filter by required properties
		candidates = []
		for obj_id, obj in state.objects.items():
			# Check required properties
			has_required = all(
				obj.has_property(prop)
				for prop in requirements.required_traits
			)

			# Check forbidden properties
			has_forbidden = any(
				obj.has_property(prop)
				for prop in requirements.forbidden_traits
			)

			if has_required and not has_forbidden:
				candidates.append(obj_id)

		return candidates


class SmartActionGenerator:
	"""
	Intelligent action generation that goes beyond simple caching.

	Uses heuristics to prioritize which actions to generate/consider:
	- Goal-relevant objects first
	- High-leverage objects first
	- Recently modified objects first
	"""

	def __init__(self, cache: ActionCache):
		self.cache = cache
		self.goal_relevant_objects: Set[str] = set()
		self.high_leverage_objects: Set[str] = set()
		self.recently_modified: Set[str] = set()

	def set_goal_context(self, goal_relevant: Set[str]):
		"""Update which objects are relevant to current goals"""
		self.goal_relevant_objects = goal_relevant

	def set_leverage_context(self, high_leverage: Set[str]):
		"""Update which objects have high leverage"""
		self.high_leverage_objects = high_leverage

	def mark_modified(self, obj_id: str):
		"""Mark an object as recently modified"""
		self.recently_modified.add(obj_id)

	def get_prioritized_actions(self, state: State, max_actions: Optional[int] = None) -> List[Action]:
		"""
		Get actions prioritized by relevance.

		Returns most promising actions first, optionally limited.
		"""
		all_actions = self.cache.get_applicable_actions(state)

		# Score actions by relevance
		scored = []
		for action in all_actions:
			score = self._score_relevance(action)
			scored.append((action, score))

		# Sort by score
		scored.sort(key=lambda x: x[1], reverse=True)

		# Return top N or all
		if max_actions:
			scored = scored[:max_actions]

		return [action for action, score in scored]

	def _score_relevance(self, action: Action) -> float:
		"""
		Score how relevant an action is based on context.

		Higher score = more likely to be useful.
		"""
		score = 0.0

		# Check if action involves goal-relevant objects
		for param_value in action.params.values():
			if param_value in self.goal_relevant_objects:
				score += 10.0

			if param_value in self.high_leverage_objects:
				score += 5.0

			if param_value in self.recently_modified:
				score += 2.0

		return score


# Optimization: Parameter symmetry detection

def detect_symmetric_params(template: ActionTemplate) -> List[Tuple[str, str]]:
	"""
	Detect which parameters are symmetric (interchangeable).

	E.g., Swap(A, B) is same as Swap(B, A).

	Returns pairs of symmetric parameters.
	"""
	# Simple heuristic: parameters with same property requirements are symmetric
	symmetric_pairs = []

	for i, param1 in enumerate(template.param_names):
		for param2 in template.param_names[i+1:]:
			req1 = _get_requirements(template, param1)
			req2 = _get_requirements(template, param2)

			if req1 == req2:
				symmetric_pairs.append((param1, param2))

	return symmetric_pairs


def _get_requirements(template: ActionTemplate, param_name: str) -> Tuple[Set, Set]:
	"""Get (required_properties, forbidden_properties) for a parameter"""
	for req in template.property_requirements:
		if req.param_name == param_name:
			return (set(req.required_traits), set(req.forbidden_traits))
	return (set(), set())


def canonicalize_params(params: Dict[str, str], symmetric_pairs: List[Tuple[str, str]]) -> Dict[str, str]:
	"""
	Canonicalize parameters by sorting symmetric ones.

	This reduces duplicate cache entries for symmetric actions.
	"""
	result = dict(params)

	for param1, param2 in symmetric_pairs:
		val1 = params.get(param1)
		val2 = params.get(param2)

		if val1 and val2 and val1 > val2:
			# Swap to canonical order
			result[param1] = val2
			result[param2] = val1

	return result
