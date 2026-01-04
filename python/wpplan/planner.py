"""
Planner: stochastic search with leverage/influence heuristics
"""

import random
import math
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from wpplan.state import State
from wpplan.action import Action, ActionTemplate
from wpplan.goal import Goal, GoalSet
from wpplan.influence import InfluenceGraph
from wpplan.knowledge import KnowledgeBase
from wpplan.reversible import ReversibleState
from wpplan.action_cache import ActionCache, SmartActionGenerator


@dataclass
class SearchNode:
	"""A node in the search tree"""
	state: State
	action: Optional[Action]  # Action that led to this state (None for root)
	parent: Optional['SearchNode']
	depth: int
	cost: float  # Cumulative cost from root

	# Heuristic values
	goal_progress: float = 0.0  # How much closer to goals
	leverage: float = 0.0  # Influence on future actions
	info_gain: float = 0.0  # Expected information discovery
	discovery_potential: float = 0.0  # Learned discovery patterns

	# For exploration tracking
	children: List['SearchNode'] = field(default_factory=list)
	visits: int = 0
	total_value: float = 0.0

	@property
	def value(self) -> float:
		"""Estimated value of this node"""
		if self.cost == 0:
			return 0.0
		# Value per unit cost
		return (self.goal_progress + self.leverage + self.info_gain + self.discovery_potential) / self.cost

	def get_path(self) -> List[Action]:
		"""Get the path of actions from root to this node"""
		path = []
		node = self
		while node.parent is not None:
			if node.action:
				path.append(node.action)
			node = node.parent
		return list(reversed(path))


class Planner:
	"""
	Stochastic planner with leverage-based search heuristics.

	Uses weighted sampling instead of deterministic best-first search
	to avoid local minima.
	"""

	def __init__(self,
				 knowledge: KnowledgeBase,
				 influence_graph: InfluenceGraph,
				 alpha: float = 1.0,  # Goal progress weight
				 beta: float = 0.5,   # Leverage weight
				 gamma: float = 0.3,  # Info gain weight
				 delta: float = 0.2,  # Discovery potential weight
				 temperature: float = 1.0):  # Stochastic sampling temperature

		self.knowledge = knowledge
		self.influence_graph = influence_graph

		# Heuristic weights
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.delta = delta

		# Temperature for stochastic selection (higher = more random)
		self.temperature = temperature

		# Action cache for efficient instantiation
		self.action_cache = ActionCache()
		self.action_generator = SmartActionGenerator(self.action_cache)
		self._cache_built = False

	def plan(self,
			 initial_state: State,
			 goals: GoalSet,
			 max_depth: int = 10,
			 max_samples: int = 1000,
			 beam_width: int = 5) -> Optional[List[Action]]:
		"""
		Plan a sequence of actions to satisfy goals.

		Uses beam search with stochastic selection to explore promising paths.

		Args:
			initial_state: Starting state
			goals: Goals to achieve
			max_depth: Maximum plan length
			max_samples: Maximum number of nodes to explore
			beam_width: Number of promising paths to maintain

		Returns:
			List of actions to achieve goals, or None if no plan found
		"""
		# Build action cache if not already built
		if not self._cache_built:
			templates = self.knowledge.get_all_templates()
			self.action_cache.build_cache(initial_state, templates)
			self._cache_built = True

		# Initialize search
		root = SearchNode(
			state=initial_state,
			action=None,
			parent=None,
			depth=0,
			cost=0.0
		)

		# Beam: maintain top beam_width promising nodes at each level
		current_beam = [root]
		samples_used = 0
		best_solution = None
		best_satisfaction = 0.0

		for depth in range(max_depth):
			if samples_used >= max_samples:
				break

			next_beam = []

			for node in current_beam:
				# Check if this node satisfies goals
				satisfaction = goals.total_satisfaction(node.state)
				if satisfaction > best_satisfaction:
					best_satisfaction = satisfaction
					best_solution = node

				if satisfaction >= 0.99:
					# Found a solution!
					return node.get_path()

				# Expand this node
				expansions = self._expand_node(node, goals, initial_state)
				samples_used += len(expansions)

				next_beam.extend(expansions)

			if not next_beam:
				break

			# Stochastically select beam_width nodes for next iteration
			current_beam = self._stochastic_select(next_beam, beam_width)

		# Return best solution found
		if best_solution:
			return best_solution.get_path()
		return None

	def _expand_node(self, node: SearchNode, goals: GoalSet, initial_state: State) -> List[SearchNode]:
		"""
		Expand a node by generating applicable actions.

		Uses leverage/influence to filter and prioritize actions.
		Uses reversible state for efficient exploration.
		"""
		# Get applicable actions
		applicable_actions = self._get_applicable_actions(node.state)

		# Score each action using reversible simulation
		scored_actions = []
		for action in applicable_actions:
			# If we have a reversible state, use it efficiently
			if isinstance(node.state, ReversibleState):
				node.state.begin_transaction()
				action.apply(node.state)
				score = self._score_action(action, node.state, goals, initial_state)
				node.state.rollback()
			else:
				# Fallback to regular scoring
				score = self._score_action(action, node.state, goals, initial_state)

			scored_actions.append((action, score))

		# Sort by score and take top candidates
		scored_actions.sort(key=lambda x: sum(x[1].values()), reverse=True)

		# Stochastically sample actions weighted by score
		num_to_expand = min(10, len(scored_actions))  # Expand top 10
		selected_actions = self._stochastic_select_actions(scored_actions, num_to_expand)

		# Create child nodes
		children = []
		for action, score in selected_actions:
			# Only copy state when we're keeping a node
			new_state = action.apply(node.state)
			child = SearchNode(
				state=new_state,
				action=action,
				parent=node,
				depth=node.depth + 1,
				cost=node.cost + action.cost(node.state),
				goal_progress=score['goal_progress'],
				leverage=score['leverage'],
				info_gain=score['info_gain'],
				discovery_potential=score['discovery_potential']
			)
			children.append(child)
			node.children.append(child)

		return children

	def _get_applicable_actions(self, state: State) -> List[Action]:
		"""
		Get all applicable actions in current state.

		Uses cached actions - just checks preconditions.
		"""
		# Use smart action generator to prioritize relevant actions
		# This returns applicable actions sorted by relevance
		return self.action_generator.get_prioritized_actions(
			state,
			max_actions=100  # Limit to most promising actions
		)

	def _score_action(self, action: Action, state: State, goals: GoalSet, initial_state: State) -> Dict[str, float]:
		"""
		Score an action based on multiple heuristics.

		Returns dictionary with individual scores.
		"""
		# 1. Goal progress: how much does this advance goals?
		goal_progress = 0.0
		for goal in goals.get_active_goals(state):
			goal_progress += goal.progress(action, state)

		# 2. Leverage: how much influence does this action have?
		# Use relation graph from state if available
		leverage = 0.0
		if hasattr(state, 'get_influence_count'):
			# Get all variables this action affects
			for effect in action.effects:
				var_path = effect.variable_path
				# Get influence count for this variable
				influence = state.get_influence_count(var_path)
				leverage += influence

		# 3. Info gain: how much might we learn?
		# Check if action affects uncertain variables
		info_gain = 0.0
		for effect in action.effects:
			var_path = effect.variable_path
			# Check if this variable has uncertainty
			# Simplified: just check if it's an exploration target
			parts = var_path.split('.')
			if len(parts) > 1:
				obj_id = parts[0]
				if obj_id in self.knowledge.get_exploration_targets():
					info_gain += 1.0

		# 4. Discovery potential: learned patterns
		discovery_potential = self.knowledge.estimate_discovery_potential(action, state)

		return {
			'goal_progress': goal_progress * self.alpha,
			'leverage': leverage * self.beta,
			'info_gain': info_gain * self.gamma,
			'discovery_potential': discovery_potential * self.delta
		}

	def _stochastic_select(self, nodes: List[SearchNode], k: int) -> List[SearchNode]:
		"""
		Stochastically select k nodes weighted by their value.

		Uses softmax with temperature for selection probabilities.
		"""
		if len(nodes) <= k:
			return nodes

		# Get values
		values = [node.value for node in nodes]

		# Apply softmax with temperature
		weights = self._softmax(values, self.temperature)

		# Sample k nodes without replacement
		selected_indices = []
		remaining_indices = list(range(len(nodes)))
		remaining_weights = list(weights)

		for _ in range(k):
			if not remaining_indices:
				break

			# Normalize remaining weights
			total = sum(remaining_weights)
			if total == 0:
				# All weights are zero, sample uniformly
				idx = random.choice(range(len(remaining_indices)))
			else:
				probs = [w / total for w in remaining_weights]
				idx = random.choices(range(len(remaining_indices)), weights=probs)[0]

			selected_indices.append(remaining_indices[idx])
			remaining_indices.pop(idx)
			remaining_weights.pop(idx)

		return [nodes[i] for i in selected_indices]

	def _stochastic_select_actions(self, scored_actions: List[Tuple[Action, Dict]], k: int) -> List[Tuple[Action, Dict]]:
		"""
		Stochastically select k actions weighted by their total score.
		"""
		if len(scored_actions) <= k:
			return scored_actions

		# Calculate total scores
		total_scores = [sum(score.values()) for action, score in scored_actions]

		# Apply softmax
		weights = self._softmax(total_scores, self.temperature)

		# Sample k actions
		selected_indices = []
		remaining_indices = list(range(len(scored_actions)))
		remaining_weights = list(weights)

		for _ in range(k):
			if not remaining_indices:
				break

			total = sum(remaining_weights)
			if total == 0:
				idx = random.choice(range(len(remaining_indices)))
			else:
				probs = [w / total for w in remaining_weights]
				idx = random.choices(range(len(remaining_indices)), weights=probs)[0]

			selected_indices.append(remaining_indices[idx])
			remaining_indices.pop(idx)
			remaining_weights.pop(idx)

		return [scored_actions[i] for i in selected_indices]

	def _softmax(self, values: List[float], temperature: float) -> List[float]:
		"""
		Apply softmax with temperature to values.

		Higher temperature = more uniform distribution (more exploration).
		Lower temperature = more peaked distribution (more exploitation).
		"""
		if not values:
			return []

		# Normalize to avoid overflow
		max_val = max(values)
		exp_values = [math.exp((v - max_val) / temperature) for v in values]
		total = sum(exp_values)

		if total == 0:
			return [1.0 / len(values)] * len(values)

		return [ev / total for ev in exp_values]

	def adjust_weights(self, success: bool, context: str):
		"""
		Adjust heuristic weights based on planning success.

		This allows agents to specialize based on experience.
		"""
		# Placeholder for learning
		# In full implementation:
		# - Track which heuristics correlated with success
		# - Increase weights for successful heuristics
		# - Decrease weights for unsuccessful ones
		pass
