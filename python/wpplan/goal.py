"""
Goals: target conditions with satisfaction predicates
"""

from typing import Callable, Any, Optional
from dataclasses import dataclass
from wpplan.state import State, Variable


@dataclass
class Goal:
	"""A goal with satisfaction conditions and priority"""

	description: str
	satisfaction_fn: Callable[[State], float]  # Returns 0.0 (not satisfied) to 1.0 (fully satisfied)
	priority: float = 1.0  # Higher = more important
	deadline: Optional[float] = None  # Time by which goal should be achieved

	# For tracking
	created_at: float = 0.0
	parent_goal: Optional['Goal'] = None  # If this is a subgoal

	def is_satisfied(self, state: State, threshold: float = 0.99) -> bool:
		"""Check if goal is satisfied (above threshold)"""
		return self.satisfaction_fn(state) >= threshold

	def satisfaction(self, state: State) -> float:
		"""Get degree of satisfaction (0.0 to 1.0)"""
		return self.satisfaction_fn(state)

	def progress(self, action, state: State) -> float:
		"""
		Estimate how much progress an action makes toward this goal.
		Returns change in satisfaction.
		"""
		current_satisfaction = self.satisfaction(state)
		new_state = action.apply(state)
		new_satisfaction = self.satisfaction(new_state)
		return new_satisfaction - current_satisfaction

	def urgency(self, current_time: float) -> float:
		"""
		Calculate urgency of this goal based on deadline and priority.
		Returns higher values for more urgent goals.
		"""
		base_urgency = self.priority

		if self.deadline is not None:
			time_remaining = self.deadline - current_time
			if time_remaining <= 0:
				return float('inf')  # Overdue!
			# Urgency increases as deadline approaches
			deadline_factor = 1.0 / time_remaining
			base_urgency *= (1.0 + deadline_factor)

		return base_urgency


class GoalSet:
	"""A collection of goals, possibly with dependencies"""

	def __init__(self):
		self.goals: list[Goal] = []

	def add_goal(self, goal: Goal):
		"""Add a goal to the set"""
		self.goals.append(goal)

	def remove_goal(self, goal: Goal):
		"""Remove a goal from the set"""
		if goal in self.goals:
			self.goals.remove(goal)

	def get_active_goals(self, state: State, threshold: float = 0.99) -> list[Goal]:
		"""Get goals that are not yet satisfied"""
		return [g for g in self.goals if not g.is_satisfied(state, threshold)]

	def get_most_urgent(self, state: State, current_time: float) -> Optional[Goal]:
		"""Get the most urgent unsatisfied goal"""
		active = self.get_active_goals(state)
		if not active:
			return None
		return max(active, key=lambda g: g.urgency(current_time))

	def total_satisfaction(self, state: State) -> float:
		"""Get weighted average satisfaction across all goals"""
		if not self.goals:
			return 1.0
		total_weight = sum(g.priority for g in self.goals)
		if total_weight == 0:
			return 1.0
		weighted_sum = sum(g.satisfaction(state) * g.priority for g in self.goals)
		return weighted_sum / total_weight
