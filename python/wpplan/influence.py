"""
Influence graph: tracks dependencies and leverage between variables
"""

from typing import Dict, Set, List
from dataclasses import dataclass, field
from wpplan.state import State, Variable
from wpplan.goal import Goal


@dataclass
class InfluenceNode:
	"""A node in the influence graph representing a state variable"""
	var_name: str

	# Direct goal relevance: how much do goals care about this variable?
	direct_goal_relevance: float = 0.0

	# Transitive relevance: sum of relevance of variables that depend on this one
	transitive_relevance: float = 0.0

	# Dependencies
	depends_on: Set[str] = field(default_factory=set)  # Variables this depends on
	influences: Set[str] = field(default_factory=set)  # Variables that depend on this

	@property
	def total_relevance(self) -> float:
		"""Total relevance = direct + transitive"""
		return self.direct_goal_relevance + self.transitive_relevance


class InfluenceGraph:
	"""
	Tracks how variables influence each other and their relevance to goals.
	Used to calculate leverage of actions.
	"""

	def __init__(self):
		self.nodes: Dict[str, InfluenceNode] = {}

	def add_variable(self, var_name: str) -> InfluenceNode:
		"""Add a variable to the influence graph"""
		if var_name not in self.nodes:
			self.nodes[var_name] = InfluenceNode(var_name)
		return self.nodes[var_name]

	def add_dependency(self, dependent: str, depends_on: str):
		"""
		Add a dependency: 'dependent' depends on 'depends_on'.
		E.g., Letter1.location depends on Bag.location when Letter1 is in Bag.
		"""
		dep_node = self.add_variable(dependent)
		base_node = self.add_variable(depends_on)

		dep_node.depends_on.add(depends_on)
		base_node.influences.add(dependent)

	def remove_dependency(self, dependent: str, depends_on: str):
		"""Remove a dependency"""
		if dependent in self.nodes and depends_on in self.nodes:
			self.nodes[dependent].depends_on.discard(depends_on)
			self.nodes[depends_on].influences.discard(dependent)

	def set_goal_relevance(self, var_name: str, relevance: float):
		"""Set direct goal relevance for a variable"""
		node = self.add_variable(var_name)
		node.direct_goal_relevance = relevance

	def update_from_goals(self, goals: List[Goal], state: State):
		"""
		Update goal relevance based on current goals.
		This is called when goals change or state changes significantly.
		"""
		# Reset all relevances
		for node in self.nodes.values():
			node.direct_goal_relevance = 0.0
			node.transitive_relevance = 0.0

		# Set direct relevances based on goals
		# TODO: Need a way for goals to declare which variables they care about
		# For now, this is a placeholder

		# Propagate transitive relevance
		self._propagate_relevance()

	def _propagate_relevance(self):
		"""
		Propagate relevance through the dependency graph.
		Variables that influence goal-relevant variables become transitively relevant.
		"""
		# Topological sort-like propagation
		# Start from variables with direct goal relevance
		# Propagate backwards through depends_on edges

		visited = set()

		def propagate(var_name: str):
			if var_name in visited:
				return
			visited.add(var_name)

			node = self.nodes.get(var_name)
			if node is None:
				return

			# Propagate to all variables this one depends on
			for dep_var in node.depends_on:
				dep_node = self.nodes.get(dep_var)
				if dep_node:
					# Add this node's total relevance to the dependency's transitive relevance
					dep_node.transitive_relevance += node.total_relevance
					propagate(dep_var)

		# Start from all nodes with direct goal relevance
		for node in self.nodes.values():
			if node.direct_goal_relevance > 0:
				propagate(node.var_name)

	def get_leverage(self, var_name: str) -> float:
		"""
		Get the leverage of a variable: how many goal-relevant variables
		does it transitively affect?
		"""
		node = self.nodes.get(var_name)
		if node is None:
			return 0.0
		return node.total_relevance

	def get_action_leverage(self, action, state: State) -> float:
		"""
		Calculate leverage of an action based on what variables it affects.
		High leverage = affects many goal-relevant variables.
		"""
		# Get variables affected by action's effects
		affected_vars = set()
		for effect in action.effects:
			affected_vars.add(effect.variable_path)

		# Sum up leverage of all affected variables
		total_leverage = 0.0
		for var_name in affected_vars:
			# Direct leverage from this variable
			total_leverage += self.get_leverage(var_name)

			# Add leverage from variables influenced by this one
			node = self.nodes.get(var_name)
			if node:
				for influenced_var in node.influences:
					total_leverage += self.get_leverage(influenced_var)

		return total_leverage

	def get_empowerment(self, var_name: str, state: State) -> int:
		"""
		Calculate empowerment: how many goal-relevant states become reachable
		by changing this variable?

		This is a simplified version - full implementation would need to
		explore reachable state space.
		"""
		# For now, use influence count as a proxy
		node = self.nodes.get(var_name)
		if node is None:
			return 0
		return len(node.influences)
