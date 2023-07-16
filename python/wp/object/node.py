from __future__ import annotations
import typing as T


"""making graph and dependency based systems less
daunting to implement.

No interaction with dirtyNode system yet
"""

import networkx as nx


class DepNode:
	"""single dependency node.
	Nodes may exist alone and define graph by their connections,
	with a new graph being generated on demand for convenience."""
	def __init__(self):
		self._inputs :dict[str, DepNode] = {}

	def addInput(self, name:str, node:DepNode):
		"""add input node"""
		self._inputs[name] = node

	def getInputs(self)->dict[str, DepNode]:
		"""get input nodes"""
		return dict(self._inputs)

	def setInputs(self, inputs:dict[str, DepNode]):
		"""set input nodes"""
		self._inputs = dict(inputs)

	def setInputNode(self, name:str, node:DepNode):
		"""set input node"""
		self._inputs[name] = node

	def removeInput(self, name:str):
		"""remove input node"""
		self._inputs.pop(name, None)

	def getInput(self, name:str)->DepNode:
		"""get input node"""
		return self._inputs.get(name, None)

	def getGraph(self)->nx.DiGraph:
		"""return nx digraph of this node and all preceding nodes"""
		graph = nx.DiGraph()
		graph.add_node(self)
		for name, node in self._inputs.items():
			graph.add_edges_from( node.getGraph().edges() )
		return graph

