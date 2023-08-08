from __future__ import annotations
import typing as T

import networkx
import networkx as nx

from wplib.object import UidElement

from chimaera.core.node import ChimaeraNode

class ChimaeraGraph(
	ChimaeraNode
):
	"""Graphs are nodes, some nodes are graphs.

	In this model the parent/child relationship is rock solid,
	and not defined by node edges.

	Each subgraph has to manage its own stored data, which could be
	more difficult - however it means that any graph only has
	to be concerned with its direct children,
	it makes it way easier to merge and reference parts of the graph, etc.

	ON BALANCE, I think the tree hierarchy model is better for now. If needed maybe we can add some way to emulate subgraphs over a flat hierarchy.

	Compound nodes hold datablocks for all contained nodes?

	Holding off on any digraph stuff - maybe evaluation is handled by
	a separate system altogether.
	If we were to have each ChimaeraGraph be a separate digraph, there would
	be absolutely no way to interact between hierarchies -
	I think we probably want that.



	"""

	def __init__(self):
		ChimaeraNode.__init__(self)

		"""consider - this .data reference is never REPLACED - 
		this data dict may be embedded live into another graph,
		but this object won't know it"""
		self.data = self.defaultData()

	@classmethod
	def defaultData(cls)->dict:
		baseData = ChimaeraNode.defaultData()
		baseData["nodes"] = {}
		baseData["attrMap"]["name"].setStructure("graph")
		return baseData

	def addNode(self, node:ChimaeraNode, nodeData:dict):
		"""add a node to the graph, with data"""
		self.data["nodes"][node.getElementId()] = nodeData
		node._parent = self

	def createNode(self, name:str, nodeType=ChimaeraNode)->ChimaeraNode:
		"""create a node of type nodeType, add it to the graph, and return it"""
		newNode = nodeType()
		newData = newNode.defaultData()
		self.addNode(newNode, newData)
		newNode.setName(name)
		return newNode

	def dataBlock(self) ->dict:
		"""if graph is top level, return its own data"""
		if self._parent is None:
			return self.data
		return super(ChimaeraGraph, self).dataBlock()

	def nodeDataBlock(self, node:ChimaeraNode)->dict:
		return self.data["nodes"][node.getElementId()]

	def uidDataBlock(self, uid:str)->dict:
		return self.data["nodes"][uid]


