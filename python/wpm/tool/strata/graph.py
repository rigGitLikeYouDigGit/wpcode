from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import networkx as nx
from networkx import DiGraph # can two points only be connected by 1 edge? for now yes

from wplib.object import UidElement, DirtyGraph, DirtyNode
from wplib.serial import Serialisable
from wpm.tool.leyline.lib import DictModelled

from wpm.tool.leyline.point import LLElement
from wpm.tool.leyline.point import LLPoint
from wpm.tool.leyline.edge import LLEdge


class LLGraph(DictModelled):
	"""we don't do fractal stuff like in chimaera so
	a separate graph class makes sense"""

	def __init__(self,
				name:str,
	            elements:T.Sequence[LLElement]=(),
	             # edges=None,
	             # faces=None,
	             ):
		elements = {i["name"] : i for i in elements}

		super().__init__(name=name,
		                 elements=elements
		                 )
		self.elements = elements
		# use separate cache lists if needed, but might not be useful
		# self.points = {}
		# self.edges = {}
		# self.faces = {}

		# first time setup of dirtygraph
		self.dirtyGraph = self.buildDirtyGraph()


	def getEl(self, e:(LLElement, str))->LLElement:
		"""return a rich element object from its id"""
		if isinstance(e, str):
			return self["elements"][e]
		# it's already a rich object
		return e

	def getId(self, e:(LLElement, str)):
		return self.getEl(e)["name"]

	def addElements(self, es:T.Sequence[LLElement]):
		for i in es:
			i.graph = self
		self.elements.update({i.name : i for i in es})

		# TODO: not so wasteful here
		self.dirtyGraph = self.buildDirtyGraph()

	def buildDirtyGraph(self)->DirtyGraph:
		"""create a new dgraph for evaluation -
		only run once at startup, edit topology whenever changed thereafter
		"""
		g = DirtyGraph()
		#log("els")
		#pprint.pp(self.elements)
		g.addNodesAndPrecedents(self.elements.values())
		# g.add_nodes_from(list(self.elements.values()))
		# # add all nodes
		#
		# toCheck = set(self.elements.values())
		# while toCheck:
		# 	toAdd = toCheck.pop()
		# 	# add edge between all things directly driving this node, and this node
		# 	drivers = toAdd["parents"] or {}
		# 	for k in drivers.keys():
		# 		g.add_edge(
		# 			self.getEl(k),
		# 			toAdd
		# 		)
		return g










