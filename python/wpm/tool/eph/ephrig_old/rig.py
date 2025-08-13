
import os, sys, copy


from networkx import DiGraph

from edRig.palette import *
from edRig.ephrig.node import EphNode
from edRig.ephrig.solver import EphSolver


# EphRig = None
class EphRig(object):
	"""
	controls building and evaluation of entire rig system
	"""

	serialScheme = {
		"nodes" : [],
		"edges" : {}
	}

	def __init__(self):

		#self.groundGraph = None # raw graph of individual nodes
		self.controllerGraph = None # graph of controllers drawn over ground

		# self.nodes = set() # ground truth set of nodes

		self.nodeControlMap = {} # map of { node : controllers affecting it }

		self.groundGraph = DiGraph()

		self.mainNode = "" # central control node in scene


	def build(self):
		pass

	@property
	def nodes(self)->T.Set[EphNode]:
		return set(self.groundGraph.nodes)

	def addNode(self, node:EphNode):
		self.groundGraph.add_node(node)
		#self.nodes.add(node)

	def addConnection(self, src:EphNode, dst:EphNode):
		print("rig add connection")
		self.groundGraph.add_edge(src, dst)
		pass

	def evaluate(self, seedNodes):
		"""
		triggers whenever a change affects seedNodes -
		evaluates all controllers currently active in rig,
		propagating updates from seedNodes
		"""

	def serialise(self):
		"""return serialised dict of an ephemeral rig"""
		base = copy.deepcopy(self.serialScheme)
		serialNodes = {i.name : i.serialise() for i in list(self.nodes)}
		nodeEdges = self.groundGraph.edges
		print("edges, ", nodeEdges)
		# flatten to strings
		flatEdges = []
		for tie in nodeEdges:
			flatEdges.append((tie[0].name, tie[1].name))
			# for src, dst in tie:

		# flatEdges = [ (src.name, dst.name) for src, dst in tie
		#               for tie in nodeEdges]
		base["nodes"] = serialNodes
		base["edges"] = flatEdges

		return base
