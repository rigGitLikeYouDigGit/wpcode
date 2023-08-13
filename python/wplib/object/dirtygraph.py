from __future__ import annotations
import typing as T

from collections import defaultdict
from weakref import WeakSet


import networkx as nx

from wplib.sentinel import Sentinel

"""

conceptual bases for building dependency system with simple dirty propagation
Make minimal assumptions about nodes - graph should handle all
node dependencies.

"""

# class DirtyState(TypeNamespace):
# 	class _Base(TypeNamespace.base()):
# 		"""base class for dirty state"""
# 		pass
#
# 	class Clean(_Base):
# 		"""node is clean"""
# 		pass
#
# 	class Dirty(_Base):
# 		"""node is dirty"""
# 		pass
#
# 	class Computing(_Base):
# 		"""node is computing"""
# 		pass

# complex state not necessary - even if a node starts
# computing and errors, it's still dirty

class DirtyNode:

	def __init__(self, name:str=""):
		"""node inits take no argument -
		should hold no direct reference to graph

		pass in a load of functions for events? requests?

		name is just a convenience for debugging
		"""
		self._dirtyNodeName = name
		self.dirtyState = True

		# cache value on computation
		self._cachedValue = Sentinel.FailToFind

	def __str__(self):
		return f"<DNode({self._dirtyNodeName})>"

	def __repr__(self):
		return self.__str__()

	# def setDirtyGraph(self, graph:DirtyGraph):
	# 	"""set graph for node -
	# 	still don't like node knowing about its graph"""
	# 	self.dirtyGraph = graph

	def getDirtyGraph(self)->DirtyGraph:
		"""FINE"""
		return self.dirtyGraph

	def _computeWhenDirty(self):
		""" OVERRIDE THIS
		compute node logic
		inheriting from this feels like too tight a coupling

		this is also required to set the node's cached value
		"""
		self._cachedValue = None

	def getDirtyNodeAntecedents(self)->tuple[DirtyNode]:
		"""OVERRIDE THIS
		return any dirty node objects that immediately
		precede this one

		used by graph to build edges
		"""
		return ()

	def getValue(self):
		"""client-facing function to get a node's semantic value -
		calls into graph if node is dirty to arrange evaluation path
		for ancestor nodes"""
		if not self.dirtyState:
			return self._cachedValue
		return self._cachedValue


	def _onSetDirty(self):
		"""called when node is set dirty"""
		pass

	def setDirty(self):
		"""atomic node operation to set dirty state,
		does not propagate"""

		if self.dirtyState:
			return
		self.dirtyState = True
		# remove cached value
		self._cachedValue = Sentinel.FailToFind
		self._onSetDirty()

	def flagNodeDirtyAndPropagate(self):
		"""flag node dirty, and propagate dirty to graph
		wordy"""
		self.getDirtyGraph().setNodeDirty(self)

	def _onSetClean(self):
		"""called when node is set clean -
		usually after computation or reset"""
		pass

	def setClean(self):
		"""set node clean state
		this doesn't propagate to graph anyway, so no use in having same
		split method as with dirty
		"""
		if not self.dirtyState:
			return
		self.dirtyState = False
		self._onSetClean()






class DirtyGraph(nx.DiGraph):
	"""track dirty status of nodes

	maybe a semi-singleton? a dirty graph can hold the entirety of a program
	"""

	# defaultInstance : DirtyGraph = None
	#
	# @classmethod
	# def getDefaultInstance(cls) -> DirtyGraph:
	# 	if not cls.defaultInstance:
	# 		cls.defaultInstance = cls()
	# 	return cls.defaultInstance

	def addNodesAndPrecedents(self, nodesToAdd:set[DirtyNode]):
		"""adds node, checks over all antecedents,
		adds them, adds edges, repeats until all antecedents
		are in graph"""
		self.add_nodes_from(nodesToAdd)
		for i in nodesToAdd:
			self.add_edges_from((n, i) for n in i.getDirtyNodeAntecedents())


	def setNodeDirty(self, node:DirtyNode):
		"""
		marks given node as dirty and propagates forwards
		iterate only future nodes only if they are not dirty"""
		nodeSet = {node}
		adj = self.adj # get single adjacency view
		while nodeSet:
			node = nodeSet.pop()
			if node.dirty:
				continue
			node.setDirty()
			nodeSet.update(adj[node])

	def earliestDirtyNodes(self, nodes:set[DirtyNode])->set[DirtyNode]:
		"""given pool of nodes, return earliest dirty nodes
		that have no dirty inputs.

		Not sure if it's more efficient to filter ancestor sets like this,
		or to just copy out a new graph containing only dirty nodes and
		edges between them
		"""
		ancestors = set()
		nodes = set(nodes)

		while nodes:
			ancestors.update(i for i in nx.ancestors(self, nodes.pop()) if i.dirtyState)
			nodes -= ancestors

		return {i for i in ancestors if not any(
			j.dirtyState for j in self.pred[i]
		)}







