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
	"""high-level mixin for user-facing nodes or tasks
	in a graph"""

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

	def _getDirtyNodeName(self)->str:
		"""get node name"""
		return self._dirtyNodeName

	def __str__(self):
		return f"<{type(self).__name__}({self._getDirtyNodeName()})>"

	def __repr__(self):
		return self.__str__()

	def dirtyComputeFn(self):
		""" OVERRIDE THIS or set with a lambda
		compute node logic
		inheriting from this feels like too tight a coupling
		return the value that will become this node's cached value
		"""

	def _computeDirtyNodeOuter(self):
		"""compute node logic, cache result
		maybe also set clean
		"""
		self._cachedValue = self.dirtyComputeFn()
		return self._cachedValue

	def getDirtyNodeAntecedents(self)->tuple[DirtyNode]:
		"""OVERRIDE THIS
		return any dirty node objects that immediately
		precede this one

		used by graph to build edges
		"""
		return ()

	# def getDirtyNodeValue(self):
	# 	"""client-facing function to get a node's semantic value -
	# 	calls into graph if node is dirty to arrange evaluation path
	# 	for ancestor nodes"""
	# 	if not self.dirtyState:
	# 		return self._cachedValue
	# 	return self._cachedValue


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
		#self._cachedValue = Sentinel.FailToFind
		self._onSetDirty()

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
	not going crazy with actual evaluation here,
	only determining the queue order to run nodes
	when changed -
	might start using something bigger like Dask
	if computation itself becomes intensive
	"""

	def addNodesAndPrecedents(self, nodesToAdd:set[DirtyNode]):
		"""adds node, checks over all antecedents,
		adds them, adds edges, repeats until all antecedents
		are in graph"""
		self.add_nodes_from(nodesToAdd)
		for i in nodesToAdd:
			self.add_edges_from((n, i) for n in i.getDirtyNodeAntecedents())


	def setNodeDirty(self, node:DirtyNode, propagate:bool=True):
		"""
		marks given node as dirty and propagates forwards
		iterate only future nodes only if they are not dirty"""
		if not propagate:
			node.setDirty()
			return
		nodeSet = {node}
		succ = self.succ # S U C C
		while nodeSet:
			node = nodeSet.pop()
			if node.dirtyState: # exit if node is already dirty
				continue
			node.setDirty()
			nodeSet.update(succ[node])

	def earliestDirtyNodes(self, beforeNodes:set[DirtyNode])->set[DirtyNode]:
		"""given pool of nodes, return earliest dirty nodes
		that have no dirty inputs.

		Not sure if it's more efficient to filter ancestor sets like this,
		or to just copy out a new graph containing only dirty nodes and
		edges between them
		"""
		ancestors = set()
		beforeNodes = set(beforeNodes)

		while beforeNodes:
			ancestors.update(i for i in nx.ancestors(self, beforeNodes.pop()) if i.dirtyState)
			beforeNodes -= ancestors

		return {i for i in ancestors if not
			any(j.dirtyState for j in self.pred[i])
		        }

	def getDirtyNodesToEval(self, fromNode:DirtyNode)->T.Iterable[DirtyNode]:
		"""return a flat sequence of nodes to evaluate in dependency order
		TODO: separate function to return nested list somehow,
			to preserve parallelism in nodes
		if desired, pass out this task list to a separate scheduler
		"""
		if not fromNode.dirtyState:
			return []
		result = [fromNode]
		idSet = {id(fromNode)}

		pred = self.pred
		nextNodes = list(pred[fromNode].keys())
		while nextNodes:
			nextNode = nextNodes.pop(-1)
			if not nextNode.dirtyState:
				continue
			if id(nextNode) in idSet: # no duplicates
				continue
			result.append(nextNode)
			nextNodes.extend(pred[nextNode].keys())
			idSet.add(id(nextNode))
		return reversed(result)

	def setNodeClean(self, node:DirtyNode):
		node.setClean()

	def execTasks(self, taskSeq:T.Iterable[DirtyNode]):
		"""add more complex stuff here if needed
		TODO: errors? set separate error state on node,
			maybe allow node to log events during its own evaluation
		"""
		for i in taskSeq:
			i._computeDirtyNodeOuter()

	def getNodeValue(self, node:DirtyNode):
		if not node.dirtyState:
			return node._cachedValue
		tasks = self.getDirtyNodesToEval(node)
		self.execTasks(tasks)
		return node._cachedValue





if __name__ == '__main__':

	import pprint

	# execution
	"""
	consider a node with ref {"vals" : "n:a or n:b or n:c"}
	
	"""

	nodeA = DirtyNode("a")
	nodeB = DirtyNode("b")
	nodeC = DirtyNode("c")

	dNodes = {nodeA, nodeB, nodeC}

	nodeC.getDirtyNodeAntecedents = lambda: [nodeA, nodeB]
	nodeB.getDirtyNodeAntecedents = lambda: [nodeA]



	dGraph = DirtyGraph()

	dGraph.addNodesAndPrecedents(dNodes)
	print(nodeC.getDirtyNodeAntecedents())
	print(nodeB.getDirtyNodeAntecedents())
	pprint.pp(dGraph.nodes())

	# for i in dGraph.nodes:
	# 	print(i, i.dirtyState)

	print(dGraph.edges)

	print(dGraph.earliestDirtyNodes(dNodes))
	nodeA.dirtyState = False
	print(dGraph.earliestDirtyNodes(dNodes))

#





