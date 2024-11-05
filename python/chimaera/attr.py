from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


import fnmatch

from collections import namedtuple

import wplib.sequence
from wplib import log, Sentinel, TypeNamespace, Pathable
from wplib.constant import MAP_TYPES, SEQ_TYPES, STR_TYPES, LITERAL_TYPES, IMMUTABLE_TYPES
from wplib.object.visitor import PARAMS_T, CHILD_LIST_T
from wplib.uid import getUid4
from wplib.inheritance import clsSuper

from wplib.object import VisitAdaptor, Visitable, ClassMagicMethodMixin, UidElement


from wptree import Tree

from wpdex import *


from chimaera.lib import tree as treelib

if T.TYPE_CHECKING:
	from .node import ChimaeraNode
	class NodeAttrRef(namedtuple):
		uid : str
		attr : str = "value"
		path : tuple[(str, int, slice), ...] = ()
else:
	NodeAttrRef = namedtuple("NodeAttrRef", "uid attr path",
	                         defaults=["", "@F", ()])


def resolveNodes(exp:str,
                 graph:ChimaeraNode,
                 fromNode:ChimaeraNode=None)->list[ChimaeraNode]:
	"""resolve a list of node expressions to nodes
	for now just uids

	this doesn't add the node Type object for a "T" expression
	"""
	results = []
	for i in exp:
		# if i == "T":
		# 	results.append(fromNode)
		# 	continue
		results.extend(graph.getNodes(i))
	return graph.getNodes(exp)

def expandIncomingTree(rawTree:Tree,
                       attrWrapper:NodeAttrWrapper,
                       parentNode:ChimaeraNode,
                       graph:ChimaeraNode)->Tree:
	"""expand the incoming tree for this attribute -

	filtering actual tree with path requires explicitly
	defining tree to use - ".p[branch, leaf]" etc
	if .p or .v is not defined, will slice nodes found by
	uid matching

	STILL need a general solution to evaluate normal python code
	within these that RESOLVES to a node expression, path etc

	path is always () for now, get back to it later

	use THIS tree to determine node dependencies

	left side always resolves to nodes, midpoint always to an attr name,
	right side always to a tree path within that attribute

	TODO: we list nodes here, only keep their uids, then list again in populateExpandedTree

	return tree[list[tuple[
		str node uid,
		str attribute,
		tuple[str] path
		]]]"""
	#assert graph
	for branch in rawTree.allBranches(includeSelf=True):
		rawValue : list[NodeAttrRef] = branch.value
		if rawValue is None:
			branch.value = []
			continue
		if not isinstance(rawValue, list):
			rawValue = [rawValue]
		resultTuples : list[NodeAttrRef] = []
		#log("raw value", rawValue)
		for i in rawValue: # individual string expressions

			# expand node lists to individual uids
			#print("EXPAND", i, i == "T",  i[0] == "T" if isinstance(i, tuple) else False)
			if i == "T":
				resultTuples.append( NodeAttrRef("T", attrWrapper.name(), ()) )
				continue
			if (i[0] == "T" if isinstance(i, tuple) else False):
				resultTuples.append( NodeAttrRef("T", attrWrapper.name(), i[2]) )
				continue

			# separate nodes / node terms from path if given
			refs = [NodeAttrRef(node.uid, i.attr, i.path)
			         for node in resolveNodes(i.uid, graph, parentNode)]
			resultTuples.extend(refs)

		branch.value = resultTuples

def populateExpandedTree(expandedTree:Tree[list[NodeAttrRef]],
                         attrWrapper:NodeAttrWrapper,
						 parentNode:ChimaeraNode,
						 graph:ChimaeraNode)->Tree:
	"""populate the expanded tree with rich trees -
	expand each node attr ref into a rich tree"""

	for branch in expandedTree.allBranches(includeSelf=True):
		newValue = []
		for i in branch.value: #type:NodeAttrRef
			#log("populateExpandedTree", i, i.uid)

			if i.uid == "T":
				newValue.append(parentNode.getAttrInputFromType(
					attrWrapper.name(), parentNode))
				continue
			# look at this beautiful line
			newValue.append(graph.getNodes(i.uid)[0]._attrMap[i.attr].resolve()[i.path])
		branch.value = newValue

def overlayPopulatedTree(populatedTree:Tree[list[Tree]],
                         attrWrapper:NodeAttrWrapper,
                         parentNode:ChimaeraNode,
                         graph:ChimaeraNode)->Tree:
	"""overlay the populated tree -
	overlay each tree in populated branch value, left to right
	then for any child branches in populated tree,
	overlay the result branch at that path with the overlaid result
	"""
	resultTree = Tree("root")
	for populatedBranch in populatedTree.allBranches(includeSelf=True,
	                                        depthFirst=True,
	                                        topDown=True):
		address = populatedBranch.address(
			#includeSelf=True,
			includeSelf=True,
			includeRoot=False, uid=False)
		#log(address)
		resultBranch = resultTree(
			address,
						create=True)
		for i in populatedBranch.value:
			resultBranch = treelib.overlayTreeInPlace(resultBranch, i,
			                                          mode="union")
	return resultTree



def resolveIncomingTree(
		rawTree:Tree,
		attrWrapper:NodeAttrWrapper,
		parentNode:ChimaeraNode,
		graph:ChimaeraNode)->Tree:
	"""resolve the incoming tree for this attribute -
	replace string references with trees to compose"""
	resultTree = rawTree.copy()

	# expand expressions to tuples of (node uid, attribute, path)
	expandIncomingTree(resultTree, attrWrapper, parentNode, graph)

	## this tree is cached so we don't have to resolve a load of node addresses
	# every time - everything in resultTree should now be exact UIDs

	# populate expanded tree with rich attribute trees
	populateExpandedTree(resultTree, attrWrapper, parentNode, graph)


	# overlay rich trees into final incoming data stream
	return overlayPopulatedTree(resultTree, attrWrapper, parentNode, graph)




def getEmptyTree():
	return Tree("root")


def newAttributeData(name: str, linking=("T", ), override=())->Tree:
	"""test not using a 'root' branch -
	doesn't serve much purpose, ui will just be set on the top branch"""
	t = Tree(name)
	t["linking"] = list(linking)
	t["override"] = list(override)
	return t

class NodeAttrWrapper:
	"""TODO: move logic from this into main chimaeraNode -
	use this class only as convenience to track data branch"""
	def __init__(self, attrData:Tree, node:ChimaeraNode):
		self.tree = attrData
		self._cachedIncomingExpandedTree : Tree = None
		self.node = node

		# set isRoot = True on "root" branches -
		# feels wrong to have this here, only other place I could think
		# is in the Modeled newData() function?
		# self.tree("linking", "root").isRoot = True
		# self.tree("override", "root").isRoot = True
		self.tree("linking").isRoot = True
		self.tree("override").isRoot = True

	def name(self)->str:
		return self.tree.name

	# region locally defined overrides
	def linking(self)->Tree:
		return self.tree("linking")
	def override(self)->Tree:
		return self.tree("override")

	# def setOverride(self, value):
	# 	"""manually override"""
	# 	self._tree["override"] = value
	#endregion

	# incoming connections
	def clearCachedIncomingExpandedTree(self):
		self._cachedIncomingExpandedTree = None
	def cachedIncomingExpandedTree(self)->Tree:
		"""return the cached incoming expanded tree"""
		return self._cachedIncomingExpandedTree
	def setCachedIncomingExpandedTree(self, value:Tree):
		"""set the cached incoming expanded tree"""
		self._cachedIncomingExpandedTree = value

	def incomingTreeRaw(self)->Tree:
		"""raw tree with string filters"""
		return self.tree("incoming", create=False)

	def incomingTreeExpanded(self)->Tree:
		"""incoming tree expanded to tuples of (node uid, attribute, path)
		returns live reference to cached tree, copy if you want to modify
		"""

		if self.cachedIncomingExpandedTree() is None:

			resultTree = self.incomingTreeRaw().copy()

			# expand expressions to tuples of (node uid, attribute, path)
			expandIncomingTree(resultTree, self, self.node, self.node.parent())
			self.setCachedIncomingExpandedTree(resultTree)

		return self.cachedIncomingExpandedTree()

	def resolveIncomingTree(self, incomingExpandedTree)->Tree:
		"""incoming tree resolved to final data"""
		#print("SELF", self, self.node, self.node.parent())

		# use cached expanded tree if possible
		# if self.cachedIncomingExpandedTree() is None:
		# 	resultTree = self.incomingTreeRaw().copy()
		#
		# 	# expand expressions to tuples of (node uid, attribute, path)
		# 	expandIncomingTree(resultTree, self, self.node, self.node.parent())
		#
		# 	self.setCachedIncomingExpandedTree(resultTree.copy())

		# expand expressions to tuples of (node uid, attribute, path)
		#incomingExpandedTree.display()
		expandedTree = incomingExpandedTree.copy()

		# populate expanded tree with rich attribute trees
		populateExpandedTree(expandedTree, self, self.node, self.node.parent())

		# overlay rich trees into final incoming data stream
		return overlayPopulatedTree(expandedTree, self, self.node, self.node.parent())





	def setIncoming(self, value:(str, list[str])):
		"""reset incoming connections to given value"""

	# endregion

	# # region resolved value
	# def resolve(self)->Tree:
	# 	"""return the resolved tree for this attribute.
	#
	# 	if defined is callable, call it with incoming data
	# 	 and this node.
	# 	if defined is tree, compose it with incoming and
	# 	eval any expressions
	#
	# 	nodeType defines default behaviour in composition, which may
	# 	be overridden at any level of tree
	# 	"""
	#
	# 	#log("RESOLVE")
	# 	#log("attr", self.name(), self.node)
	#
	# 	if self.name() == "value":
	# 		# send to nodeType compute
	# 		return self.node.compute(
	# 			self.resolveIncomingTree(
	# 				self.incomingTreeExpanded()
	# 			)
	# 		)
	# 	if not self.node.parent(): # unparented nodes can't do complex behaviour
	# 		return self.override()
	# 	incoming = self.resolveIncomingTree(
	# 		self.incomingTreeExpanded()
	# 	)
	#
	# 	defined = self.override()
	#
	# 	return treelib.overlayTrees([incoming, defined])
	#
	# 	try:
	# 		return treelib.overlayTrees([incoming, defined])
	# 	except Exception as e:
	# 		log("error overlaying incoming and defined")
	# 		log("incoming")
	# 		incoming.display()
	# 		log("override")
	# 		defined.display()
	# 		raise e

	# def resolveToList(self)->list:
	# 	"""return the resolved value of the top branch for this attribute
	# 	"""
	# 	val = self.resolve().value
	# 	if isinstance(val, list):
	# 		return val
	# 	return [val]

	def __call__(self) -> Tree:
		# specialcase type, messy but whatever
		if self.name() == "type" :

			# T E M P
			# use directly defined type for now to avoid recursion
			return ChimaeraNode.getNodeType(self.override().value[0])

		return self.resolve()