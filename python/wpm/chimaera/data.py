

from __future__ import annotations
import typing as T

from dataclasses import dataclass

import numpy as np

from wptree import Tree

from chimaera.core import DataTree

from wpm import om, WN
from wpm.core import classConstantValueToNameMap
from wpm.lib.tracker import NodeLifespanTracker, NodeHierarchyTracker
from wpm.lib import hierarchy as h



@dataclass
class Transform:
	"""data object for transform
	for now also includes optional dag path
	"""

	matrix : np.ndarray
	rotateOrder : str = "XYZ"
	dagPath : str = ""

	def applyToMObject(self, mayaObject:om.MObject):
		"""set mayaObject to this transform data
		for now use direct api calls, might look into DGModifiers
		if it's faster or lets us build changes with multiple threads
		"""
		mfn = om.MFnTransform(mayaObject)
		mfn.setTranslation(om.MVector(self.matrix[3, :3]), om.MSpace.kTransform)
		mfn.setRotation(om.MEulerRotation(self.matrix[:3, :3]), om.MSpace.kTransform)
		mfn.setScale(om.MVector(self.matrix[:3, :3]), om.MSpace.kTransform)

	@classmethod
	def fromMObject(cls, obj:om.MObject):
		mfn = om.MFnTransform(obj)
		matrix = np.identity(4)
		matrix[:3, :3] = np.array(mfn.rotation().asMatrix())
		matrix[:3, 3] = np.array(mfn.translation())
		matrix[:3, :3] *= np.array(mfn.scale())
		return cls(matrix)

@dataclass
class Mesh:
	"""data object for mesh shape
	"""

	facePointCounts : np.ndarray
	facePointConnects : np.ndarray
	pointCoords : np.ndarray

	def applyToMObject(self, mayaObject:om.MObject):
		"""set mayaObject to this transform data
		for now use direct api calls, might look into DGModifiers
		if it's faster or lets us build changes with multiple threads
		"""
		mfn = om.MFnMesh(mayaObject)
		mfn.createInPlace(
			self.pointCoords, self.facePointCounts, self.facePointConnects
		)

	@classmethod
	def fromMObject(cls, obj:om.MObject):
		mfn = om.MFnMesh(obj)
		return cls(
			*mfn.getVertices(),
			np.array(mfn.getPoints())[:, :3]
		)


class MayaData(DataTree):
	"""DataTree for Maya data.
	not sure of inheritance, only sketch for now

	If we use dataclasses as tree values, how do we detect changes to them?
	"""


	def createMayaNode(self, parentNode:om.MObject):
		pass



class TransformData(MayaData):

	def createMayaNode(self, parentNode:om.MObject):
		data : Transform = self.value
		dagMod = om.MDagModifier()
		transform = dagMod.createNode("transform", parentNode)
		dagMod.doIt()

		data.applyToMObject(transform)



def gatherData(topNode:om.MObject):
	"""return tree from data"""
	objMap : dict[om.MObject, DataTree] = {}
	topPath = om.MDagPath.getAPathTo(topNode)
	topMFn = om.MFnDagNode(topPath)
	topTree = Tree("root")
	for node in h.iterDagChildren(topNode, includeRoot=False):
		mfn = om.MFnDagNode(om.MDagPath.getAPathTo(node))
		relPath = h.relativeDagTokens(topPath, mfn.dagPath())
		topTree(*relPath, create=True).value = mfn.name()

	return topTree




class MayaDataView(NodeHierarchyTracker):
	"""specify top transform to display data of given
	object, updating live when data changes.
	Also allow maya changes to affect data,
	if they pass validation.

	Can we formalise this somehow - effectively
	managing a central source of truth with multiple
	views and ways to edit it.
	Feels like a server and clients

	"""

	# def __init__(self, node:om.MObject, data:DataTree=None):
	# 	super(MayaDataView, self).__init__(node)
	# 	self.data : DataTree = None

	def displayDataTree(self):
		tree = gatherData(self.node())
		tree.display()


	def onChildNameChanged(self, node:om.MObject, prevName:str, *args):
		self.displayDataTree()

	def onChildNodeAdded(self, childPath:om.MDagPath, parentPath:om.MDagPath):
		self.displayDataTree()

	def onChildNodeRemoved(self,
	                       newChildPath:om.MDagPath,
	                       newParentPath:om.MDagPath,
	                       oldChildPath:om.MDagPath,
	                       oldParentPath:om.MDagPath
	                       ):
		self.displayDataTree()



