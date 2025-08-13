from __future__ import annotations
"""lib functions specific for the plugin"""
import numpy as np

from tree.lib.string import camelJoin
from edRig.palette import *
from edRig import cmds, om
from edRig.maya.lib import attr

from edRig.maya.tool.feldspar.datastruct import BarData, FeldsparData, GroupData

if T.TYPE_CHECKING:
	from edRig.maya.tool.feldspar.plugin.setupnode import FeldsparSetupNode
	from edRig.maya.tool.feldspar.plugin.solvernode import FeldsparSolverNode


def makeVertexAttr(array=True, suffix="", prefix=""):
	"""creates consistent attribute for feldspar vertices
	"""
	cFn = om.MFnCompoundAttribute()
	nFn = om.MFnNumericAttribute()
	vertexName = camelJoin(prefix, "vertex", suffix)
	posName = camelJoin(prefix, "pos", suffix)
	indexName = camelJoin(prefix, "index", suffix)
	parentObj = cFn.create(vertexName, vertexName)
	posObj = nFn.createPoint(posName, posName)
	indexObj = nFn.create(indexName, indexName, om.MFnNumericData.kInt)

	childObjs = [posObj, indexObj]
	for i in childObjs:
		cFn.addChild(i)

	cFn.array = array
	cFn.usesArrayDataBuilder = array
	return [parentObj] + childObjs
# cls.aVertex, cls.aVertexPos, cls.aVertexIndex



def makeBarAttr(array=True):
	"""attr describing rigid bar connection between vertices"""
	cFn = om.MFnCompoundAttribute()
	nFn = om.MFnNumericAttribute()
	tFn = om.MFnTypedAttribute()

	parentObj = cFn.create("bar", "bar")

	indexAObj = nFn.create("vertex0", "vertex0", om.MFnNumericData.kInt)
	indexBObj = nFn.create("vertex1", "vertex1", om.MFnNumericData.kInt)
	bindLengthObj = nFn.create("bindLength", "bindLength", om.MFnNumericData.kFloat)
	targetLengthObj = nFn.create("targetLength", "targetLength", om.MFnNumericData.kFloat, 1.0)
	lengthObj = nFn.create("length", "length", om.MFnNumericData.kFloat)
	strengthObj = nFn.create("strength", "strength", om.MFnNumericData.kFloat, 100.0)
	softObj = nFn.create("soft", "soft", om.MFnNumericData.kBoolean, False)
	matrixObj = tFn.create("barMatrix", "barMatrix", om.MFnData.kMatrix,
	                    om.MFnMatrixData().create())

	childObjs = [indexAObj, indexBObj, softObj, bindLengthObj, targetLengthObj, lengthObj]

	for i in childObjs:
		cFn.addChild(i)
	cFn.array, cFn.usesArrayDataBuilder = array, array
	return [parentObj] + childObjs
# cls.aBar, cls.aBarVertexA, cls.aBarVertexB, cls.aBarBindLength, cls.aBarTargetLength, cls.aBarLength


def makeRigidGroupAttr(array=True):
	"""attr for rigid groups of vertices and bars
	also works to assign more rich data to its vertices"""
	cFn = om.MFnCompoundAttribute()
	nFn = om.MFnNumericAttribute()
	tFn = om.MFnTypedAttribute()

	parentObj = cFn.create("group", "group")

	cFn.array = array
	cFn.usesArrayDataBuilder = array

	vertexObj = nFn.create("groupVertexIndex", "groupVertexIndex", om.MFnNumericData.kInt)
	nFn.array=True
	nFn.usesArrayDataBuilder = True

	matrixObj = tFn.create("groupMatrix", "groupMatrix", om.MFnData.kMatrix,
	                       om.MFnMatrixData().create())
	fixedObj = nFn.create("groupFixed", "groupFixed", om.MFnNumericData.kBoolean, False)

	childObjs = [vertexObj, matrixObj, fixedObj]
	for i in childObjs:
		cFn.addChild(i)
	return [parentObj] + childObjs

	# cls.aGroup, cls.aGroupVertexIndex, cls.aGroupMatrix, cls.aGroupFixed

def vertexArrayFromVertexArrayDH(vertexArrayDH, vtxPosMObj)->np.ndarray:
	"""return (n, 3) array of vertex positions from given arrayDH"""
	vtxArray = np.zeros((len(vertexArrayDH), 3))
	for i, dh in attr.iterArrayDataHandle(vertexArrayDH):
		vtxArray[i] = dh.child(vtxPosMObj).asFloat3()
	return vtxArray

def barTiesFromBarArrayDH(barArrayDH, barAMObject, barBMObject)->np.ndarray:
	"""return (n, 2) array of ints for the basic indices of each bar"""
	barTiesArray = np.zeros((len(barArrayDH), 2), dtype=int)
	for i, dh in attr.iterArrayDataHandle(barArrayDH):
		barTiesArray[i][0] = dh.child(barAMObject).asInt()
		barTiesArray[i][1] = dh.child(barBMObject).asInt()
	return barTiesArray


def barDatasFromBarArrayDH(node:FeldsparSetupNode, barArrayDH,
                           vertexArray:np.ndarray) -> list[BarData]:
	"""return array of full data structs for each bar"""
	barDataArray = [None] * len(barArrayDH)
	for i, dh in attr.iterArrayDataHandle(barArrayDH):
		indices = (dh.child(node.aBarVertexA).asInt(),
			         dh.child(node.aBarVertexB).asInt())
		barLength = np.linalg.norm(
			vertexArray[indices[0]] - vertexArray[indices[1]])
		barDataArray[i] = BarData(
			indices=indices,
			length=barLength,
			soft=dh.child(node.aSoft).asBool(),
			bindLength=barLength,
			targetLength=dh.child(node.aBarTargetLength).asFloat()
		)
	return barDataArray


def groupDatasFromGroupArrayDH(node:FeldsparSetupNode, groupArrayDH)->list[GroupData]:
	groupDatas = [None] * len(groupArrayDH)
	for i, dh in attr.iterArrayDataHandle(groupArrayDH):
		vtxArrDH = om.MArrayDataHandle(dh.child(node.aGroupVertexIndex))
		vtxArr = attr.npIntArrayFromArrayDH(vtxArrDH)

		groupDatas[i] = GroupData(
			vtxArr,
			dh.child(node.aGroupFixed).asBool(),
			#np.array(dh.child(node.aGroupMatrix).asMatrix(), dtype=float)
		)
	return groupDatas

def arraysFromMFnMesh(meshFn:om.MFnMesh):
	"""return vertex array for points,
	and bar array for edges"""
	vtxArray = np.array(meshFn.getPoints())[:, :3]
	barArray = np.array([meshFn.getEdgeVertices(i) for i in meshFn.numEdges])
	return vtxArray, barArray











