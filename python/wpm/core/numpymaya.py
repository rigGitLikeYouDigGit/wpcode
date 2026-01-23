from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
import ctypes
import numpy as np
from maya import OpenMaya as om1

from wpm.core.api import getMObject, getMObjectOld, getMDagPath

def rawPointsFromMeshObj(
		meshObj:om1.MObject,
)->np.ndarray:
	"""cast rawpoints pointer directly into array """
	if not isinstance(meshObj, om1.MObject):
		api2dp = getMDagPath(meshObj)
		meshObj = getMObjectOld(api2dp.fullPathName())
	meshFn = om1.MFnMesh()
	meshFn.setObject(meshObj)
	rawPoints = meshFn.getRawPoints()
	numPoints = meshFn.numVertices()
	cFloatArray = (ctypes.c_float * numPoints * 3).from_address(int(rawPoints))
	arr = np.ctypeslib.as_array(cFloatArray)

def rawNormalsFromMeshObj(
		meshObj:om1.MObject,
)->np.ndarray:
	"""cast rawnormals pointer directly into array """
	if not isinstance(meshObj, om1.MObject):
		api2dp = getMDagPath(meshObj)
		meshObj = getMObjectOld(api2dp.fullPathName())
	meshFn = om1.MFnMesh()
	meshFn.setObject(meshObj)
	rawNormals = meshFn.getRawNormals()
	numNormals = meshFn.numNormals()
	cFloatArray = (ctypes.c_float * numNormals * 3).from_address(int(rawNormals))
	arr = np.ctypeslib.as_array(cFloatArray)
	return arr