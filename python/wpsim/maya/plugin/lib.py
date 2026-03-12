from __future__ import annotations
import types, typing as T
import pprint
import numpy as np
from wplib import log

import jax
import jax.numpy as jnp

from wpm import cmds, om
from wpsim.kine.builder import (BuilderBody, BuilderMesh,
                                BuilderTransform, BuilderNurbsCurve)
# can't use optimum here as we get an api2 object
# from the node mesh data
#from wpm.core.numpymaya import rawPointsFromMeshObj

def makeGenericInputAttribute(gFn:om.MFnGenericAttribute, name:str)->om.MObject:
	""""""
	aParamValue = gFn.create(name, name )
	gFn.addNumericType(om.MFnNumericData.kInt)
	gFn.addNumericType(om.MFnNumericData.k3Int)
	gFn.addNumericType(om.MFnNumericData.kBoolean)
	gFn.addNumericType(om.MFnNumericData.kShort)
	gFn.addNumericType(om.MFnNumericData.k3Short)
	gFn.addNumericType(om.MFnNumericData.kFloat)
	gFn.addNumericType(om.MFnNumericData.k3Float)
	gFn.addNumericType(om.MFnNumericData.kDouble)
	gFn.addNumericType(om.MFnNumericData.k3Double)
	gFn.addNumericType(om.MFnNumericData.kFloatArray)
	gFn.addNumericType(om.MFnNumericData.kDoubleArray)
	gFn.addNumericType(om.MFnNumericData.kIntArray)

	gFn.addDataType(om.MFnData.kString)
	gFn.addDataType(om.MFnData.kStringArray)
	gFn.addDataType(om.MFnData.kMatrix)
	gFn.addDataType(om.MFnData.kMatrixArray)
	gFn.addDataType(om.MFnData.kPointArray)
	gFn.addDataType(om.MFnData.kVectorArray)
	# gFn.addDataType(om.MFnData.kMesh)
	# gFn.addDataType(om.MFnData.kNurbsCurve)
	# gFn.addDataType(om.MFnData.kNurbsSurface)
	return aParamValue


def arrayFromGenericInput(pDH:om.MDataHandle)->jnp.ndarray|str|list[str]:
	"""extract array data from a generic input attribute"""
	if pDH.isNumeric():
		if(pDH.numericType() == om.MFnNumericData.kDouble):
			return jnp.array([pDH.asDouble()])
		elif(pDH.numericType() == om.MFnNumericData.kFloat):
			return jnp.array([pDH.asFloat()])
		elif(pDH.numericType() == om.MFnNumericData.kInt):
			return jnp.array([pDH.asInt()])
		elif(pDH.numericType() == om.MFnNumericData.k3Double):
			return jnp.array(pDH.asDouble3())
		elif(pDH.numericType() == om.MFnNumericData.k3Float):
			return jnp.array(pDH.asFloat3())
	if pDH.type() == om.MFnData.kString:
		return pDH.asString()
	if pDH.type() == om.MFnData.kStringArray:
		return list(om.MFnStringArrayData(pDH.data()).array())
	if pDH.type() == om.MFnData.kMatrix:
		return jnp.array(om.MFnMatrixData(pDH.data()).matrix())
	if pDH.type() == om.MFnData.kMatrixArray:
		return jnp.array(om.MFnMatrixArrayData(pDH.data()).array())
	if pDH.type() == om.MFnData.kPointArray:
		return jnp.array(om.MFnPointArrayData(pDH.data()).array())
	if pDH.type() == om.MFnData.kVectorArray:
		return jnp.array(om.MFnVectorArrayData(pDH.data()).array())
	raise TypeError(f"Unsupported data type for generic input: "
	                f"{pDH.type(), pDH.typeId()}")

def makeValueOrStrAttributes(name, tFn, gFn)->tuple[om.MObject, om.MObject]:
	"""make a pair of attributes - one for a direct value, and one for a string input
	that can be used to reference other attributes. The node will use the value attribute
	if the string attribute is empty, otherwise it will try to resolve the string as an
	attribute path and use that value instead.
	"""
	aStr = tFn.create(f"{name}Str", f"{name}Str", om.MFnData.kString)
	aValue = makeGenericInputAttribute(gFn, name + "Val")
	return aValue, aStr


def makeFloatOrStrAttributes(name, tFn, nFn)->tuple[om.MObject, om.MObject]:
	"""make a pair of attributes - one for a direct value, and one for a string input
	that can be used to reference other attributes. The node will use the value attribute
	if the string attribute is empty, otherwise it will try to resolve the string as an
	attribute path and use that value instead.
	"""
	aStr = tFn.create(f"{name}Str", f"{name}Str", om.MFnData.kString)
	aValue = nFn.create(f"{name}Val", f"{name}Val",
	                    om.MFnNumericData.kDouble, 0.0)
	return aValue, aStr

def makeBodyAttributes(cls):
	"""create consistent set of static MObjects to represent rigid
	body attributes.
	Repetition of 'body' in each of these is intentional to avoid
	name clashes when multiple nodes use these attributes. Also to be absolutely
	clear in solver node
	"""
	tFn = om.MFnTypedAttribute()
	mFn = om.MFnMatrixAttribute()
	nFn = om.MFnNumericAttribute()
	cFn = om.MFnCompoundAttribute()
	# leave blank to just use name of node
	cls.aBodyName = tFn.create("name", "name", om.MFnData.kString)
	# worldspace initial matrix
	cls.aBodyMatrix = mFn.create("matrix", "matrix")
	# combined collision and mass mesh
	cls.aBodyMesh = tFn.create("mesh", "mesh", om.MFnData.kMesh)

	cls.aBodyActive = nFn.create("active", "active", om.MFnNumericData.kBoolean,
	                         True)

	cls.aBodyMass = nFn.create("mass", "mass", om.MFnNumericData.kDouble, 1.0)
	nFn.setMin(0.0)
	cls.aBodyDamping = nFn.create("damping", "damping", om.MFnNumericData.kDouble,
	                          0.0)
	nFn.setMin(0.0)

	cls.aBodyIndex = nFn.create("index", "index", om.MFnNumericData.kInt, -1)
	nFn.setMin(-1)
	nFn.writable = False

	# aux geo
	cls.auxBodyTf = cFn.create("auxTf", "auxTf")
	cFn.array = True
	cFn.usesArrayDataBuilder = True
	cls.auxBodyTfName = tFn.create("auxTfName", "auxTfName", om.MFnData.kString)
	# local space matrix
	cls.auxBodyMatrix = mFn.create("matrix", "matrix")
	cFn.addChild(cls.auxBodyTfName)
	cFn.addChild(cls.auxBodyMatrix)

	cls.auxBodyCurve = cFn.create("auxCurve", "auxCurve")
	cFn.array = True
	cFn.usesArrayDataBuilder = True
	cls.auxBodyCurveName = tFn.create("auxCurveName", "auxCurveName",
	                              om.MFnData.kString)
	cls.auxBodyCurveData = tFn.create("auxCurveData", "auxCurveData",
	                              om.MFnData.kNurbsCurve)
	cFn.addChild(cls.auxBodyCurveName)
	cFn.addChild(cls.auxBodyCurveData)
	return [cls.aBodyName, cls.aBodyMatrix, cls.aBodyMesh, cls.aBodyActive,
	        cls.aBodyMass, cls.aBodyDamping, cls.aBodyIndex,
	        cls.auxBodyTfName, cls.auxBodyMatrix, cls.auxBodyMesh,
	        cls.auxBodyCurveName, cls.auxBodyCurveData]


def quaternionFromMatrix(mat:om.MMatrix)->om.MQuaternion:
	"""extract quaternion from MMatrix"""
	mTrans = om.MTransformationMatrix(mat)
	return mTrans.rotation()

def computeMeshCOMInertia(mesh:om.MFnMesh=None,
                          positions=None)->jnp.ndarray:
	"""compute center of mass of a mesh and inertia tensor using vectorized operations
	Returns 4x4 matrix where:
	- upper-left 3x3 is the inertia tensor in world space
	- 4th column (first 3 rows) is center of mass position
	- bottom row is [0, 0, 0, 1]
	"""

	verts = positions or jnp.array(mesh.getPoints())[:, :3]
	numVerts = verts.shape[0]

	# Calculate center of mass (vectorized mean)
	com = jnp.mean(verts, axis=0)

	# Calculate relative positions from center of mass
	r = verts - com
	rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]

	# Calculate squared distance from COM
	r2 = jnp.sum(r * r, axis=1)

	# Calculate inertia tensor components (vectorized)
	# I = sum(mass_i * (r_i^2 * I - r_i ⊗ r_i))
	# Assuming uniform density, mass_i = 1/numVerts
	Ixx = jnp.sum(r2 - rx * rx) / numVerts
	Iyy = jnp.sum(r2 - ry * ry) / numVerts
	Izz = jnp.sum(r2 - rz * rz) / numVerts
	Ixy = -jnp.sum(rx * ry) / numVerts
	Ixz = -jnp.sum(rx * rz) / numVerts
	Iyz = -jnp.sum(ry * rz) / numVerts

	# Build MMatrix: inertia tensor in upper 3x3, COM in 4th column
	result = jnp.array([
		[Ixx, Ixy, Ixz, 0.0],
		[Ixy, Iyy, Iyz, 0.0],
		[Ixz, Iyz, Izz, 0.0],
		[com[0], com[1], com[2], 1.0]
	])

	return result

def builderMeshDataFromMFnMesh(
		meshFn:om.MFnMesh,
		name="",
		parent="",
		mass=1.0
)->BuilderMesh:
	"""construct builder mesh data from an MFnMesh"""

	# get raw points
	points = jnp.array(meshFn.getPoints())[:, :3]

	# get face vertex counts and indices
	indices = jnp.array(meshFn.getTriangles()[1]).reshape(-1, 3)

	comData = computeMeshCOMInertia(positions=points)

	return BuilderMesh(
		name=name,
		parent=parent,
		points=points,
		indices=indices,
		com=comData[3, :3],
		inertia=comData[:3, :3],
		mass=mass
	)


def builderCurveDataFromMFnNurbsCurve(
		curveFn:om.MFnNurbsCurve,
		name="",
		parent="",
)->BuilderNurbsCurve:
	"""construct builder nurbs curve data from an MFnNurbsCurve"""

	# get raw points
	points = jnp.array(curveFn.cvPositions())[:, :3]

	# get knot vector
	knots = jnp.array(curveFn.knots())

	return BuilderNurbsCurve(
		name=name,
		parent=parent,
		points=points,
		degree=curveFn.degree(),
		knots=knots,
		isPeriodic=(curveFn.form == om.MFnNurbsCurve.kPeriodic)
	)
