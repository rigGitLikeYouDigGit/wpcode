from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wpm import cmds, om

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

