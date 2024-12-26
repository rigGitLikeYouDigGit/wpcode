from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from maya import OpenMaya as om1

from wpm import om, cmds

if T.TYPE_CHECKING:
	import swig

"""any specific resources for the old api"""

def om1GetMObject(node)->om1.MObject:
	"""this is specialised for dg nodes -
	component MObjects will have their own functions anyway if needed
	"""
	if isinstance(node, om1.MObject):
		if node.isNull():
			raise RuntimeError("OM1 object for ", node, " is invalid")
		return node
	elif isinstance(node, str):
		sel = om1.MSelectionList()
		mobj = om1.MObject()
		sel.add(node)
		sel.getDependNode(0, mobj)
		if mobj.isNull():
			raise RuntimeError("OM1 object for ", node, " is invalid")
		return mobj
	elif isinstance(node, om1.MDagPath):
		return node.node()
	else:
		try:
			return node.object()  # supports MFnBase and WN
		except:
			pass
		raise TypeError("OM1 Cannot retrieve MObject from ", node, type(node))

def getRawPoints(obj:om1.MObject): # rtype:<Swig Object of type 'float *'>
	return om1.MFnMesh(obj).getRawPoints()
def getRawDoublePoints(obj:om1.MObject): # rtype:<Swig Object of type 'double *'>
	return om1.MFnMesh(obj).getRawDoublePoints()

#### AMAZING functions from tfox on techArtists.org
import numpy as np
from ctypes import c_float, c_double, c_int, c_uint

_CONVERT_DICT = {
	om1.MPointArray:		  (float, 4, c_double, om1.MScriptUtil.asDouble4Ptr),
	om1.MFloatPointArray:  (float, 4, c_float , om1.MScriptUtil.asFloat4Ptr),
	om1.MVectorArray:	  (float, 3, c_double, om1.MScriptUtil.asDouble3Ptr),
	om1.MFloatVectorArray: (float, 3, c_float , om1.MScriptUtil.asFloat3Ptr),
	om1.MDoubleArray:	  (float, 1, c_double, om1.MScriptUtil.asDoublePtr),
	om1.MFloatArray:		  (float, 1, c_float , om1.MScriptUtil.asFloatPtr),
	om1.MIntArray:		  (int	, 1, c_int	 , om1.MScriptUtil.asIntPtr),
	om1.MUintArray:		  (int	, 1, c_uint  , om1.MScriptUtil.asUintPtr),
}

def _swigConnect(mArray, count, util):
	'''
	Use an MScriptUtil to build SWIG array that we can read from and write to.
	Make sure to get the MScriptUtil from outside this function, otherwise
	it may be garbage collected
	The _CONVERT_DICT holds {mayaType: (pyType, numComps, cType, ptrType)} where
		pyType: The type that is used to fill the MScriptUtil array.
		numComps: The number of components. So a double4Ptr would be 4
		cType: The ctypes type used to read the data
		ptrType: An unbound method on MScriptUtil to cast the pointer to the correct type
			I can still call that unbound method by manually passing the usually-implicit
			self argument (which will be an instance of MScriptUtil)
	'''
	pyTyp, comps, ctp, ptrTyp = _CONVERT_DICT[type(mArray)]
	cc = (count * comps)
	util.createFromList([pyTyp()] * cc, cc)

	#passing util as 'self' to call the unbound method
	ptr = ptrTyp(util)
	mArray.get(ptr)

	if comps == 1:
		cdata = ctp * count
	else:
		# Multiplication follows some strange rules here
		# I would expect (ctype*3)*N to be an Nx3 array (ctype*3 repeated N times)
		# However, it gets converted to a 3xN array
		cdata = (ctp * comps) * count

	# int(ptr) gives the memory address
	cta = cdata.from_address(int(ptr))

	# This makes numpy look at the same memory as the ctypes array
	# so we can both read from and write to that data through numpy
	npArray = np.ctypeslib.as_array(cta)
	return npArray, ptr

def mayaToNumpy(mArray):
	''' Convert a maya array to a numpy array
	Parameters
	----------
	ary : MArray
		The maya array to convert to a numpy array
	Returns
	-------
	: np.array :
		A numpy array that contains the data from mArray
	'''
	util = om1.MScriptUtil()
	count = mArray.length()
	npArray, _ = _swigConnect(mArray, count, util)
	return np.copy(npArray)

def numpyToMaya(ary, mType):
	''' Convert a numpy array to a specific maya type array
	Parameters
	----------
	ary : np.array
		The numpy array to convert to a maya array
	mType : type
		The maya type to convert to out of: MPointArray, MFloatPointArray, MVectorArray,
		MFloatVectorArray, MDoubleArray, MFloatArray, MIntArray, MUintArray
	Returns
	-------
	: mType :
		An array of the provided type that contains the data from ary
	'''
	util = om1.MScriptUtil()
	# Add a little shape checking
	comps = _CONVERT_DICT[mType][1]
	if comps == 1:
		if len(ary.shape) != 1:
			raise ValueError("Numpy array must be 1D to convert to the given maya type")
	else:
		if len(ary.shape) != 2:
			raise ValueError("Numpy array must be 2D to convert to the given maya type")
		if ary.shape[1] != comps:
			msg = "Numpy array must have the proper shape. Dimension 2 has size {0}, but needs size {1}"
			raise ValueError(msg.format(ary.shape[1], comps))
	count = ary.shape[0]
	mArray = mType(count)
	npArray, ptr = _swigConnect(mArray, count, util)
	np.copyto(npArray, ary)
	return mType(ptr, count)


_NTYPE_DICT={
	om1.MFnNumericData.kInvalid: (om1.MDataHandle.asDouble,  om1.MDataHandle.setDouble),
	om1.MFnNumericData.kFloat:   (om1.MDataHandle.asDouble,  om1.MDataHandle.setDouble),
	om1.MFnNumericData.kDouble:  (om1.MDataHandle.asDouble,  om1.MDataHandle.setDouble),
	om1.MFnNumericData.kByte:    (om1.MDataHandle.asInt,     om1.MDataHandle.setInt),
	om1.MFnNumericData.kChar:    (om1.MDataHandle.asChar,    om1.MDataHandle.setChar),
	om1.MFnNumericData.kShort:   (om1.MDataHandle.asShort,   om1.MDataHandle.setShort),
	om1.MFnNumericData.kInt:     (om1.MDataHandle.asInt,     om1.MDataHandle.setInt),
	#om1.MFnNumericData.kInt64:  (om1.MDataHandle.asInt,     om1.MDataHandle.setInt64),
	om1.MFnNumericData.kAddr:    (om1.MDataHandle.asInt,     om1.MDataHandle.setInt),
	om1.MFnNumericData.kLong:    (om1.MDataHandle.asInt,     om1.MDataHandle.setInt),
	om1.MFnNumericData.kBoolean: (om1.MDataHandle.asBool,    om1.MDataHandle.setBool),

	om1.MFnNumericData.k2Short:  (om1.MDataHandle.asShort2,  om1.MDataHandle.set2Short),
	om1.MFnNumericData.k2Long:   (om1.MDataHandle.asInt2,    om1.MDataHandle.set2Int),
	om1.MFnNumericData.k2Int:    (om1.MDataHandle.asInt2,    om1.MDataHandle.set2Int),
	om1.MFnNumericData.k3Short:  (om1.MDataHandle.asShort3,  om1.MDataHandle.set3Short),
	om1.MFnNumericData.k3Long:   (om1.MDataHandle.asInt3,    om1.MDataHandle.set3Int),
	om1.MFnNumericData.k3Int:    (om1.MDataHandle.asInt3,    om1.MDataHandle.set3Int),
	om1.MFnNumericData.k2Float:  (om1.MDataHandle.asFloat2,  om1.MDataHandle.set2Float),
	om1.MFnNumericData.k2Double: (om1.MDataHandle.asDouble2, om1.MDataHandle.set2Double),
	om1.MFnNumericData.k3Float:  (om1.MDataHandle.asFloat3,  om1.MDataHandle.set3Float),
	om1.MFnNumericData.k3Double: (om1.MDataHandle.asDouble3, om1.MDataHandle.set3Double),
}

_DTYPE_DICT = {
	om1.MFn.kPointArrayData:  (om1.MFnPointArrayData,  om1.MPointArray),
	om1.MFn.kDoubleArrayData: (om1.MFnDoubleArrayData, om1.MDoubleArray),
	om1.MFn.kFloatArrayData:  (om1.MFnFloatArrayData,  om1.MFloatArray),
	om1.MFn.kIntArrayData:    (om1.MFnIntArrayData,    om1.MIntArray),
	om1.MFn.kUInt64ArrayData: (om1.MFnUInt64ArrayData, om1.MPointArray),
	om1.MFn.kVectorArrayData: (om1.MFnVectorArrayData, om1.MVectorArray),
}

def getNumpyAttr(attrName):
	''' Read attribute data directly from the plugs into numpy
	This function will read most numeric data types directly into numpy arrays
	However, some simple data types (floats, vectors, etc...) have api accessors
	that return python tuples. These will not be turned into numpy arrays.
	And really, if you're getting simple data like that, just use cmds.getAttr
	Parameters
	----------
	attrName : str or om1.MPlug
		The name of the attribute to get (For instance "pSphere2.translate", or "group1.pim[0]")
		Or the MPlug itself
	Returns
	-------
	: object :
		The numerical data from the provided plug. A np.array, float, int, or tuple
	'''
	if isinstance(attrName, str):
		sl = om1.MSelectionList()
		sl.add(attrName)
		plug = om1.MPlug()
		sl.getPlug(0, plug)
	elif isinstance(attrName, om1.MPlug):
		plug = attrName

	#First just check if the data is numeric
	mdh = plug.asMDataHandle()
	if mdh.isNumeric():
		# So, at this point, you should really just use getattr
		ntype = mdh.numericType()
		if ntype in _NTYPE_DICT:
			return _NTYPE_DICT[ntype][0](mdh)
		elif ntype == om1.MFnNumericData.k4Double:
			NotImplementedError("Haven't implemented double4 access yet")
		else:
			raise RuntimeError("I don't know how to access data from the given attribute")
	else:
		# The data is more complex than a simple number.
		try:
			pmo = plug.asMObject()
		except RuntimeError:
			# raise a more descriptive error. And make sure to actually print the plug name
			raise RuntimeError("I don't know how to access data from the given attribute")
		apiType = pmo.apiType()

		# A list of types that I can just pass to mayaToNumpy
		if apiType in _DTYPE_DICT:
			fn, dtype = _DTYPE_DICT[apiType]
			fnPmo = fn(pmo)
			ary = fnPmo.array()
			return mayaToNumpy(ary)

		elif apiType == om1.MFn.kComponentListData:
			fnPmo = om1.MFnComponentListData(pmo)
			mirs = []
			mir = om1.MIntArray()
			for attrIndex in range(fnPmo.length()):
				fnEL = om1.MFnSingleIndexedComponent(fnPmo[attrIndex])
				fnEL.getElements(mir)
				mirs.append(mayaToNumpy(mir))
			return np.concatenate(mirs)

		elif apiType == om1.MFn.kMatrixData:
			fnPmo = om1.MFnMatrixData(pmo)
			mat = fnPmo.matrix()
			return mayaToNumpy(mat)
		else:
			apiTypeStr = pmo.apiTypeStr()
			raise NotImplementedError("I don't know how to handle {0} yet".format(apiTypeStr))
	raise NotImplementedError("Fell all the way through")

def setNumpyAttr(attrName, value):
	''' Write a numpy array directly into a maya plug
	This function will handle most numeric plug types.
	But for single float, individual point, etc.. types, consider using cmds.setAttr
	THIS DOES NOT SUPPORT UNDO
	Parameters
	----------
	attrName : str or om1.MPlug
		The name of the attribute to get (For instance "pSphere2.translate", or "group1.pim[0]")
		Or the MPlug itself
	value : int, float, tuple, np.array
		The correctly typed value to set on the attribute
	'''
	if isinstance(attrName, str):
		sl = om1.MSelectionList()
		sl.add(attrName)
		plug = om1.MPlug()
		sl.getPlug(0, plug)
	elif isinstance(attrName, om1.MPlug):
		plug = attrName
	else:
		raise ValueError("Data must be string or MPlug. Got {0}".format(type(attrName)))

	#First just check if the data is numeric
	mdh = plug.asMDataHandle()
	if mdh.isNumeric():
		# So, at this point, you should really just use setattr
		ntype = mdh.numericType()
		if ntype in _NTYPE_DICT:
			_NTYPE_DICT[ntype][1](mdh, *value)
			plug.setMObject(mdh.data())
		elif ntype == om1.MFnNumericData.k4Double:
			NotImplementedError("Haven't implemented double4 access yet")
		else:
			raise RuntimeError("I don't know how to set data on the given attribute")
	else:
		# The data is more complex than a simple number.
		try:
			pmo = plug.asMObject()
		except RuntimeError:
			# raise a more descriptive error. And make sure to actually print the plug name
			raise RuntimeError("I don't know how to access data from the given attribute")
		apiType = pmo.apiType()

		if apiType in _DTYPE_DICT:
			# build the pointArrayData
			fnType, mType = _DTYPE_DICT[apiType]
			fn = fnType()
			mPts = numpyToMaya(value, mType)
			dataObj = fn.create(mPts)
			plug.setMObject(dataObj)
			return

		elif apiType == om1.MFn.kComponentListData:
			fnCompList = om1.MFnComponentListData()
			compList = fnCompList.create()
			fnIdx = om1.MFnSingleIndexedComponent()
			idxObj = fnIdx.create(om1.MFn.kMeshVertComponent)
			mIdxs = numpyToMaya(value, om1.MIntArray)
			fnIdx.addElements(mIdxs)
			fnCompList.add(idxObj)
			plug.setMObject(compList)
			return
		else:
			apiTypeStr = pmo.apiTypeStr()
			raise NotImplementedError("I don't know how to handle {0} yet".format(apiTypeStr))

	raise NotImplementedError("WTF? How did you get here??")

################################################################################

def test():
	import time
	from maya import cmds
	meshName = 'pSphere1'
	bsName = 'blendShape1'
	meshIdx = 0
	bsIdx = 0

	# A quick test showing how to build a numpy array
	# containing the deltas for a shape on a blendshape node
	numVerts = cmds.polyEvaluate(meshName, vertex=True)
	baseAttr = '{0}.it[{1}].itg[{2}].iti[6000]'.format(bsName, meshIdx, bsIdx)
	inPtAttr = baseAttr + '.inputPointsTarget'
	inCompAttr = baseAttr + '.inputComponentsTarget'

	start = time.time()
	points = getNumpyAttr(inPtAttr)
	idxs = getNumpyAttr(inCompAttr)
	ret = np.zeros((numVerts, 4))
	ret[idxs] = points
	end = time.time()





