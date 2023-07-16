
"""lib for core openmaya stuff
mainly MObject cache"""

from __future__ import annotations

import typing as T

from wplib import sequence
from tree.lib.object import UserSet


# small class representing each MFn type constant
from .cache import getCache, om


# access functions
def MObjectRegister():
	return getCache().mObjRegister

def apiTypeMap():
	return getCache().apiTypeMap

def apiTypeCodeMap():
	return getCache().apiTypeCodeMap

def apiCodeNameMap():
	return getCache().apiCodeNameMap

def apiTypeDataMap():
	return getCache().apiTypeDataMap

def mfnDataConstantTypeMap():
	return getCache().mfnDataConstantTypeMap

def toMFn(obj: (om.MObject, str, om.MFn))->om.MFnDependencyNode:
	"""return mfn function set initialised on the given object
	returns most specialised mfn possible for given object"""
	return getCache().getMFn(obj)

def getMFnType(obj: om.MObject) -> T.Type[om.MFnBase]:
	"""returns the highest available MFn
	for given object, based on sequence order
	above"""
	return getCache().getMFnType(obj)

def toMObject(node)->om.MObject:
	"""this is specialised for dg nodes -
	component MObjects will have their own functions anyway if needed
	"""
	return getCache().getMObject(node)

def nodeTypeFromMObject(mobj:om.MObject):
	"""return a nodeType string that can be passed to cmds.createNode
	"""
	return getCache().nodeTypeFromMObject(mobj)

# region specific mfn functions
# dissuaded other than for explicit function use, since toMFn is superior
def toMFnDep(obj)->om.MFnDependencyNode:
	return om.MFnDependencyNode(toMObject(obj))
def toMFnDag(obj)->om.MFnDagNode:
	return om.MFnDagNode(toMObject(obj))
def toMFnTransform(obj)->om.MFnTransform:
	return om.MFnTransform(toMObject(obj))

def toMFnMesh(obj)->om.MFnMesh:
	return om.MFnMesh(toMObject(obj))
def toMFnCurve(obj)->om.MFnNurbsCurve:
	return om.MFnNurbsCurve(toMObject(obj))
def toMFnSurface(obj)->om.MFnNurbsSurface:
	return om.MFnNurbsSurface(toMObject(obj))

#endregion

def isDag(obj:om.MObject):
	return obj.hasFn(om.MFn.kDagNode)

def isTransform(obj:om.MObject):
	return obj.hasFn(om.MFn.kTransform)

def isShape(obj:om.MObject):
	return getCache().isShape(obj)


class MObjectSet(UserSet):
	"""convenience class providing filtering functions over contents
	"""
	# def uniqueObjs(self)->MObjectSet:
	# 	return MObjectSet(set(self))

	def validObjs(self)->MObjectSet:
		return MObjectSet(filter(lambda x: not x.isNull(), self))

	def filterObjs(self, mfnTypes:T.Iterable[int], toType=None)->(MObjectSet, set[EdNode]):
		"""if toType, given type is mapped across filter results, and
		a normal set of these items is returned"""
		filtered = MObjectSet(filter(lambda x: any(x.hasFn(i) for i in sequence.toSeq(mfnTypes)), self))
		if toType:
			return set(map(toType, filtered))
		return filtered

	# set-like methods



def listMObjects(type=om.MFn.kInvalid)->set[om.MObject]:
	mit = om.MItDependencyNodes(type)
	# objs = MObjectSet()
	objs = set()
	while not mit.isDone():
		objs.add(mit.thisNode())
		mit.next()
	return objs

# def listNamedMObjects()

# endregion


