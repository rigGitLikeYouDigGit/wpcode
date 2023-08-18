
from __future__ import annotations
import typing as T

from wpm import om, WN, core


"""to state the obvious: yes, working with materials in maya is torture
so let's do this properly and then never think about it again
"""

def dgConnectExecIfNotPassed(plugA:om.MPlug, plugB:om.MPlug, dgMod=None):
	"""connect plugA to plugB and execute if an existing modifier object is not passed.

	If it is, add the connect operation but do not execute it
	"""
	if dgMod is not None:
		dgMod.connect(plugA, plugB)
	else:
		dgMod = om.MDGModifier().connect(plugA, plugB).doIt()
	return dgMod

def shadingGroupPlugsFromMaterial(materialNode:om.MObject)->T.List[om.MPlug]:
	"""a single material node may feed multiple shading groups -
	return all of them or an empty list.

	We also assume that only .outColor is taken into account by group
	"""
	return om.MFnDependencyNode(core.toMObject(materialNode)).findPlug(
		"outColor", False
	).destinations()

def assignMaterialToMesh(materialNode:om.MObject, meshNode:om.MObject,
                         dgMod=None)->om.MDGModifier:
	"""assign material to mesh
	"""

	# get shading group
	shadingGroup = shadingGroupPlugsFromMaterial(materialNode)[0].node()
	# get set plug
	setPlug = om.MFnDependencyNode(shadingGroup).findPlug("dagSetMembers", False)
	# last element
	setPlug = setPlug.elementByLogicalIndex(setPlug.numElements())

	# get mesh plug
	meshPlug = om.MFnDependencyNode(meshNode).findPlug("instObjGroups", False)
	# last element
	meshPlug = meshPlug.elementByLogicalIndex(meshPlug.numElements())
	# connect
	return dgConnectExecIfNotPassed( meshPlug, setPlug, dgMod=dgMod)


