from __future__ import annotations
import types, typing as T
import pprint
from collections import defaultdict

from wplib import log
import numpy as np
import hou

from wplib.maths import shape

attrDataTypeStrings = ("int", "float", "string", "dict")

def geoAttrsToDict(
		geo:hou.Geometry,
		elTypes:tuple[str]=("point", "prim", "detail", "vertex"),
		letList:tuple[str]=None,
		notList:tuple[str]=()
                   )->dict[
	str, dict[str, dict[str, np.ndarray]]]:
	"""return dict of
	{"point" : {"myAttr" : { "v" : array vals, "info" : metadata}
	"""
	# gather geo attributes
	result = defaultdict(dict)
	for elType in ["point", "prim", "detail", "vertex"]:
		if not elType in elTypes:
			continue
		if elType == "detail":
			attrs = geo.globalAttribs()
			for attr in attrs:
				if letList is not None and not attr.name() in letList:
					continue
				if notList and attr.name() in notList:
					continue
				dataType = str(attr.dataType())
				lToken = "List" if attr.isArrayType() else ""
				lookup = f"{dataType.title()}{lToken}AttribValue"
				vals = getattr(geo, lookup)(attr)
				if attr.isArrayType():
					indices, values = shape.indexValueArrsFromTupleList(vals)
					data = {"v" : values, "i" : indices}
				else:
					data = {"v" : np.array(vals)}
				data["info"] = {
					"size": attr.size(),
					"dataType": str(attr.dataType()),
				}
				result[elType][attr.name()] = data
			continue

		attribs = getattr(geo, f"{elType}Attribs")()
		for attr in attribs:
			attrName = attr.name()
			if letList is not None and not attrName in letList:
				continue
			if notList and attrName in notList:
				continue
			dataType = str(attr.dataType())

			if attr.isArrayType():
				lookup = f"{elType}{dataType.title()}ListAttribValues"
				vals = getattr(geo, lookup)(attr)
				indices, values = shape.indexValueArrsFromTupleList(vals)
				data = {"v" : values, "i" : indices}
			else:
				lookup = f"{elType}{dataType.title()}AttribValues"
				vals = getattr(geo, lookup)(attr)
				data = {"v": np.array(vals)}

			data["info"] = {
				"size": attr.size(),
				"dataType": str(attr.dataType()),
			}
			result[elType][attrName] = data
	return result

def geoGroupsToDict(geo:hou.Geometry,
                    elTypes:tuple[str]=("point", "prim", "vertex", "edge"),
                    letList:tuple[str]=None,
					notList:tuple[str]=())->dict[
		str, dict[str, np.ndarray]]:
	"""return dict of
	{"point" : {"groupName" : array of indices}
	"""
	result = defaultdict(dict)
	for elType in elTypes:
		lookup = f"{elType}Groups"
		groups = getattr(geo, lookup)()
		for group in groups:
			if elType == "edge":
				pass
			group : hou.PointGroup | hou.PrimGroup
			indices = np.empty(group.count(), dtype=int)
			for i, el in enumerate(group.elements()):
				indices[i] = el.number()

	# point groups
	pointGroups = geo.pointGroups()
	nPoints = len(geo.points())
	for group in pointGroups:
		name = group.name()
		membership = np.zeros(nPoints, dtype=bool)
		for pt in group.points():
			membership[pt.number()] = True
		result["point"][name] = membership

	# primitive groups
	primGroups = geo.primGroups()
	nPrims = len(geo.prims())
	for group in primGroups:
		name = group.name()
		membership = np.zeros(nPrims, dtype=bool)
		for prim in group.prims():
			membership[prim.number()] = True
		result["prim"][name] = membership

	return result

def geoTopologyToDict(geo:hou.Geometry,
                      savePrims:bool=True,
                      savePoints:bool=False):
	"""return dict of

	"""
	result = {}
	if savePoints:
		# point to prim topology
		nPointPrimIndices = 0
		pointPrimCounts = np.empty(geo.pointCount(), dtype=int)
		pointPrimStartEnd = np.empty(geo.pointCount() + 1, dtype=int)
		pointPrimStartEnd[0] = 0
		maxPrimsPerPoint = 0
		for i, pt in enumerate(geo.points()):
			prims = pt.prims()
			nPointPrimIndices += len(prims)
			pointPrimCounts[i] = len(prims)
			pointPrimStartEnd[i + 1] = nPointPrimIndices
		pointPrimIndicesArr = np.empty(nPointPrimIndices, dtype=int)
		points = geo.points()
		for i, prims in enumerate(pointPrimStartEnd[:-1]):
			pointPrimIndicesArr[
				pointPrimStartEnd[i]:pointPrimStartEnd[i + 1]
			] = points[i].primIndices()

		result["points"] = {
			"primIndices": pointPrimIndicesArr,
			"primCounts": pointPrimCounts,
			"primStartEnd": pointPrimStartEnd
		}
	if savePrims:
		# primitive to point topology
		nPrimPointIndices = 0
		primPointCounts = np.empty(geo.primCount(), dtype=int)
		primPointStartEnd = np.empty(geo.primCount() + 1, dtype=int)
		primPointStartEnd[0] = 0
		for i, prim in enumerate(geo.prims()):
			points = prim.points()
			nPrimPointIndices += len(points)
			primPointCounts[i] = len(points)
			primPointStartEnd[i + 1] = nPrimPointIndices
		primPointIndicesArr = np.empty(nPrimPointIndices, dtype=int)
		prims = geo.prims()
		for i, points in enumerate(primPointStartEnd[:-1]):
			primPointIndicesArr[
				primPointStartEnd[i]:primPointStartEnd[i + 1]
			] = prims[i].pointIndices()

		result["prims"] = {
			"pointIndices": primPointIndicesArr,
			"pointCounts": primPointCounts,
			"pointStartEnd": primPointStartEnd
		}

	return result

def geoToDict(geo:hou.Geometry,
              attrElTypes:tuple[str]=("point", "prim", "detail", "vertex"),
              attrLetList:tuple[str]=None,
              attrNotList:tuple[str]=(),
			  groupElTypes:tuple[str]=("point", "prim", "vertex", "edge"),
              groupLetList:tuple[str]=None,
			  groupNotList:tuple[str]=(),
              savePrimTopology:bool=True,
			  savePointTopology:bool=False
			  )->dict[str, T.Any]:
	return {
		"attributes": geoAttrsToDict(
			geo, elTypes=attrElTypes, letList=attrLetList, notList=attrNotList),
		"groups": geoGroupsToDict(
			geo, elTypes=groupElTypes, letList=groupLetList, notList=groupNotList),
		"topology": geoTopologyToDict(
			geo, savePrims=savePrimTopology, savePoints=savePointTopology),
	}


