
from __future__ import annotations

"""unified file for any module-level caching operations
run when maya loads -
having everything in one place makes it easier to manage when
caching functions run,
also safer to modify those modules without polluting original maya
"""

import inspect, pprint
import typing as T
from dataclasses import dataclass

#from maya.api import OpenMaya as om, OpenMayaAnim as oma
#from maya import cmds
from .patch import cmds, om, oma, omr, omui

@dataclass
class MFnTypeData:
	typeId: int
	name: str
	nearestMfnCls: type


shapeTypeConstantNames = {
	"kMesh", "kMeshGeom", "kNurbsCurve", "kNurbsCurveGeom",
	"kNurbsSurface", "kNurbsSurfaceGeom",
	"kSubdiv", "kSubdivGeom",
	"kLocator"
}

conformGetName = lambda x: (x[-1] + x[:-1] if x[-1].isdigit() else x)

conformNameMap = {
	"Bool"
}

class APICache:
	"""object encapsulating various api-related registers,
	functionset maps"""

	def __init__(self):
		self.mObjRegister = {}
		# MObjects can't be weakref'd for some reason
		# this has to be emptied when file is opened,
		# as different MObjects pick up the same uids

		""" { every api MFn int type constant : nearest matching MFn class object} """
		self.apiTypeMap : dict[int, T.Type[om.MFn]] = {}

		""" { every api MFn int type constant : nearest matching MFn int type constant } """
		self.apiTypeCodeMap : dict[int, int] = {}

		""" { every api MFn int type constant : nice name of that MFn constant}
		eg:
		{ 1002 : 'kKeyframeRegionManip' } """
		self.apiCodeNameMap : dict[int, str] = {}

		self.apiTypeDataMap : dict[int, MFnTypeData] = {}

		# type constants for shape nodes
		self.shapeTypeConstants : set[int] = {getattr(om.MFn, name) for name in shapeTypeConstantNames}

		# map of kName to MFnData class object - kVectorArray to MFnVectorArrayData
		self.mfnDataConstantTypeMap : dict[str, T.Type[om.MFnData]] = {}

		self.dhGetMethods : dict[str, T.Callable] = {}
		self.dhSetMethods : dict[str, T.Callable] = {}
		self.dhNumericTypeGetMethodMap : dict[int, T.Callable] = {}
		self.dhNumericTypeSetMethodMap : dict[int, T.Callable] = {}
		self.unitAttrPlugMethodMap = {
			om.MFnUnitAttribute.kDistance: (om.MPlug.asDouble, om.MPlug.setDouble),
			om.MFnUnitAttribute.kTime: (om.MPlug.asMTime, om.MPlug.setMTime),
			om.MFnUnitAttribute.kAngle: (om.MPlug.asMAngle, om.MPlug.setMAngle),
		}

		self.numericAttrPlugMethodMap = {
			("kLong", "kShort", "kInt", "kBoolean"): (om.MPlug.asInt, om.MPlug.setInt),
			("kFloat", "kDouble"): (om.MPlug.asDouble, om.MPlug.setDouble)
		}

	@staticmethod
	def MObjectHash(obj)->int:
		"""return maya-compatible hash for given mobject - could supercede
		the ls uid method"""
		return om.MObjectHandle(obj).hashCode()

	def getMObject(self, node)->om.MObject:
		"""this is specialised for dg nodes -
		component MObjects will have their own functions anyway if needed
		"""
		if isinstance(node, om.MObject):
			if node.isNull():
				raise RuntimeError("object for ", node, " is invalid")
			return node
		else:
			try:
				return node.object() # supports MFnBase and EdNode
			except:
				pass

		uid = cmds.ls(node, uuid=1)[0]

		if self.mObjRegister.get(uid):
			obj = self.mObjRegister[uid]
			if not obj.isNull():
				return obj
		sel = om.MSelectionList()
		sel.add(node)
		obj = sel.getDependNode(0)
		if obj.isNull():
			raise RuntimeError("object for ", node, " is invalid")

		self.mObjRegister[uid] = obj
		return obj

	def gatherClassConstantMaps(self, gatherCls):
		"""gather static constants like "kDouble" from classes
		if no lemma is provided, reverts to checking for "k" at start of attrs
		returns { constant value : name }
		"""
		valueNameMap = {}
		for name, value in inspect.getmembers(gatherCls):
			if not name.startswith("k"):
				continue
			valueNameMap[value] = name
		return valueNameMap

	# mFnCollectMap = {
	# 	om.MFnTransform : (om.MFn.kTransform, om.MFn.kJoint, om.MFn.kDagNode),
	# 	om.MFnMesh : (om.MFn.kMesh, om.MFn.kMeshGeom),
	# 	om.MFnNurbsSurface : (om.MFn.kNurbsSurface, om.MFn.kNurbsSurfaceGeom,),
	# 	om.MFnNurbsCurve : (om.MFn.kNurbsCurve, om.MFn.kNurbsCurveGeom,
	# 	                    om.MFn.kBezierCurve),
	# 	om.MFnCamera : (om.MFn.kCamera,)
	#
	# }

	#print("api begin typemap")
	def generateApiTypeMap(self):
		"""returns 3 maya constant maps -

		apiTypeMap:
		{ every api MFn int type constant : nearest matching MFn class object}

		apiTypeCodeMap :
		{ every api MFn int type constant : nearest matching MFn class int type constant }

		apiCodeNameMap :
		{ every api MFn int type constant : nice name of that MFn constant}
		eg:
		{ 1002 : 'kKeyframeRegionManip' }

		"""
		# get all constants from om.MFn
		# [ ("kFloat", 2), ("kDouble", 3), ... ]
		constants = [i for i in inspect.getmembers(om.MFn) if i[0].startswith("k")]

		constants = sorted(constants, key=lambda x:x[1])

		# map of int constants to available python classes
		mfnCodeClassMap = {}
		codeNameMap = {}

		for k, v in constants:
			codeNameMap[v] = k

		abstractClasses = (om.MFn, om.MFnBase)
		# extract all MFn classes and index them by their type constant
		for k, v in (inspect.getmembers(om) + inspect.getmembers(oma)):
			#print(k, v)

			if not isinstance(v, type):
				continue
			if not issubclass(v, om.MFnBase):
				continue
			# if not k.startswith("MFn"):
			# 	continue

			if v in abstractClasses:
				continue
			classId = v().type()
			mfnCodeClassMap[classId] = v

		# naively, we might assume that earlier class constants are more likely to be wide

		# build map of hasType() compatibility for available classes
		# this will likely be slow
		compatPools = []
		compatMap = {}

		# map of { mfn class constant : number of supported constants }
		mfnWidthMap = {mfnCode : 0 for mfnCode in mfnCodeClassMap.keys()}

		# {constant : list of compatible mfn classes}
		constantToClassCompatMap = {code : [] for name, code in constants}
		
		for mfnCode, cls in mfnCodeClassMap.items():
			if cls == om.MFnBase:
				continue
			pool = set()
			for name, code in constants:
				if not name.startswith("k"):
					continue
				# we need to instantiate an MFn here for hasObj(), bit weird
				# also no way to just get all compatible types for a given MFn
				if cls().hasObj(code):
					#clsData["compat"].add(code)
					pool.add(code)
					constantToClassCompatMap[code].append(mfnCode)
					mfnWidthMap[mfnCode] += 1

			compatMap[mfnCode] = pool
			#mfnWidthMap[mfnCode] = len(pool)
			compatPools.append((mfnCode, pool))

		sortedMFnWidths = [i[0] for i in sorted(mfnWidthMap.items(), key=lambda x: x[1], reverse=True)] # list of class constants sorted by number of compatible constants

		"""for now remove them - these are things like kBirailSrf,
		kManip, kFilter - internals, edge cases, contexts, etc
		"""
		constantToClassCompatMap = {k : v for k, v in constantToClassCompatMap.items() if v}


		constantToTargetClassMap = {code : None for code in constantToClassCompatMap.keys()}
		constantToTargetConstantMap = {code : None for code in constantToClassCompatMap.keys()}
		for constant, clsCodes in constantToClassCompatMap.items():
			constantToTargetClassMap[constant] = \
				mfnCodeClassMap[sorted(clsCodes, key=lambda x: mfnWidthMap[x])[0]]
			constantToTargetConstantMap[constant] = sorted(clsCodes, key=lambda x: mfnWidthMap[x])[0]

		return constantToTargetClassMap, constantToTargetConstantMap, codeNameMap

	#apiTypeMap, apiTypeCodeMap, apiCodeNameMap = generateApiTypeMap()


	def nodeTypeFromMObject(self, mobj:om.MObject)->str:
		"""return a nodeTypeName string that can be passed to cmds.createNode
		"""
		name = self.apiCodeNameMap[mobj.apiType()]
		# name = mobj.apiTypeStr
		return name[1].lower() + name[2:]


	# function presenting the above as dict of dataclasses
	def buildApiTypeDataMap(self, apiTypeMap, apiCodeNameMap):
		typeDataMap = {}
		for typeConstant, cls in apiTypeMap.items():
			typeDataMap[typeConstant] = MFnTypeData(
				typeConstant,
				apiCodeNameMap[typeConstant],
				cls
			)
		return typeDataMap

	# apiTypeDataMap = buildApiTypeDataMap(apiTypeMap, apiTypeCodeMap, apiCodeNameMap)

	def buildMFnDataMap(self):
		"""build a map of MFn constant kName to corresponding
		MFnData class"""
		"""look up MFnVectorArrayData from kVectorArray"""
		valueNameMap = self.gatherClassConstantMaps(om.MFnData)
		valueTypeMap = {}
		for value, name in valueNameMap.items():
			"""look up MFnVectorArrayData from kVectorArray"""
			clsName = "MFn{}Data".format(name[1:])
			lookupCls = getattr(om, clsName, None)
			if lookupCls is None:
				continue
			valueTypeMap[value] = lookupCls
		return valueTypeMap

	# mfnDataConstantTypeMap = buildMFnDataMap()

	#pprint.pprint(apiTypeMap)
	# coercing input to MObject in functions below is not most futureproof,
	# but worth it for the lines saved
	def getMFnType(self, obj:om.MObject)->T.Type[om.MFnBase]:
		"""returns the highest available MFn
		for given object, based on sequence order
		above"""
		if isinstance(obj, int):
			return self.apiTypeMap[obj]
		obj = self.getMObject(obj)
		# print("getMFnType", obj.apiType(),
		# 	  apiCodeNameMap[obj.apiType()], apiTypeMap[obj.apiType()])
		return self.apiTypeMap[obj.apiType()]

	def getMFn(self, obj:om.MObject)->om.MFnBase:
		"""return mfn function set initialised on the given object"""
		obj = self.getMObject(obj)
		return self.getMFnType(obj)(obj)

	def isShape(self, obj: om.MObject):
		return obj.apiType() in (self.shapeTypeConstants)

	def buildDataHandleGetSetNameMaps(self)->T.Tuple[
		T.Dict[str, callable], T.Dict[str, callable]]:
		"""MDatahandle methods :
		- asDouble3()
		- set3Double()

		- asAngle()
		- setMAngle()
		fucking end me

		3Double is consistent with MFnNumericData, so that's the form we'll use
		"""
		dhMembers = inspect.getmembers(om.MDataHandle)
		# maps of { "Double3" : <method asDouble3> } etc
		getMethods = {}
		setMethods = {}
		for name, method in dhMembers:
			if "Clean" in name:
				continue
			if "Generic" in name:
				continue
			if name.startswith("as"):
				getMethods[conformGetName(name[2:])] = method
			elif name.startswith("set"):
				setMethods[name[3:]] = method

		setMethods["Angle"] = setMethods["MAngle"]
		# print("get", getMethods.keys())
		# print("set", setMethods.keys())

		return getMethods, setMethods

	def buildDataHandleNumericMethodMap(self, dhGetMethods, dhSetMethods)->tuple[
			dict[int, callable], dict[int, callable]]:
		"""return { numeric data id : get method}
		and same for set method"""
		baseMembers = inspect.getmembers(om.MFnData)
		numericMembers = inspect.getmembers(om.MFnNumericData)

		typeConstants = {}
		for name, value in numericMembers:
			if not name.startswith("k"):
				continue
			typeConstants[name[1:]] = int(value)
		typeConstants["Bool"] = typeConstants["Boolean"]  # f f s
		# print("base constants", typeConstants)
		for name, value in baseMembers:
			if name.startswith("k"):
				name = name[1:]
			# print("remove", name, name in typeConstants)
			if name in typeConstants:
				typeConstants.pop(name)

		typeConstantNameMap = {v: k for k, v in typeConstants.items()}

		typeGetMap, typeSetMap = {}, {}
		for typeMap, methodMap in ((typeGetMap, dhGetMethods), (typeSetMap, dhSetMethods)):
			for name, method in methodMap.items():
				numericTypeId = typeConstants.get(name)
				if numericTypeId is None:
					continue
				typeMap[int(numericTypeId)] = method
		return typeGetMap, typeSetMap

	def _expandClassLookup(self, baseDict, lookupCls):
		for k, v in dict(baseDict).items():
			for attrName in k:
				value = getattr(lookupCls, attrName)
				baseDict[value] = v
		return baseDict

	# numericAttrPlugMethodMap = _expandClassLookup(
	# 	numericAttrPlugMethodMap, om.MFnNumericData)

	def _build(self):
		"""main initialisation function for manager - rerun on scene open?
		rerun on reloading edrig
		"""
		self.reset()
		self.apiTypeMap, self.apiTypeCodeMap, self.apiCodeNameMap = self.generateApiTypeMap()
		self.apiTypeDataMap = self.buildApiTypeDataMap(
			self.apiTypeMap, self.apiCodeNameMap)
		self.mfnDataConstantTypeMap = self.buildMFnDataMap()

		self.dhGetMethods, self.dhSetMethods = self.buildDataHandleGetSetNameMaps()
		self.dhNumericTypeGetMethodMap, self.dhNumericTypeSetMethodMap =\
			self.buildDataHandleNumericMethodMap(
				self.dhGetMethods, self.dhSetMethods)

		self.numericAttrPlugMethodMap = self._expandClassLookup(
			self.numericAttrPlugMethodMap, om.MFnNumericData)

	def reset(self):
		"""clear any live information from this manager
		at present that is only the MObject map, the rest is just
		static mapping through the api itself"""
		self.mObjRegister.clear()


class _APIGlobals:
	"""probably unnecessary - avoids the main APIManager reference being
	module-level, avoids dealing with 'global' keyword issues"""
	apiCache : APICache = None

	@classmethod
	def refresh(cls):
		cls.apiCache = APICache()
		cls.apiCache._build()

def getCache()->APICache:
	if _APIGlobals.apiCache is None:
		_APIGlobals.refresh()
	return _APIGlobals.apiCache

#if hostDict[constant.Program.Maya]:
#_APIGlobals.refresh()

setFnMap = {
	str : "setString",
	int : "setInt",
	float : "setDouble",
	bool : "setBool",
}
getFnMap = {
	str : "asString",
	int : "asInt",
	float : "asDouble",
	bool : "asBool"
}




# unitAttrPlugMethodMap = {
# 	om.MFnUnitAttribute.kDistance : (om.MPlug.asDouble, om.MPlug.setDouble),
# 	om.MFnUnitAttribute.kTime : (om.MPlug.asMTime, om.MPlug.setMTime),
# 	om.MFnUnitAttribute.kAngle : (om.MPlug.asMAngle, om.MPlug.setMAngle),
# }
# numericAttrPlugMethodMap = {
# 	("kLong", "kShort", "kInt", "kBoolean"): (om.MPlug.asInt, om.MPlug.setInt),
# 	("kFloat", "kDouble") : (om.MPlug.asDouble, om.MPlug.setDouble)
# }

# def _expandClassLookup(baseDict, lookupCls):
# 	for k, v in dict(baseDict).items():
# 		for attrName in k:
# 			value = getattr(lookupCls, attrName)
# 			baseDict[value] = v
# 	return baseDict
# numericAttrPlugMethodMap = _expandClassLookup(
# 	numericAttrPlugMethodMap, om.MFnNumericData)



