
from __future__ import annotations
import typing as T

"""unified file for any module-level caching operations
run when maya loads -
having everything in one place makes it easier to manage when
caching functions run,
also safer to modify those modules without polluting original maya
"""

from collections import defaultdict, namedtuple
import inspect, pprint
from dataclasses import dataclass
import threading # cache api stuff as threaded jobs to avoid slow startup

#from maya.api import OpenMaya as om, OpenMayaAnim as oma
#from maya import cmds
from wplib import log
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

# for all api classes, allow looking up and retrieving the "kConstant" name,
# as well as actual values
def getApiConstantNameMap(cls)->dict[int, str]:
	"""return a dict of { constant value : name } for all constants in the class"""
	valueNameMap = {}
	for name, value in inspect.getmembers(cls):
		if not name.startswith("k"):
			continue
		if not isinstance(value, int):
			continue
		valueNameMap[value] = name
	return valueNameMap


if T.TYPE_CHECKING:
	MFnT = T.Type[om.MFn, om.MFnBase]
class APICache:
	"""
	cache useful relationships for MFn classes and Maya's
	object type system
	"""

	def __init__(self):
		self.mObjRegister = {}
		# MObjects can't be weakref'd for some reason
		# this has to be emptied when file is opened,
		# as different MObjects pick up the same uids

		""" { MFn class apiType : MFn class }
		ALL MFn classes in maya"""
		self.apiTypeMFnMap : dict[int, MFnT] = {}

		self.classConstantNameMaps : dict[MFnT, dict[int, str]] = {}
		self.classNameConstantMaps : dict[MFnT, dict[str, int]] = {}

		self.apiTypeLeafMFnMap : dict[int, MFnT] = {} # specific to MFnBase
		self.apiStrLeafMFnMap: dict[str, MFnT] = {}

		self.apiTypeMFnDataMap : dict[int, type[om.MFnData]] = {}

	def classTypeIdNameMemberMap(self, MFnCls:MFnT)->dict[int, str]:
		if MFnCls not in self.classConstantNameMaps:
			typeNameMap, nameTypeMap = self.gatherClassConstantMaps(MFnCls)
			self.classConstantNameMaps[MFnCls] = typeNameMap
			self.classNameConstantMaps[MFnCls] = nameTypeMap
			#print("built", MFnCls, typeNameMap, nameTypeMap)
		return self.classConstantNameMaps[MFnCls]

	def classNameTypeIdMemberMap(self, MFnCls:MFnT)->dict[str, int]:
		if MFnCls not in self.classNameConstantMaps:
			typeNameMap, nameTypeMap = self.gatherClassConstantMaps(MFnCls)
			self.classConstantNameMaps[MFnCls] = typeNameMap
			self.classNameConstantMaps[MFnCls] = nameTypeMap
		return self.classNameConstantMaps[MFnCls]

	def buildCache(self):
		"""build everything at startup - see if there's a way
		to thread this"""
		# gather all MFn classes to work with
		self.apiTypeMFnMap = self.gatherMFnClasses()
		# log("apiTypeMFnMap")
		# pprint.pp(self.apiTypeMFnMap) # works

		# run the big name maps
		mfnMap = self.classTypeIdNameMemberMap(om.MFn)

		nameMap = self.classNameTypeIdMemberMap(om.MFn)
		# mfnBaseMap = self.classTypeIdNameMemberMap(om.MFnBase)
		# pprint.pp(mfnMap)
		#
		# pprint.pp(mfnBaseMap)
		#pprint.pp(self.classTypeIdNameMemberMap(om.MFnNumericData))

		typeMFnListMap, mfnTypeListMap, typeLeafMFnMap = self.getCompatibilityMaps(
			self.apiTypeMFnMap, nameMap)
		self.apiTypeLeafMFnMap = typeLeafMFnMap
		self.apiStrLeafMFnMap = {
			self.classTypeIdNameMemberMap(om.MFn)[k] : v
			for k, v in typeLeafMFnMap.items()}
		#pprint.pp(self.apiStrLeafMFnMap)


		"""there is apparently no hard link by type ids from 
		MFnData.kConstant to the appropriate MFnData subclass.
		 so we make our own."""
		apiTypeMFnDataMap = {}
		#print("SUBCLASSES", om.MFnData.__subclasses__())
		mfnDataMembers = self.classTypeIdNameMemberMap(om.MFnData)
		#print("MEMBERS", mfnDataMembers)
		for subCls in om.MFnData.__subclasses__():
			#print(subCls.__name__)
			for typeId, name in mfnDataMembers.items():
				# this is quite villainous
				if name[1:] in subCls.__name__:
					apiTypeMFnDataMap[typeId] = subCls
		#pprint.pp(apiTypeMFnDataMap)
		self.apiTypeMFnDataMap = apiTypeMFnDataMap




	def gatherMFnClasses(self) -> dict[int, type[om.MFn, om.MFnBase]]:
		"""return { apiType : MFnClass } for all MFn classes
		"""
		abstractClasses = (om.MFn, om.MFnBase)
		mfnCodeClassMap = {-2 : om.MFn,
		                   -1 : om.MFnBase}
		# extract all MFn classes and index them by their type constant
		for k, v in (
				inspect.getmembers(om)
				+ inspect.getmembers(oma)
				+ inspect.getmembers(omui)
				+ inspect.getmembers(omr)
		):
			if not isinstance(v, type):
				continue
			if not issubclass(v, om.MFnBase):
				continue
			if v in abstractClasses: # raise error if you try to instantiate
				continue
			classId = v().type()
			mfnCodeClassMap[classId] = v
		return {k : mfnCodeClassMap[k] for k in sorted(mfnCodeClassMap.keys())}

	def gatherClassConstantMaps(self, gatherCls)->tuple[dict[int, str], dict[str, int]]:
		"""gather static constants like "kDouble" from classes
		if no lemma is provided, reverts to checking for "k" at start of attrs
		returns { constant value : name }

		INHERITED class constants can REUSE TYPE IDs in the PARENT :)
		"""
		valueNameMap = {}
		nameValueMap = {}
		members = inspect.getmembers(gatherCls)
		#print("gatherCls", gatherCls, gatherCls.__bases__)
		if gatherCls.__bases__:
			members = set(members) - set(inspect.getmembers(gatherCls.__bases__[0]))
		for name, value in sorted(
				members,
				key=lambda x: x[1] if isinstance(x[1], int) else 0
		):
			#print("check ", name, value)
			if not name.startswith("k"):
				continue
			valueNameMap[value] = name
			nameValueMap[name] = value
		return valueNameMap, nameValueMap


	def getCompatibilityMaps(self,
	                         mfnCodeClassMap:dict[int, MFnT],
	                         constants:dict[str, int]):
		"""build map of hasType() compatibility for available classes.
		Don't think there's a way to avoid NxM complexity,
		checking every constant against
		every class

		maybe there's a more structured way of passing this around,
		for now return 3 results i guess
		"""
		# first pass: get all MFns compatible with given constant
		typeMFnListMap : dict[int : list[MFnT]] = defaultdict(list)
		mfnTypeListMap : dict[MFnT, list[int]] = defaultdict(list)
		for mfnCode, cls in mfnCodeClassMap.items():
			if mfnCode < 0: # abstract class
				continue
			if cls == om.MFnBase:
				continue
			for name, code in constants.items():
				if not name.startswith("k"):
					continue
				if cls().hasObj(code):
					typeMFnListMap[code].append(cls)
					mfnTypeListMap[cls].append(code)

		for cls, typeList in mfnTypeListMap.items():
			typeList.sort(
				key=lambda x: len(typeMFnListMap[x]))

		for typeConstant, clsList in typeMFnListMap.items():
			clsList.sort(
				key=lambda x: len(mfnTypeListMap[x]), reverse=True)

		# get lowest leaf MFn classes to prefer lookup
		typeLeafMFnMap = {k : v[-1] for k, v in sorted(
			typeMFnListMap.items(), key=lambda x: x[0])}

		return typeMFnListMap, mfnTypeListMap, typeLeafMFnMap



	@staticmethod
	def MObjectHash(obj)->int:
		"""return maya-compatible hash for given mobject - could supercede
		the ls uid method"""
		return om.MObjectHandle(obj).hashCode()

	def getMObjectCached(self, node)->om.MObject:
		"""this is specialised for dg nodes -
		component MObjects will have their own functions anyway if needed
		TODO: rework this good grief
			in general getting persistent MObjects is dodgy, the only time
			we want to do it is in the WN system - maybe we just do a normal
			getMObject function like every other sane person
		"""
		if isinstance(node, om.MObject):
			if node.isNull():
				raise RuntimeError("object for ", node, " is invalid")
			return node
		else:
			try:
				return node.object() # supports MFnBase and WN
			except:
				pass

		sel = om.MSelectionList()
		sel.add(node)
		obj = sel.getDependNode(0)
		if obj.isNull():
			raise RuntimeError("object for ", node, " is invalid")

		return obj



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



	def reset(self):
		"""clear any live information from this manager
		there's nothing here ._.
		"""



class _APIGlobals:
	"""probably unnecessary - avoids the main APIManager reference being
	module-level, avoids dealing with 'global' keyword issues"""
	apiCache : APICache = None

	@classmethod
	def refresh(cls):
		cls.apiCache = APICache()
		cls.apiCache.buildCache()

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



