

"""core functions for MPlugs"""

from __future__ import annotations

import typing as T

from enum import Enum

import copy, fnmatch, re

from dataclasses import dataclass

from wplib import log
from wplib.string import sliceFromString
from wplib.sequence import flatten, isSeq
from wplib.object import TypeNamespace
from wptree import Tree

from .bases import PlugBase
from .api import toMFn, mfnDataConstantTypeMap
from .cache import getCache, om


if T.TYPE_CHECKING:
	pass


def getMPlug(plug)->om.MPlug:
	if isinstance(plug, om.MPlug):
		return plug
	if isinstance(plug, PlugBase):
		return plug.MPlug
	if not isinstance(plug, str):
		raise TypeError(f"cannot retrieve MPlug from argument {plug} of type {type(plug)}")
	try:
		sel = om.MSelectionList()
		sel.add(plug)
		return sel.getPlug(0)
	except RuntimeError:
		raise NameError(f"cannot retrieve MPlug from string {plug}")

# do we need a way to iterate over nested structure to convert to MPlugs?


# replicate something like numpy's slicing, but for connecting and setting plugs



# region querying and structure

class HType(Enum):
	Leaf = 1
	Compound = 2
	Array = 3

def plugHType(plug:om.MPlug)->HType:
	"""compound takes priority over array, so check array first"""
	if plug.isArray:
		return HType.Array
	if plug.isCompound:
		return HType.Compound
	return HType.Leaf

def plugSubPlugs(plug:om.MPlug):
	if plug.isArray:
		return [plug.elementByPhysicalIndex(i)
		        for i in range(plug.numElements())]
	if plug.isCompound:
		return [plug.child(i) for i in range(plug.numChildren())]
	return []

def subPlugMap(plug)->dict[(int, str), om.MPlug]:
	"""map of all child / element plugs available here """
	subPlugs = plugSubPlugs(plug)
	if plug.isArray:
		return {i : val for i, val in enumerate(subPlugs)}
	if plug.isCompound:
		result = {i.name().split(".")[-1] : i for i in subPlugs}
		return result
	return {}

def ensureOneArrayElement(plug:om.MPlug):
	"""ensure at least one element in array
	the logic behind this is super sketchy -
	maya plug arrays are sparse, with the "logical" index being the "nice" index,
	and the "physical" index being the real dense index of the value
	(normally hidden from the user)

	accessing an empty array or adding new entry
	with physical values doesn't work, since
	no physical entries exist, so the first access has to be logical

	after that, we will probably still use the logical index to
	maintain parity with whatever is displayed in maya's ui

	some "simple" numeric attributes don't accept setting with MObject

	"""
	if not plug.numElements():
		elPlug = plug.elementByLogicalIndex(0)
		try:
			elPlug.setMObject(elPlug.asMObject())
		except:
			pass

def newLastArrayElement(plug:om.MPlug):
	"""returns a new open plug at the end of the array"""
	assert plug.isArray, "invalid plug {} is not array".format( (plug, type(plug)) )
	n = plug.evaluateNumElements()
	ensureOneArrayElement(plug)
	return plug.elementByLogicalIndex(n)

def arrayElement(plug:om.MPlug, index:int)->om.MPlug:
	"""final atomic function for retrieving plug by index
	process negative indices before this"""
	nElements = plug.evaluateNumElements()
	if index < 0:
		index = max(nElements, 1) + index
	# print("el index", index,  "nElements", nElements)
	elPlug = plug.elementByLogicalIndex(index)
	if index > (nElements - 1):
		try:
			elPlug.setMObject(elPlug.asMObject())
		except:
			pass
	return elPlug

def parentPlug(childPlug:om.MPlug)->om.MPlug:
	"""return the topological parent plug for the given plug,
	or None -
	small convenience to unify syntax between array and compound plugs"""
	if childPlug.isElement:
		return childPlug.array()
	elif childPlug.isChild:
		return childPlug.parent()
	return None

#endregion

dimTypes = {
	"0D" : ("matrix",),
	"1D" : ("nurbsCurve", "bezierCurve"),
	"2D" : ("nurbsSurface", "mesh")
}

nodeObjMap = {}

# type-anonymous ways to get and set plug values
# it's possible this could be faster using datahandles, but they crash
# so sticking to MFnData objects for now

# build method map for MDataHandles and numericData constants

# conformGetName = lambda x: (x[-1] + x[:-1] if x[-1].isdigit() else x)
#
# conformNameMap = {
# 	"Bool"
# }

#dhGetMethods, dhSetMethods = buildDataHandleGetSetNameMaps()
# print("get keys", sorted(dhGetMethods.keys()))
# print("set keys", sorted(dhSetMethods.keys()))


# dhNumericTypeGetMethodMap, dhNumericTypeSetMethodMap = buildDataHandleNumericMethodMap(
#	dhGetMethods, dhSetMethods)

# brute force typing, rip




def plugMFnAttr(plug:om.MPlug):
	"""return MFnAttribute for this plug's Attribute object"""
	attribute = plug.attribute()
	attrFn = toMFn(attribute)
	return attrFn

def plugMFnDataType(plug:om.MPlug):
	"""return MFn<Type>Data clss for the type of this plug
	if plug is not a typed attribute, return None"""
	mfnAttr = plugMFnAttr(plug)
	if not isinstance(mfnAttr, om.MFnTypedAttribute):
		return None
	return mfnDataConstantTypeMap()[mfnAttr.attrType()]

def ensurePlugHasMObject(plug:om.MPlug):
	mfnDataCls = plugMFnDataType(plug)
	if mfnDataCls is None:
		return
	mfnData = mfnDataCls()
	try:
		return plug.asMObject()
	except:
		pass
	# create new object with data class
	newObj = mfnData.create()
	plug.setMObject(newObj)
	return newObj


#testing separate functions for different iteration systems -
# more code, but simpler funtions and more explicit in calling
def iterPlugsTopDown(rootPlug:om.MPlug):
	"""we know the drill by now, breadth/depth-first iteration over
	existing plug indices"""
	yield rootPlug
	for subPlug in plugSubPlugs(rootPlug):
		yield from iterPlugsTopDown(subPlug)

def iterPlugsBottomUp(rootPlug:om.MPlug):
	for subPlug in plugSubPlugs(rootPlug):
		yield from  iterPlugsBottomUp(subPlug)
	yield rootPlug


def _plugOrConstant(testPlug):
	try:
		return getMPlug(testPlug)
	except (NameError, TypeError):
		return testPlug


def iterLeafPlugs(rootPlug:om.MPlug):
	"""return all leaf plugs for the given plug"""
	subPlugs = plugSubPlugs(rootPlug)
	if subPlugs:
		for i in subPlugs:
			yield from iterLeafPlugs(i)
	else:
		yield rootPlug

def nestedLeafPlugs(rootPlug:om.MPlug):
	"""return nested leaf plugs for the given plug"""
	subPlugs = plugSubPlugs(rootPlug)
	return [nestedLeafPlugs(i) for i in subPlugs] if subPlugs else [rootPlug]

def _expandPlugSeq(seq):
	result = []
	for i in seq:
		val = _plugOrConstant(i)
		if isinstance(val, om.MPlug):
			val = nestedLeafPlugs(val)
			result.extend(val)
		else:
			result.append(val)
	return result

def plugTreePairs(a:(om.MPlug, tuple), b:(om.MPlug, tuple)):
	"""
	yield final pairs of leaf plugs or values
	for connection or setting
	try to get to nested tuples

	check for plug types for shortcuts - float3, vector plugs etc

	still need to check for array plugs

	maybe we can use tuples inside argument to denote units that should
	not be expanded

	"""
	# convert to sequences
	if not isSeq(a):
		a = (a,)
	if not isSeq(b):
		b = (b,)

	# check if can be made plugs
	#print("base", a, b)
	a = _expandPlugSeq(a)
	b = _expandPlugSeq(b)

	#log("plugTreePairs", a, b)
	if len(a) == 1:
		if len(b) == 1:
			yield (a[0], b[0])
			return
		else:
			for i in b:
				yield from (plugTreePairs(a, i))
			return
	if len(a) == len(b):
		for i, j in zip(a, b):
			yield from plugTreePairs(i, j)
		return

	raise ValueError("plugTreePairs: mismatched plug sequences", a, b)



#region getting

def _dataFromPlugMObject(obj:om.MObject):
	"""retrieve data from direct plug MObject"""
	dataFn = toMFn(obj)
	try:
		return list(dataFn.array())
	except:
		pass
	if isinstance(dataFn, om.MFnMatrixData):
		return dataFn.matrix()
	elif isinstance(dataFn, om.MFnStringData):
		return dataFn.string()
	return dataFn.getData()

def leafPlugValue(plug:om.MPlug,
                  asDegrees=True):

	# populate plug if needed
	ensurePlugHasMObject(plug)

	# easy out, check if plug has a direct data MObject
	try:
		obj = plug.asMObject()
		data = _dataFromPlugMObject(obj)
		return data
	except RuntimeError:  # "unexpected internal failure"
		pass

	# we now need to check the attribute type of the plug
	attribute = plug.attribute()
	attrFn = toMFn(attribute)
	attrFnType = type(attrFn)
	data = None
	if attrFnType is om.MFnUnitAttribute:
		data = getCache().unitAttrPlugMethodMap[attrFn.unitType()][0](plug)
		if attrFn.unitType() == om.MFnUnitAttribute.kAngle and asDegrees:
			data = data.asDegrees()
	elif attrFnType is om.MFnNumericAttribute:
		getMethod = getCache().numericAttrPlugMethodMap[attrFn.numericType()][0]
		data = getMethod(plug)
	elif attrFnType is om.MFnTypedAttribute:
		if attrFn.attrType() == om.MFnData.kString:
			data = plug.asString()


	# raise error on missing value
	if data is None:
		raise RuntimeError("No value retrieved for plug {}".format(plug))

	return data


def plugValue(plug:om.MPlug,
              asDegrees=True):
	"""return a corresponding python data for a plug's value
	returns nested list of individual values"""
	if plugHType(plug) == HType.Leaf:
		return leafPlugValue(plug, asDegrees=asDegrees)

	subPlugs = plugSubPlugs(plug)
	#if subPlugs:
	return [plugValue(i) for i in subPlugs]


#endregion

# region setting
def _setDataOnPlugMObject(obj, data):
	dataFn = toMFn(obj)
	dataFn.set(data)


def setLeafPlugValue(plug:om.MPlug, data,
                     toRadians=False):
	# populate plug if needed
	ensurePlugHasMObject(plug)
	print("set leaf plug value", plug, data)
	try:
		obj = plug.asMObject()
		oldData = _dataFromPlugMObject(obj)
		#return data
	except RuntimeError:  # "unexpected internal failure"

		pass
	# we now need to check the attribute type of the plug
	attribute = plug.attribute()
	attrFn = toMFn(attribute)
	attrFnType = type(attrFn)
	if attrFnType is om.MFnUnitAttribute:
		if attrFn.unitType() == om.MFnUnitAttribute.kAngle and toRadians:
			data = om.MAngle(data).toRadians()
		setFn = getCache().unitAttrPlugMethodMap[attrFn.unitType()][1]
		setFn(plug, data)

	elif attrFnType is om.MFnNumericAttribute:
		setMethod = getCache().numericAttrPlugMethodMap[attrFn.numericType()][1]
		data = setMethod(plug, data)

	elif attrFnType is om.MFnTypedAttribute:
		# for a typed attribute, we need to create a new data MObject

		if attrFn.attrType() == om.MFnData.kString:
			dataMObject = om.MFnStringData().create(data)
		else:
			raise RuntimeError("unsupported typed attribute type {}".format(attrFn.attrType()))
		plug.setMObject(dataMObject)
		return


def setPlugValue(plug:om.MPlug, data:T.Union[T.List, object],
                 fromDegrees=True,
                 errorOnFormatMismatch=False):
	"""given a plug and data, set value on that plug and any plugs below it
	expands data out to depth of plugs - eg vector arrays are preserved to
	pass to vectorArray plugs

	if not errorOnFormatMismatch, the last value of a sequence will be
	copied out to the length of plugs for that level

	"""
	subPlugs = plugSubPlugs(plug)
	if plugHType(plug) == HType.Leaf:
		return setLeafPlugValue(plug, data)

	if errorOnFormatMismatch:
		assert len(subPlugs) == len(data), "incorrect format passed to setPlugData - \n length of {} \n and {} \n must match".format(subPlugs, data)
	# copy out last array value if insufficient length
	if len(data) > len(subPlugs):
		data = (*data, *(copy.deepcopy(data[-1])
		                 for i in range(len(subPlugs) - len(data))))
	for subPlug, dataItem in zip(subPlugs, data):
		setPlugValue(subPlug, dataItem,
		             errorOnFormatMismatch=errorOnFormatMismatch)
	return


#endregion
def splitNodePlugName(plugName)->tuple[str, str]:
	"""return node, plug for plug name"""
	tokens = plugName.split(".")
	return (tokens[0], ".".join(tokens[1:]))


# character denoting last open element of array
plugLastOpenChar = "+"
plugSplitChar = "."
plugMatchChar = "*"
sliceChar = ":"
plugIndexChar = "["

topSplitPattern = re.compile("[\.\[\]]")
tokensToKeep = { plugLastOpenChar }
expandingTokens = { plugMatchChar, sliceChar}

def splitPlugLastNameToken(plug:om.MPlug)->str:
	"""return a name token specific to the given plug
	eg for a plug "foo.bar[0]", return "bar[0]"
	for an array plug return only last string index
	"""
	name = plug.name().split(".")[-1]
	if "[" in name: # return only the index characters
		name = name.split("[")[-1][:-1]
	return name


def splitPlugTokens(*stringParts:str):
	"""all necessary logic for splitting plug string call to individual tokens
	called directly with lookup - can be string, sequence, int, slice, whatever

	we convert all to strings first for ease, then eval
	some chars should be split and discarded; some split and kept as their own token
	"""
	parts = map(str, flatten(stringParts))
	resultParts = []
	for part in parts:
		pieces = list(filter(None, re.split(topSplitPattern, part)))
		for i, val in enumerate(pieces):
			if val[-1] in tokensToKeep:
				pieces[i] = (val[:-1], val[-1])
		resultParts += pieces
	return list(filter(None, flatten(resultParts)))

def parseEvalPlugTokens(strTokens:list[str])->list[(str, int, slice)]:
	"""where appropriate converts string tokens into int indices or
	slice objects"""
	realTokens = list(strTokens)
	for i, val in enumerate(strTokens):
		# check for square bracket index - [3]
		if plugIndexChar in val:
			val = val[1:-1]
		if sliceChar in val:
			val = sliceFromString(val, sliceChar) or int(val)
		else:
			try:
				val = int(val)
			except:
				pass
		realTokens[i] = val
	return realTokens

def checkLookupExpands(strTokens:list[str])->bool:
	"""return True if this lookup can result in multiple plugs,
	false if it will only return a single one"""
	return bool(set("".join(map(str, strTokens))).intersection(expandingTokens))

def plugLookupSingleLevel(parentPlug:om.MPlug,
                          token:(str, int, slice))->list[om.MPlug]:
	"""look up child plug or plugs of parent
	using string name, int index, int slice etc"""
	# print("lookup single parent", parentPlug,
	#       "token", token, parentPlug.isCompound, parentPlug.isArray)
	if parentPlug.isArray:
		if isinstance(token, int):
			return [arrayElement(parentPlug, token)]

		if token == plugLastOpenChar:
			return [newLastArrayElement(parentPlug)]
		elif isinstance(token, slice):
			# if slice has negative start or end, respect current number of plugs
			# otherwise act infinitely
			n = max(token.start, token.stop, parentPlug.evaluateNumElements())
			return [arrayElement(parentPlug, i) for i in
			        range(*token.indices(n))]

	elif parentPlug.isCompound:
		if isinstance(token, int):
			return [plugSubPlugs(parentPlug)[token]]
		plugMap = subPlugMap(parentPlug)
		if token in plugMap:
			return [plugMap[token]]
		if plugMatchChar in token:
			return [plugMap[i] for i in
			        fnmatch.filter(plugMap.keys(), token)]


def parsePlugLookup(parentPlug: om.MPlug,
                    plugTokens: (str, list[str, int, slice])) -> list[om.MPlug]:
	"""parse given tokens recursively into parentPlug's children
	"""
	if not plugTokens:
		return [parentPlug]
	hType = plugHType(parentPlug)
	# check for index into leaf plug
	if hType == HType.Leaf and plugTokens:
		raise TypeError("Cannot process lookup {} into leaf plug {}".format(
			(plugTokens), (parentPlug, type(parentPlug))))

	plugs = [parentPlug]
	while plugTokens:
		firstToken = plugTokens.pop(0)
		plugs = flatten(plugLookupSingleLevel(plug, firstToken) for plug in plugs)

	return flatten(plugs)

def topLevelPlugLookup(node:om.MFnDependencyNode,
                       plugTokens):
	"""look up a top-level plug from given mfn node
	no support yet for retrieving multiple top-level plugs
	"""
	tokens = splitPlugTokens(plugTokens)
	plugName = tokens[0]
	return node.findPlug(plugName, False)



# region connecting plugs

def checkCompoundConnection(compPlugA:om.MPlug,
                            compPlugB:om.MPlug):
	"""check that 2 compound plugs can be connected directly -
	for now just compare the number of their children"""
	assert compPlugA.isCompound and compPlugB.isCompound
	return compPlugA.numChildren() == compPlugB.numChildren()


"""
con(leafA, leafB)
fine

con(compoundA, leafB)
decompose into list
con([cAx, cAy, cAz], leafB)
truncate to shortest?
con([cAx], leafB)
NO - the above errors, as it is too hidden

con(leafA, compoundB)
con(leafA, [cBx, cBy])

con([a, b, c], [x, y, z, w])
con([a, b, c], [x, y, z])

con(leafA, arrayB)
connect to last (new) available index
con(arrayA, leafB)
connect with first index
"""

def plugPairLogic(a:om.MPlug, b:om.MPlug,
				  compoundFailsafe=False
				  )\
		->list[tuple[om.MPlug, om.MPlug]]:
	"""given 2 MPlugs or lists of MPlugs,
	return pairs of plugs defining (src, dst) for each
	individual connection
	error if formats do not adequately match

	no logic options here, write other wrapper functions for specific
	behaviour. This is confusing enough as it is

	if compoundFailsafe, connect elements of compound directly even between
	matching plugs - war flashbacks to maya2019 not
	updating compound children properly

	"""
	aType = plugHType(a)
	bType = plugHType(b)

	# special case, both are single
	if aType == HType.Leaf and bType == HType.Leaf:
		return [(a, b)]

	# if source is leaf
	if aType == HType.Leaf:

		if bType == HType.Compound:
			# connect simple plug to all compound plugs
			return [(a, i) for i in plugSubPlugs(b)]
		else:
			# b is array - connect simple to last
			return plugPairLogic(a, newLastArrayElement(b))

	aSubPlugs = plugSubPlugs(a)

	# a is compound or array
	# direct connection to leaf is illegal, too obscure
	if bType == HType.Leaf:
		raise TypeError("Array / compound plug {} cannot be directly connected to leaf plug {}".format((a, type(a)), (b, type(b))))

	bSubPlugs = plugSubPlugs(b)

	# if source is array (precludes source being compound)
	if aType == HType.Array:

		# dest is either compound or array - zip plugs and check recursively
		pairs = []
		for aSubPlug, bSubPlug in zip(aSubPlugs, bSubPlugs):
			pairs.extend(plugPairLogic(aSubPlug, bSubPlug))
		return pairs


	elif aType == HType.Compound :
		# error if destination is leaf, too obscure otherwise
		if bType == HType.Leaf:
			raise TypeError("Compound plug {} cannot be directly connected to leaf plug {}".format( (a, type(a)), (b, type(b)) ))

		"""ambiguity here: con( compound, array ) could mean to connect elementwise
		to array elements, or to connect entire plug to last array index
		
		last case is more common than first - go with that
		"""
		if bType == HType.Array:
			return plugPairLogic(a, newLastArrayElement(b))
		else: # both compound
			# if number of children equal, connect directly
			if len(aSubPlugs) == len(bSubPlugs):
				return [ (a, b) ]
			else: # zip to shortest
				return list(zip(aSubPlugs, bSubPlugs))

def plugDrivers(dstPlug:om.MPlug)->list[om.MPlug]:
	# some om functions don't allow keywords
	return dstPlug.connectedTo(True, # asDest
	                           False # asSrc
	)

def con(srcPlug:om.MPlug, dstPlug:om.MPlug, _dgMod:om.MDGModifier=None):
	"""pair up given plugs, add connections to dg modifier, execute
	if a modifier is passed in, we assume it is under external control,
	 and do not execute it within this function"""
	modifier = _dgMod or om.MDGModifier()
	plugPairs = plugPairLogic(srcPlug, dstPlug)
	# print("pairs", [(i.name(), j.name()) for i, j in plugPairs])
	for src, dst in plugPairs:
		# disconnect any existing plugs
		for prevDriver in plugDrivers(dst):
			modifier.disconnect(prevDriver, dst)
		modifier.connect(src, dst)
	if _dgMod is None: # leave to calling code to execute if specified
		modifier.doIt()

def _con(a, b, _dgMod=None):
	modifier = _dgMod or om.MDGModifier()
	# disconnect any existing plugs
	for prevDriver in plugDrivers(b):
		modifier.disconnect(prevDriver, b)
	modifier.connect(a, b)
	if _dgMod is None: # leave to calling code to execute if specified
		modifier.doIt()

def _set(plug, val, _dgMod=None):
	modifier = _dgMod or om.MDGModifier()
	modifier.newPlugValue(plug, val)
	if _dgMod is None: # leave to calling code to execute if specified
		modifier.doIt()

def conSet(src:(om.MPlug, object), dst:(om.MPlug, object), _dgMod:om.MDGModifier=None):
	"""con() but with sets of plugs or objects"""

	if isinstance(src, om.MPlug):
		con(src, dst, _dgMod)
	else:
		setPlugValue(src, dst, _dgMod)


#endregion

#region AttrSpec helper
class AttrType(TypeNamespace):
	class _Base(TypeNamespace.base()):
		args = []
		pass

	class Float(_Base):
		"""default for any new attribute"""
		args = [om.MFnNumericAttribute,
		        om.MFnNumericData.kDouble]
		pass
	class Int(_Base):
		args = [om.MFnNumericAttribute,
		        om.MFnNumericData.kInt]
		pass
	class Bool(_Base):
		args = [om.MFnNumericAttribute,
		        om.MFnNumericData.kBoolean]
		pass
	class String(_Base):
		args = [om.MFnTypedAttribute,
		        om.MFnData.kString]
		pass
	class Enum(_Base):
		args = [om.MFnEnumAttribute]
		pass
	class Message(_Base):
		args = [om.MFnMessageAttribute]
		pass
	class Matrix(_Base):
		args = [om.MFnMatrixAttribute]
		pass
	class Vector(_Base):
		"""creates 3 float attributes"""
		args = [om.MFnNumericAttribute,
		        om.MFnNumericData.k3Double]
		pass

	# arrays
	class FloatArray(_Base):
		args = [om.MFnTypedAttribute,
		        om.MFnData.kDoubleArray]
		pass
	class IntArray(_Base):
		args = [om.MFnTypedAttribute,
		        om.MFnData.kIntArray]
		pass
	class MatrixArray(_Base):
		args = [om.MFnTypedAttribute,
		        om.MFnData.kMatrixArray]
		pass

	# geo
	class Mesh(_Base):
		args = [om.MFnTypedAttribute,
		        om.MFnData.kMesh]
		pass
	class NurbsCurve(_Base):
		args = [om.MFnTypedAttribute,
		        om.MFnData.kNurbsCurve]
		pass
	class NurbsSurface(_Base):
		args = [om.MFnTypedAttribute,
		        om.MFnData.kNurbsSurface]
		pass

	# other
	class Time(_Base):
		args = [om.MFnUnitAttribute,
		        om.MFnUnitAttribute.kTime]
		pass
	class Untyped(_Base):
		args = [om.MFnAttribute]
		pass

"""test for a more consistent (wordy) way to specify new attributes - 
full object wrapper.
and of course it's a tree"""

@dataclass
class AttrData:
	type : type[AttrType._Base] = AttrType.Float
	keyable : bool = False
	array: bool = False
	min: float = None
	max: float = None
	default: float = 0.0
	channelBox: bool = False
	# enum options - order of dict is preserved
	options : dict[str, int] = None
	readable : bool = True
	writable : bool = True


class AttrSpec(Tree):
	"""tree layout for defining new attributes
	does NOT map directly to actual plugs on real nodes

	can probably unify this closer with MFns but it's a decent start
	"""

	data : AttrData = Tree.TreePropertyDescriptor("data", AttrData())

def addAttrFromSpec(node, spec:AttrSpec, parentAttrFn=None):
	"""add a new attribute hierarchy to a node based on the given spec"""
	mfn = toMFn(node)
	if spec.branches:
		attrFn = om.MFnCompoundAttribute()
		obj = attrFn.create(spec.name, spec.name)

	else:
		attrFn = spec.data.type.args[0]()
		obj = attrFn.create(
			spec.name, spec.name, *spec.data.type.args[1:])
	attrFn.array = spec.data.array
	attrFn.keyable = spec.data.keyable
	attrFn.readable = spec.data.readable
	attrFn.writable = spec.data.writable

	for i in spec.branches:
		addAttrFromSpec(node, i, attrFn)

	# if parent is given, add to it - else to node
	if parentAttrFn:
		parentAttrFn.addChild(obj)
	else:
		mfn.addAttribute(obj)

	return attrFn








