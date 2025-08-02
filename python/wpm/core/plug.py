

"""core functions for MPlugs"""

from __future__ import annotations

import typing as T

from enum import Enum

import copy, fnmatch, re

from collections import defaultdict, namedtuple
from typing import TypedDict, NamedTuple
from dataclasses import dataclass

from wplib import log, Sentinel
from wplib.wpstring import sliceFromString
from wplib.sequence import flatten, isSeq, toSeq, firstOrNone, getFirst
from wplib.object import TypeNamespace, Broadcaster, ToType, to
from wptree import Tree

from .bases import PlugBase
from .api import getMFn, mfnDataConstantTypeMap, getMObject
from .cache import getCache, om
from . import adaptor as _ # register all type conversions

if T.TYPE_CHECKING:
	pass





# region BROADCASTING
"""reminder of broadcasting rules:

single -> single
	obviously fine

single -> compound
	expand to length of compound, eg
	[single, single, single] -> compound

single -> array 
	ERROR, ambiguous, could reasonably be first, last, all, etc
	explicitly define array slice

single -> [list / slice of plugs]
	expand to length of list


list / slice -> leaf
	truncate to first value
	[ list[0] ] -> leaf
	
array plug -> leaf
	expand to array values
	[ array[1], array[2] ] -> leaf
		as above, truncate to first value


		

list / slice -> list / slice
	truncate to shortest
	use([a, b, c], [x, y, z, w])
	use([a, b, c], [x, y, z])


list / slice -> compound
	expand compound
	as above, truncate to shortest

list / slice -> array
	connect all plugs from start
	
	
in short, always expand compound to array of children
array plugs are only ones with special cases

"""


# endregion


# sick of this, time to overengineer the hell out of it



def getShape(data):
	"""return a numpy-esque shape for the given set of data -
	we only consider the first entry of each layer? in case structure
	isn't homogenous

	????????
	"""

	if isinstance(data, (tuple, list)):
		return len(data) + getShape(data[0])  # ????
	return ()


class PlugBroadcaster(Broadcaster):

	def _getElementsMPlug(self, obj: om.MPlug):
		if obj.isCompound:
			return tuple(obj.child(i) for i in range(obj.numChildren()))
		if obj.isArray:  # absolutely no damn idea
			raise RuntimeError("nooooooooooooo")
		return [obj]

	def _getElements(self, obj):
		"""return EXPANDABLE version of obj?
		no, can't bake in 'only one level'
		of immutability
		or maybe we can, maybe that's the most sane thing to do
		"""
		if plug := getMPlug(obj, None):
			return self._getElementsMPlug(plug)
		return super()._getElements(obj)
		# if isinstance(obj, PlugBase):
		# 	obj = obj.MPlug
		# if isinstance(obj, om.MPlug):

		# if isinstance(obj, (tuple, list)):
		# 	return obj
		# return (obj, )
		# if isinstance(obj, (tuple, list)):
		# 	return list(obj)
		# return [obj]

	def _isLeaf(self, obj):
		if plug := getMPlug(obj, None):
			if plug.isElement:
				return not (plug.isArray or plug.isCompound)
			return not (plug.isArray or plug.isCompound)
		return super()._isLeaf(obj)

	def _isImmutable(self, obj):
		return isinstance(obj, (str, tuple, float, int, om.MMatrix, om.MPlug))

	def _complexSourceMatchesLeafTarget(self, possible, leaf):
		"""MAYBE???
		for the case of floatArray plugs, where a normally non-leaf value
		might match a specific leaf target"""
		return False

	def _complexSidesMatchDirect(self, src, dst):
		""" check if 2 arbitrary complex objects
		can easily be said to match -
		EG if you have 2 compound attributes with the same structure.

		in this case, this pair is yielded and no more recursion done
		"""
		return False


_broadcaster = PlugBroadcaster()
# set this as default library broadcast function
broadcast = _broadcaster.broadcast
"""
for arrays - 
check if plug has sparse indices - eg if max logical index > max physical index
if yes, treat it as a map - 
if no, treat it as a dense sequence

"""





def _triplePlugValidKeyMap(plug:om.MPlug):
	"""for a given plug, if it's 3 or 4 long,
	super annoying to work out the right key for it
	 int index is preferred, but maybe we try and allow for
	 "x", "X", "r", "R", "translateX" etc?

	 """



def getMPlug(plug, default:(T.Any, None)=Sentinel.FailToFind)->om.MPlug:
	if isinstance(plug, om.MPlug):
		return plug
	try:
		return plug.MPlug
	except AttributeError:
		pass
	try:
		return plug.plug()
	except AttributeError:
		pass

	# check for initialising with (node object, attr object)
	if isinstance(plug, (tuple, list)):
		return om.MPlug(plug[0], plug[1])

	if not isinstance(plug, str):
		if default is Sentinel.FailToFind:
			raise TypeError(f"cannot retrieve MPlug from argument {plug} of type {type(plug)}")
		return default
	try:
		sel = om.MSelectionList()
		sel.add(plug)
		return sel.getPlug(0)
	except RuntimeError:
		if default is Sentinel.FailToFind:
			raise NameError(f"cannot retrieve MPlug from string {plug}")
		return default

# region querying and structure
class HType: # maybe this should be a proper Enum
	Leaf = 1
	Compound = 2
	Array = 3

def plugHType(mPlug:om.MPlug)->int:
	"""both an array and an array element plug will
	return true for isCompound if they represent a compound attribute -
	thus we check first for array, then compound"""
	return HType.Array if mPlug.isArray else (HType.Compound if mPlug.isCompound else HType.Leaf)

def plugSubPlugs(mPlug:om.MPlug)->list[om.MPlug]:
	if mPlug.isArray:
		return [mPlug.elementByPhysicalIndex(i)
		        for i in range(mPlug.numElements())]
	if mPlug.isCompound:
		return [mPlug.child(i) for i in range(mPlug.numChildren())]
	return []

def subPlugMap(mPlug)->dict[(int, str), om.MPlug]:
	"""map of all child / element plugs available here
	if array, return { logical index : plug }
	if compound, return {local plug name : plug }

	"""
	if mPlug.isArray:
		return {index : mPlug.elementByLogicalIndex(index)
		        for index in mPlug.getExistingArrayAttributeIndices()}
	if mPlug.isCompound:
		# return {mPlug.child(i).name().split(".")[-1] : mPlug.child(i)
		#         for i in range(mPlug.numChildren())}
		return {mPlug.child(i).partialName(
			includeNodeName=0,
			includeNonMandatoryIndices=False,
			includeInstancedIndices=False,
			useAlias=False,
			useFullAttributePath=False,
			useLongNames=1,
		                                   ).split(".")[-1] : mPlug.child(i)
		        for i in range(mPlug.numChildren())}
		# return {mPlug.child(i).name(): mPlug.child(i)
		#         for i in range(mPlug.numChildren())}
	return {}


def arrayPlugSlicePhysicalIndices(mPlug:om.MPlug, sl:slice)->T.Iterable[int]:
	"""simpler function returning only indices - could be useful
	so we don't have to recover indices of sparse plugs eventually
	only consider physical indices
	"""
	nElements = mPlug.evaluateNumElements()
	if sl.stop is None: # do we just error here? or assume that you mean the max existing index?
		if sl.start is None: # mans just slicin for love of the slicin
			return range(0, nElements, sl.step)
		return range(sl.start, nElements, sl.step)
	if sl.start is None:
		return range(0, sl.stop, sl.step)
	return range(sl.start, sl.stop, sl.step)


def arrayPlugSlice(mPlug:om.MPlug, sl:slice, physical=True):
	"""common logic for slice operations on mplugs
	NB: I found out a slice can contain any objects, not just ints
	if stop but no start, return all plugs up to that point

	TODO: do we have separate syntax for logical plug lookups? returning none if not found??
		not yet we don't, physical addresses always
	"""
	if physical:
		return (mPlug.elementByPhysicalIndex(i) for i in arrayPlugSlicePhysicalIndices(mPlug, sl))
	raise NotImplementedError


def clearPlugElements(mPlug:om.MPlug):
	"""up to/after certain index?
	"""


# def plugLeafName(mPlug:om.MPlug)->(int, str):
# 	if(mPlug.isElement):
# 		return mPlug.partialName(useLongNames=1)
#
# def plugPhysicalIndex(elementPlug:om.MPlug):
# 	elementPlug.array().evaluateNumElements
#
def plugParent(mPlug:om.MPlug)->om.MPlug:
	return mPlug.array() if mPlug.isElement else(
		mPlug.parent() if mPlug.isChild else None
	)

class PlugData(NamedTuple):
	"""test caching structural data to cut down on iteration -
	obviously becomes invalid after structural changes"""
	#name : str
	hType : int
	subPlugMap : dict[(int, str), om.MPlug]

	@classmethod
	def get(cls, mPlug:om.MPlug):
		return PlugData(
			#plug.name(),
			plugHType(mPlug),
			subPlugMap(mPlug)
		                )

# class PlugInterface:
# 	"""small (singleton?) classes to give uniform interfaces for
# 	setting/getting plugs"""

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



#endregion





def plugMFnAttr(plug:om.MPlug):
	"""return MFnAttribute for this plug's Attribute object"""
	attribute = plug.attribute()
	attrFn = getMFn(attribute)
	return attrFn

def plugMFnDataType(plug:om.MPlug):
	"""return MFn<Type>Data clss for the type of this plug
	if plug is not a typed attribute, return None"""
	mfnAttr = plugMFnAttr(plug)
	if not isinstance(mfnAttr, om.MFnTypedAttribute):
		return None
	return getCache().apiTypeMFnDataMap[mfnAttr.attrType()]
	# return mfnDataConstantTypeMap()[mfnAttr.attrType()]

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

def plugHash(plug:om.MPlug, includeIndex=True):
	if includeIndex:
		return hash((plug.node(), plug.attribute(), plug.logicalIndex()))
	return hash((plug.node(), plug.attribute()))


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

"""for connections, by default we work only at leaf level

but return some data from iteration to check if exact match - if so,
we can connect directly at the end of the function

we could try and defie tuple vs list as different purposes in structure,
seems opaque
"""
def _setPlugValue(target:om.MPlug, value):
	"""lowest function to directly set MPlug to given value
	CHECK here if plug is a float / vector array - these
	will be valid with deeper structures in value where simple
	numeric plugs are not
	"""
def stringMatches(s:str, matchMap:dict[str, T.Any], default=None):
	"""check if string a matches any of the keys in matchMap"""
	for k, v in matchMap.items():
		if fnmatch.fnmatch(s, k):
			return v
	return default

valT = (om.MPlug, float)
def broadcastPlugPairs(target:om.MPlug, value:(valT, tuple[valT], dict[str, valT])):
	"""set target plug to source plug's value"""
	if target.isArray:
		targetSubPlugs = plugSubPlugs(target)
		if isinstance(value, om.MPlug):
			# check rules for connecting arrays
			if plugHType(value) == HType.Array:
				"""illegal for now - should we connect all available plugs,
				should we coerce target to length of given, etc"""
				raise RuntimeError("Cannot directly connect array to array"
				                   "\n{}\nto\n{}".format(value, target))
			"""do we allow ANY connection to arrays? should it default to
			first, last, should it override first, etc
			too ambiguous for now
			"""
			raise RuntimeError("Cannot directly connect plug to array plug"
			                   "\n{}\nto\n{}".format(value, target))
			return
		if isinstance(value, dict):
			# if a list of indices on dict has been given
			# don't try to mix strings and int indices
			if isinstance(next(iter(value.keys())), int):
				for index, v in value.items():
					yield from broadcastPlugPairs(
						#arrayElement(target, index),
						target.elementByLogicalIndex(index),
						v)
				return
			for subPlug in targetSubPlugs:
				yield from broadcastPlugPairs(subPlug, value)
			return

		#TODO: WHAT IF IT'S AN ARRAY OF FLOATARRAY PLUGS
		# AAAAA
		if isSeq(value):
			for i, v in enumerate(value):
				yield from broadcastPlugPairs(target.elementByPhysicalIndex(i), v)
			return

		for i in range(target.numElements()):
			yield from broadcastPlugPairs(target.elementByPhysicalIndex(i),
			                   value)
		return

	if target.isCompound:
		targetSubPlugs = plugSubPlugs(target)
		if isinstance(value, om.MPlug):
			if plugHType(value) == HType.Leaf:
				value = (value,) * target.numChildren()
			else:
				value = plugSubPlugs(value)
			assert target.numChildren() == len(value), "compound plugs must have equal number of children to broadcast"
			for i in range(target.numChildren()):
				yield from broadcastPlugPairs(target.child(i), value[i])
			return
		elif isinstance(value, dict):
			for i in targetSubPlugs:
				yield from broadcastPlugPairs(i, value)
			return


	# we have a single plug
	if isinstance(value, dict):
		result = stringMatches(target.name(), value, default=None)
		if result is not None:
			yield (target, result)
			#_set(target, result)
			return
	yield (target, value)
	#_setPlugValue(target, value)







# def plugTreePairs(a:(om.MPlug, tuple), b:(om.MPlug, tuple)):
# 	"""
# 	yield final pairs of leaf plugs or values
# 	for connection or setting
# 	try to get to nested tuples
#
# 	check for plug types for shortcuts - float3, vector plugs etc
#
# 	still need to check for array plugs
#
# 	maybe we can use tuples inside argument to denote units that should
# 	not be expanded
#
# 	"""
# 	# convert to sequences
# 	if not isSeq(a):
# 		a = (a,)
# 	if not isSeq(b):
# 		b = (b,)
#
# 	# check if can be made plugs
# 	#print("base", a, b)
# 	a = _expandPlugSeq(a)
# 	b = _expandPlugSeq(b)
#
# 	#log("plugTreePairs", a, b)
# 	if len(a) == 1:
# 		if len(b) == 1:
# 			yield (a[0], b[0])
# 			return
# 		else:
# 			for i in b:
# 				yield from (plugTreePairs(a, i))
# 			return
# 	if len(a) == len(b):
# 		for i, j in zip(a, b):
# 			yield from plugTreePairs(i, j)
# 		return
#
# 	raise ValueError("plugTreePairs: mismatched plug sequences", a, b)



#region getting

def _dataFromPlugMObject(obj:om.MObject):
	"""retrieve data from direct plug MObject"""
	dataFn = getMFn(obj)
	try:
		return list(dataFn.array())
	except:
		pass
	if isinstance(dataFn, om.MFnMatrixData):
		return dataFn.matrix()
	elif isinstance(dataFn, om.MFnStringData):
		return dataFn.string()
	return dataFn.getData()

def _leafPlugValue(plug:om.MPlug,
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
	attrFn = getMFn(attribute)
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

def isTriplePlug(plug:om.MPlug):
	"""test if plug represents a triple element like a colour or vector
	"""

def plugValue(plug:om.MPlug,
              asDegrees=True,
              returnMap=False
              ):
	"""return a corresponding python data for a plug's value
	returns nested list of individual values

	make dict for compound
	EXCEPT if it's a compound of 3 or 4 numerics - eg transform.translate, colorRGB etc
		seems common enough to make exception

	"""
	hType = plugHType(plug)
	if hType == HType.Leaf:
		return _leafPlugValue(plug, asDegrees=asDegrees)
	subPlugs = plugSubPlugs(plug)
	if hType == HType.Array:
		return [plugValue(i) for i in subPlugs]

	valMap = {k : plugValue(v, asDegrees) for k, v in subPlugMap(plug).items()}
	if len(valMap) in (3, 4) and all(isinstance(i, (int, float)) for i in valMap.values()):
		return list(valMap.values()) # triple attributes to tuples
	if returnMap:
		return valMap
	return list(valMap.values())



#endregion

# region setting
def _setDataOnPlugMObject(obj, data):
	dataFn = getMFn(obj)
	dataFn.set(data)


def _setLeafPlugValue(plug:om.MPlug, data,
                     toRadians=True,
                      _dgMod:om.MDGModifier=None
                      ):
	"""TODO: array-valued plug types -
	MFloatArray, MDoubleArray, MVectorArray, MMPointArray, MMatrixArray, etc

	run after broadcasting, set a leaf-level value on a leaf-level plug


	SO -
	if you have a k3Double or k2Float attribute, which generates its own leaf attributes,
	om.MFnAttribute( leafPlug.attribute() ) will ERROR on getting name, numericType, etc
	it doesn't exist / isn't valid with the mfns

	2 courses of action - try to check at every higher level if we hit a triple attribute,
		and have a special way of setting them early before leaf level -
			could be viable
	easier: check apiType() of the MObject of the plug -

	transform.translate plug.attribute().apiType() -> kAttribute3Double
	transform.translateX plug.attribute().apiType() -> kDoubleLinearAttribute

	NEVERMIND, I'll delete this next commit, but it's because (naively, like an idiot)
		I assumed child attributes of a Numeric attribute would also be Numeric attributes

	transform.translate : MFnNumericAttribute , kAttribute3Double , k3Double numeric type etc
	transform.translateX : MFnUnitAttribute, kDistance

	getMFn already accounts for this, so depend on the type it gives
	"""
	modifier = _dgMod or om.MDGModifier()

	# we now need to check the attribute type of the plug
	attribute = plug.attribute()
	attrFn = getMFn(attribute)
	attrFnType = type(attrFn)
	if attrFnType == om.MFnUnitAttribute: # spaghett
		if attrFn.unitType() == om.MFnUnitAttribute.kAngle:
			if toRadians:
				data = om.MAngle(data).asRadians()
			if not isinstance(data, om.MAngle):
				data = om.MAngle(data, om.MAngle.kDegrees)
				modifier.newPlugValueMAngle(plug, data)
		elif attrFn.unitType() == om.MFnUnitAttribute.kTime:
			modifier.newPlugValueMTime(plug, om.MTime(data))
		elif attrFn.unitType() == om.MFnUnitAttribute.kDistance:
			modifier.newPlugValueMDistance(plug, om.MDistance(data))
		else:
			raise RuntimeError("Unknown/unsupported typed attr plug", plug)

	elif attrFnType == om.MFnNumericAttribute:
		if attrFn.numericType() == om.MFnNumericData.kInt:
			modifier.newPlugValueInt(plug, data)
		elif attrFn.numericType() == om.MFnNumericData.kLong:
			modifier.newPlugValueInt(plug, data)
		elif attrFn.numericType() == om.MFnNumericData.kShort:
			modifier.newPlugValueShort(plug, data)
		elif attrFn.numericType() == om.MFnNumericData.kBoolean:
			modifier.newPlugValueBool(plug, data)
		elif attrFn.numericType() == om.MFnNumericData.kFloat:
			modifier.newPlugValueFloat(plug, data)
		elif attrFn.numericType() == om.MFnNumericData.kDouble:
			modifier.newPlugValueDouble(plug, data)

	elif attrFnType == om.MFnTypedAttribute:
		if attrFn.attrType() == om.MFnData.kString:
			# dataMObject = om.MFnStringData().create(data)
			# plug.setMObject(dataMObject)
			modifier.newPlugValueString(plug, str(data))
		elif attrFn.attrType() in (om.MFnData.kNurbsCurve, om.MFnData.kMesh, om.MFnData.kNurbsSurface):
			assert isinstance(data, om.MObject), "must directly specify MObject to set geometry plug: " + str(plug)
			#plug.setMObject(data)
			modifier.newPlugValue(plug, data)
		else:
			raise RuntimeError("unsupported typed attribute type {}".format(attrFn.attrType()))

	elif attrFnType == om.MFnMatrixAttribute:
		modifier.newPlugValue(plug, om.MFnMatrixData().create( to(data, om.MMatrix )))

	if not _dgMod:
		modifier.doIt()


def setPlugValue(plug:om.MPlug, data:T.Union[T.List, object],
                 fromDegrees=True,
                 _dgMod:om.MDGModifier=None # might use this eventually
                 ):
	"""given a plug and data, set value on that plug and any plugs below it
	expands data out to depth of plugs - eg vector arrays are preserved to
	pass to vectorArray plugs

	if not errorOnFormatMismatch, the last value of a sequence will be
	copied out to the length of plugs for that level

	"""
	modifier = _dgMod or om.MDGModifier()

	dst = [getMPlug(i, i) for i in toSeq(plug)]
	for pair in broadcast(data, dst):
		if not isinstance(pair[1], om.MPlug):
			raise RuntimeError("destination is not an MPlug")
		if isinstance(pair[0], om.MPlug):
			value = plugValue(plug)
		else:
			value = pair[0]
		_setLeafPlugValue(pair[1], value,
		             toRadians=fromDegrees,
		                  _dgMod=_dgMod

		             )
	if _dgMod is None: # leave to calling code to execute if specified
		modifier.doIt()


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
use(leafA, leafB)
fine

use(compoundA, leafB)
decompose into list
use([cAx, cAy, cAz], leafB)
truncate to shortest?
use([cAx], leafB)
NO - the above errors, as it is too hidden

BUT - if you explicitly supply single-item list? then it's allowed?


use(leafA, compoundB)
use(leafA, [cBx, cBy])

use([a, b, c], [x, y, z, w])
use([a, b, c], [x, y, z])

use(leafA, arrayB)
connect to last (new) available index
use(arrayA, leafB)
connect with first index
"""

# def plugPairLogic(a:om.MPlug, b:om.MPlug,
# 				  compoundFailsafe=False
# 				  )\
# 		->list[tuple[om.MPlug, om.MPlug]]:
# 	"""given 2 MPlugs or lists of MPlugs,
# 	return pairs of plugs defining (src, dst) for each
# 	individual connection
# 	error if formats do not adequately match
#
# 	no logic options here, write other wrapper functions for specific
# 	behaviour. This is confusing enough as it is
#
# 	if compoundFailsafe, connect elements of compound directly even between
# 	matching plugs - war flashbacks to maya2019 not
# 	updating compound children properly
#
# 	"""
# 	aType = plugHType(a)
# 	bType = plugHType(b)
#
# 	# special case, both are single
# 	if aType == HType.Leaf and bType == HType.Leaf:
# 		return [(a, b)]
#
# 	# if source is leaf
# 	if aType == HType.Leaf:
#
# 		if bType == HType.Compound:
# 			# connect simple plug to all compound plugs
# 			return [(a, i) for i in plugSubPlugs(b)]
# 		else:
# 			# b is array - connect simple to last
# 			return plugPairLogic(a, newLastArrayElement(b))
#
# 	aSubPlugs = plugSubPlugs(a)
#
# 	# a is compound or array
# 	# direct connection to leaf is illegal, too obscure
# 	if bType == HType.Leaf:
# 		raise TypeError("Array / compound plug {} cannot be directly connected to leaf plug {}".format((a, type(a)), (b, type(b))))
#
# 	bSubPlugs = plugSubPlugs(b)
#
# 	# if source is array (precludes source being compound)
# 	if aType == HType.Array:
#
# 		# dest is either compound or array - zip plugs and check recursively
# 		pairs = []
# 		for aSubPlug, bSubPlug in zip(aSubPlugs, bSubPlugs):
# 			pairs.extend(plugPairLogic(aSubPlug, bSubPlug))
# 		return pairs
#
#
# 	elif aType == HType.Compound :
# 		# error if destination is leaf, too obscure otherwise
# 		if bType == HType.Leaf:
# 			raise TypeError("Compound plug {} cannot be directly connected to leaf plug {}".format( (a, type(a)), (b, type(b)) ))
#
# 		"""ambiguity here: use( compound, array ) could mean to connect elementwise
# 		to array elements, or to connect entire plug to last array index
#
# 		last case is more common than first - go with that
# 		"""
# 		if bType == HType.Array:
# 			return plugPairLogic(a, newLastArrayElement(b))
# 		else: # both compound
# 			# if number of children equal, connect directly
# 			if len(aSubPlugs) == len(bSubPlugs):
# 				return [ (a, b) ]
# 			else: # zip to shortest
# 				return list(zip(aSubPlugs, bSubPlugs))

def tryConvertToMPlugs(struct:termType)->(om.MPlug, list[om.MPlug]):
	"""iterate through structure, convert to mplugs if possible"""
	if isinstance(struct, (tuple, list)):
		return [tryConvertToMPlugs(i) for i in struct]
	return getMPlug(struct, None) or struct

def plugDrivers(dstPlug:om.MPlug)->list[om.MPlug]:
	# some om functions don't allow keywords
	return dstPlug.connectedTo(True, # asDest
	                           False # asSrc
	)

def use(src:(T.Any, om.MPlug), dst:om.MPlug,
        fromDegrees=True,
        _dgMod:om.MDGModifier=None):
	"""use() but with sets of plugs or objects
	DO WE convert to MPlugs before or after broadcasting?
	complex addressing / expressions are all taken care of before this, here we
	expect at most flat lists
	"""
	#log("use", src, dst)
	modifier = _dgMod or om.MDGModifier()

	src = ([getMPlug(i, i) for i in toSeq(src)])
	dst = ([getMPlug(i, i) for i in toSeq(dst)])
	for s, d in zip(src, dst):
		for pair in broadcast(s, d):
			if not isinstance(pair[1], om.MPlug):
				raise RuntimeError("destination is not an MPlug")
			if isinstance(pair[0], om.MPlug):
				_con(pair[0], pair[1], _dgMod)
			else:
				_setLeafPlugValue(pair[1], pair[0],
				             toRadians=fromDegrees,
				                  _dgMod=_dgMod

				             )
	if _dgMod is None: # leave to calling code to execute if specified
		modifier.doIt()

def _con(a:om.MPlug, b:om.MPlug, _dgMod:om.MDGModifier=None):
	modifier = _dgMod or om.MDGModifier()
	# disconnect any existing plugs
	for prevDriver in plugDrivers(b):
		modifier.disconnect(prevDriver, b)
	modifier.connect(a, b)
	if _dgMod is None: # leave to calling code to execute if specified
		modifier.doIt()


"""
semantics - 
for setAttr, this order is backwards from normal maya, where you would say 'setAttr attribute (to) value'
here we just stick to a basic order of (source, target)
and hope it clicks eventually
(other naming suggestions welcome)

TEST:
find a statement that must include "src", "dst" IN THAT ORDER
which is readable

drive( src, dst )?
	' drive src with dst ' - unclear
	' use src to drive dst '
	
set( src, dst )?
	' set src to dst '
	' set src with dst '
	' set using src to drive dst '
	- trash, also confuses with native setAttr, this can also connect

equate( src, dst )?

put( src, dst ) ?
	' put src in dst ' - this seems quite good? can't see how you would misread the order
	- 'put' may sound like a more static action than making a live connection

use( src, dst ) ?
	' use src as dst '
	' use src to drive dst '
	- this seems good


"""
# def use(src:(om.MPlug, object), dst:(om.MPlug, list[om.MPlug]),
#         fromDegrees=True,
#         _dgMod:om.MDGModifier=None,
#         ):




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
	mfn = getMFn(node)
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
		parentAttrFn.addBranch(obj)
	else:
		mfn.addAttribute(obj)

	return attrFn








