
from __future__ import annotations

import types, pprint
import typing as T
import inspect
from types import FunctionType
from collections import defaultdict
from dataclasses import dataclass
from collections import deque, namedtuple
from typing import TypedDict

from wplib import log
from wplib.log import getDefinitionStrLink
#from wplib.sentinel import Sentinel
from wplib.object import Adaptor
from wplib.inheritance import superClassLookup, SuperClassLookupMap, isNamedTupleInstance, isNamedTupleClass
from wplib.object.namespace import TypeNamespace
from wplib.constant import MAP_TYPES, SEQ_TYPES, LITERAL_TYPES

#from wptree import Tree

class ChildType(TypeNamespace):

	class _Base(TypeNamespace.base()):
		pass

	class SequenceElement(_Base):
		pass

	class MapItem(_Base):
		pass

	class MapKey(_Base):
		pass

	class MapValue(_Base):
		pass

	class ObjectAttribute(_Base):
		pass


"""
consider case of visiting dict items -
it's useful to see key and value together,
while it's also useful to know when a contained object
is used as a key or value

how do we chain multiple levels of iteration like this?
could do a hack : return a custom tuple type, and
set up special processing for it


[
(item, MapItem), 
((item[0], MapKey), (item[1], MapValue))
]
??

"""

#region get child objects

MapItemTie = namedtuple("MapItemTie", "key value")

#childTypeItemMapType = T.Mapping[ChildType.T, list[T.Any]]
typeChildObjectsFnMap = {
	type(None) : lambda obj: (),
	MAP_TYPES : lambda obj: (
			(MapItemTie(*i), ChildType.MapItem)
	             for i in obj.items()
	),
	MapItemTie : lambda obj: ((obj[0], ChildType.MapKey), (obj[1], ChildType.MapValue)),

	SEQ_TYPES : lambda obj: ((i, ChildType.SequenceElement) for i in obj),

	LITERAL_TYPES : lambda obj: (),

}
# endregion

# region updating
# need to return values in case of immutable types
def _updateMap(parentObj:dict, childObj:tuple, childType):
	parentObj[childObj[0]] = childObj[1]
	return parentObj

typeUpdateFnMap = {
	type(None) : lambda parentObj, childObj, childType: childObj,
	SEQ_TYPES : lambda parentObj, childObj, childType: parentObj + (childObj,),
	MAP_TYPES : _updateMap,

}
# endregion

# region creating new

def _newMap(baseParent, childTypeItemMap:dict[ChildType.T, list[T.Any]]):
	#log("newMap", childTypeItemMap)
	return dict(childTypeItemMap[ChildType.MapItem])

def _newSeq(baseParent, childTypeItemMap:dict[ChildType.T, list[T.Any]]):
	#log("newSeq", childTypeItemMap)
	#return tuple(childTypeItemMap[ChildType.SequenceElement])
	if isNamedTupleInstance(baseParent):
		return type(baseParent)(*childTypeItemMap[ChildType.SequenceElement])
	#newType = type(baseParent)

	return type(baseParent)(childTypeItemMap[ChildType.SequenceElement])

# def _newTuple(baseParent, childTypeItemMap:dict[ChildType.T, list[T.Any]]):
# 	if isNamedTupleInstance(baseParent):
# 		return type(baseParent)(*childTypeItemMap[ChildType.SequenceElement])
# 	return tuple(childTypeItemMap[ChildType.SequenceElement])

childTypeItemMapType = T.Mapping[ChildType.T, list[T.Any]]
typeNewFnMap = {
	type(None) : lambda baseParent, childTypeItemMap: None,
	object : lambda baseParent, childTypeItemMap: type(baseParent)(
		(childTypeItemMap.popitem()[1])),
	MapItemTie : lambda baseParent, childTypeItemMap: type(baseParent)(
		childTypeItemMap[ChildType.MapKey][0], childTypeItemMap[ChildType.MapValue][0]),

	SEQ_TYPES : _newSeq,
	#MAP_TYPES : lambda baseParent, childTypeItemMap: dict(childTypeItemMap[ChildType.MapItem]),
	MAP_TYPES : _newMap,

}
# endregion

# test new adaptor system
class VisitAdaptor(Adaptor):
	"""adaptor for visit system - defines how to traverse and regenerate
	registered objects.

	It would make sense to combine this with pathing functions for the Traversable system"""
	# new base class, declare new map
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	# declare abstract methods
	@classmethod
	def childObjects(cls, obj:T.Any)->T.Iterable[tuple[T.Any, ChildType.T]]:
		"""return iterable of (childObject, childType) pairs"""
		raise NotImplementedError

	@classmethod
	def newObj(cls, baseObj:T.Any, childTypeItemMap:dict[ChildType.T, list[T.Any]])->T.Any:
		"""create new object from base object and child type item map,
		a child type item map is a dict of {childType : [list of child objects of that type]}
		"""
		raise NotImplementedError

class NoneVisitAdaptor(VisitAdaptor):
	forTypes = (type(None),)
	@classmethod
	def childObjects(cls, obj:T.Any) ->T.Iterable[tuple[T.Any, ChildType.T]]:
		return ()

class LiteralVisitAdaptor(VisitAdaptor):
	forTypes = LITERAL_TYPES
	@classmethod
	def childObjects(cls, obj:T.Any) ->T.Iterable[tuple[T.Any, ChildType.T]]:
		return ()

class MapVisitAdaptor(VisitAdaptor):
	forTypes = MAP_TYPES
	@classmethod
	def childObjects(cls, obj:T.Any) ->T.Iterable[tuple[T.Any, ChildType.T]]:
		return (
			(MapItemTie(*i), ChildType.MapItem)
	             for i in obj.items()
		)
	@classmethod
	def newObj(cls, baseObj:T.Any, childTypeItemMap:dict[ChildType.T, list[T.Any]])->T.Any:
		return dict(childTypeItemMap[ChildType.MapItem])

class MapItemTieVisitAdaptor(VisitAdaptor):
	forTypes = (MapItemTie,)
	@classmethod
	def childObjects(cls, obj:T.Any) ->T.Iterable[tuple[T.Any, ChildType.T]]:
		return ((obj[0], ChildType.MapKey), (obj[1], ChildType.MapValue))
	@classmethod
	def newObj(cls, baseObj:T.Any, childTypeItemMap:dict[ChildType.T, list[T.Any]])->T.Any:
		return type(baseObj)(
			childTypeItemMap[ChildType.MapKey][0], childTypeItemMap[ChildType.MapValue][0])

class SeqVisitAdaptor(VisitAdaptor):
	forTypes = SEQ_TYPES
	@classmethod
	def childObjects(cls, obj:T.Any) ->T.Iterable[tuple[T.Any, ChildType.T]]:
		return ((i, ChildType.SequenceElement) for i in obj)
	@classmethod
	def newObj(cls, baseObj:T.Any, childTypeItemMap:dict[ChildType.T, list[T.Any]])->T.Any:
		if isNamedTupleInstance(baseObj):
			return type(baseObj)(*childTypeItemMap[ChildType.SequenceElement])

		return type(baseObj)(childTypeItemMap[ChildType.SequenceElement])

class VisitableBase:
	"""custom base interface for custom types -
	we associate an adaptor type for these later"""
	def childObjects(self):
		raise NotImplementedError
	@classmethod
	def newObj(cls, baseObj, childTypeItemMap):
		raise NotImplementedError

class VisitableVisitAdaptor(VisitAdaptor):
	"""integrate derived subclasses with adaptor system
	"""
	forTypes = (VisitableBase,)
	@classmethod
	def childObjects(cls, obj:T.Any) ->T.Iterable[tuple[T.Any, ChildType.T]]:
		return obj.childObjects()
	@classmethod
	def newObj(cls, baseObj:T.Any, childTypeItemMap:dict[ChildType.T, list[T.Any]])->T.Any:
		return baseObj.newObj(baseObj, childTypeItemMap)


# just do this basic function
def transformTestTopDown(target):
	# first transform object (not recursively in serialisation
	result = transform(target)
	# get children of result
	childTypeItemMapType = getChildren(result)

	# return a new object (NOT the exact object returned from transform - bit weird)
	return newFn(
		origObj=result,
	    children={k : transformTest(v) for k, v in childTypeItemMapType} )


# collect all in single class
class VisitTypeFunctionRegister:
	"""register functions for visiting types"""

	def __init__(self,
	             typeChildObjectsFnMap:dict[T.Type, T.Callable]=None,
	             typeUpdateFnMap:dict[T.Type, T.Callable]=None,
	             typeNewFnMap:dict[T.Type, T.Callable]=None,

	             ):
		self.typeChildObjectsFnMap = SuperClassLookupMap(typeChildObjectsFnMap)
		self.typeUpdateFnMap = SuperClassLookupMap(typeUpdateFnMap)
		self.typeNewFnMap = SuperClassLookupMap(typeNewFnMap)

	def registerChildObjectsFnForType(self, type, fn):
		self.typeChildObjectsFnMap.updateClassMap({type:fn})

	def registerUpdateFnForType(self, type, fn):
		self.typeUpdateFnMap.updateClassMap({type:fn})

	def registerNewFromArgsFnForType(self, type, fn):
		self.typeNewFnMap.updateClassMap({type:fn})

	def getChildObjectsFnForType(self, type):
		return self.typeChildObjectsFnMap.lookup(type)

	def getUpdateFnForType(self, type):
		return self.typeUpdateFnMap.lookup(type)

	def getNewFromArgsFnForType(self, type):
		return self.typeNewFnMap.lookup(type)


visitFunctionRegister = VisitTypeFunctionRegister(
	typeChildObjectsFnMap=typeChildObjectsFnMap,
	typeUpdateFnMap=typeUpdateFnMap,
	typeNewFnMap=typeNewFnMap,
)


class VisitObjectData(TypedDict):
	base : T.Any # root object of the visit
	visitResult : T.Any # temp result of the visit
	copyResult : T.Any # copy of final result
	childType : ChildType.T # type of current object
	childDatas : list[VisitObjectData] # tuple of child data for current object
	makeNewObjFromVisitResult : bool # if true, make new object from visit result - if false, use visit result as is


@dataclass
class VisitPassParams:
	"""Parametres governing single iteration of visit"""
	topDown:bool = True
	depthFirst:bool = True
	runVisitFn:bool = True # if false, only yield objects to visit
	transformVisitedObjects:bool = False # if true, modifies visited objects - yields (original, transformed) pairs
	visitFn:visitFnType = None # if given, overrides visitor's visit function
	visitKwargs:dict = None # if given, kwargs to pass to visit function


	pass


class DeepVisitOp:

	def __init__(self):
		pass

	def visit(self,
	          obj:T.Any,
	              visitor:DeepVisitor,
	              visitObjectData:VisitObjectData,
	              visitPassParams:VisitPassParams,
	              )->T.Any:
		"""template function to override for custom transform"""
		raise NotImplementedError



class DeepVisitor:
	"""base class for visit and transform operations over all elements
	of a data structure.

	For now a transformation cannot add or remove elements - maybe add later
	using extensions to visitData.

	Run filter function over all elements for now, leave any filtering to
	client code.
	"""

	ChildType = ChildType
	VisitObjectData = VisitObjectData
	VisitPassParams = VisitPassParams
	DeepVisitOp = DeepVisitOp

	@classmethod
	def getDefaultVisitTypeRegister(cls,
	                                        )->VisitTypeFunctionRegister:
		"""get default type function register"""
		return visitFunctionRegister


	@classmethod
	def visitSingleObject(cls,
	                      obj,
	                      visitor:DeepVisitor,
	                      visitObjectData:VisitObjectData=None,
	                      visitPassParams:VisitPassParams=None,
	                      )->None:
		"""visit a single object - if needed,
		populate VisitResult and modify VisitParams
		"""
		return obj

	@classmethod
	def getNextVisitDestinationsFn(cls,
	                          )->callable:
		"""get default function for getting next objects to visit"""
		return

	@classmethod
	def checkVisitFnSignature(cls, fn:visitFnType):
		"""check that the given function has the correct signature"""
		fnId = f"\n{fn} def {getDefinitionStrLink(fn)} \n"
		if not isinstance(fn, (types.FunctionType, types.MethodType)):
			raise TypeError(f"visit function " + fnId + " is not a function")
		# if fn.__code__.co_argcount != 4:
		# 	raise TypeError(f"visit function {fn} does not have 4 arguments")
		argSeq = ("obj", "visitor", "visitObjectData", "visitPassParams")
		#if fn.__code__.co_varnames[-4:] != argSeq:
		# if not (set(argSeq) <= set(fn.__code__.co_varnames)):
		# 	raise TypeError(f"visit function " + fnId + f"does not have correct argument names\n{argSeq} \n{argSeq[-4:]}\n{fn.__code__.co_varnames}")
		return True

	def __init__(self,
	             visitSingleObjectFn:visitFnType=None,
	             visitTypeFunctionRegister: VisitTypeFunctionRegister=visitFunctionRegister,
	             #nextVisitDestinationsFn:callable=None,
	             ):
		"""initialize the visitor with functions
		to handle visiting single objects and
		getting next objects to visit
		"""
		self.visitTypeFunctionRegister = visitTypeFunctionRegister
		self.visitSingleObjectFn = visitSingleObjectFn or self.visitSingleObject
		self.checkVisitFnSignature(self.visitSingleObjectFn)

		#self.nextVisitDestinationsFn = nextVisitDestinationsFn or self.nextVisitDestinations


	def _visitAll(self,
	              parentObj,
	              visitParams:VisitPassParams,
	              )->T.Generator[tuple, None, None]:
		"""visit objects in a flat top-down manner,
		no readback, no transformation"""
		toVisit = deque()
		visited = []

		visitData = VisitObjectData(
			base=parentObj,
			visitResult=None,
			copyResult=None,
			childType=None,
			childDatas=[],
		)

		toVisit.append(visitData)

		while toVisit:
			visitData = toVisit.pop() #type: VisitObjectData

			# visit object
			toIter = visitData["base"]
			if visitParams.topDown:
				if visitParams.runVisitFn:
					result = visitParams.visitFn(
						visitData["base"], self, visitData, visitParams)
					visitData["visitResult"] = result

					# yield base and result
					yield (visitData["base"], result)
					toIter = result
				else:
					yield visitData["base"]

			visited.append(visitData)

			# get next objects to visit
			childObjectsFn = self.visitTypeFunctionRegister.getChildObjectsFnForType(type(toIter))
			nextObjTies = childObjectsFn(toIter)

			# build new visit datas for next objects
			for nextObj, childType in nextObjTies:
				nextVisitData = VisitObjectData(
					base=parentObj,
					visitResult=None,
					copyResult=None,
					childType=childType,
					childDatas=[],
				)
				toVisit.append(nextVisitData)

		# if top down, we're done
		if visitParams.topDown:
			return

		# if bottom up, we need to iterate over visited array
		# and yield results
		for visitData in reversed(visited):
			if visitParams.runVisitFn:
				# run visit function
				result = visitParams.visitFn(
					visitData["base"], self, visitData, visitParams)
				visitData["visitResult"] = result
				# yield base and result
				yield (visitData["base"], result)
			else:
				yield visitData["base"]



	def _transform(self,
	               parentObj,
	               visitParams:VisitPassParams,
	               ):
		"""iterate over objects without recursion

		does this need multiple passes? one to collect objects,
		one to visit them?

		need at least 2 passes - top down to process objects and collect
		results, bottom up to reconstruct new objects
		"""

		toVisit = deque()
		visited = []

		visitData = VisitObjectData(
			base=parentObj,
			visitResult=None,
			copyResult=None,
			childType=None,
			childDatas=[],
			makeNewObjFromVisitResult=True
		)

		toVisit.append(visitData)


		#print("visitParams", visitParams)
		while toVisit:
			#log("toVisit", toVisit)

			visitData = toVisit.pop() #type: VisitObjectData

			# visit object
			toIter = visitData["base"]
			if visitParams.topDown:
				result = visitParams.visitFn(
					visitData["base"], self, visitData, visitParams)
				visitData["visitResult"] = result

				# # yield base and result
				# yield (visitData["base"], result)
				toIter = result
				#log("base", visitData["base"], "result", result)

			visited.append(visitData)

			# get next objects to visit
			childObjectsFn = self.visitTypeFunctionRegister.getChildObjectsFnForType(
				type(toIter)
			)
			nextObjTies = childObjectsFn(toIter)

			# build new visit datas for next objects
			for nextObj, childType in nextObjTies:
				nextVisitData = VisitObjectData(
					base=nextObj,
					visitResult=None,
					copyResult=None,
					childType=childType,
					childDatas=[],
					makeNewObjFromVisitResult=True
				)
				# add ref to child data
				visitData["childDatas"].append(nextVisitData)
				# add new data to be visited next
				toVisit.append(nextVisitData)

		# reconstruct objects, bottom up
		from wptree import Tree
		for visitData in reversed(visited):
			visitData : VisitObjectData

			result = None
			# if bottom up, do visiting now
			if not visitParams.topDown:
				result = visitParams.visitFn(
					visitData["base"], self, visitData, visitParams)
				if isinstance(result, Tree):
					log("TREE")
					print("visitData")
					print(visitData["childDatas"])
					print(visitData["makeNewObjFromVisitResult"])
				visitData["visitResult"] = result


			if not visitData['childDatas'] or not visitData['makeNewObjFromVisitResult']:
				try:
					if isinstance(result, Tree):
						log("continuing")
				except:
					pass
				visitData['copyResult'] = visitData['visitResult']
				continue

			newObjFn = self.visitTypeFunctionRegister.getNewFromArgsFnForType(
				type(visitData['visitResult'])
			)

			if isinstance(result, Tree):
				log("newObjFn", newObjFn)

			# build map of {childType : [ list of child objects of that type]}
			childTypeToChildObjectsMap = defaultdict(list)
			for childData in visitData['childDatas']:
				childTypeToChildObjectsMap[childData['childType']].append(childData['copyResult'])

			newObjFn : callable[[T.Any, dict[ChildType.T(), T.Any]], T.Any]
			try:
				visitData['copyResult'] = newObjFn(visitData["visitResult"],
				                                   childTypeToChildObjectsMap)
				if isinstance(result, Tree):
					log("copyResult", visitData['copyResult'])
			except Exception as e:
				log(f"Error creating new object from {visitData['visitResult']} \n and {pprint.pformat(childTypeToChildObjectsMap, indent=4, compact=False)}")
				pprint.pprint(visitData)
				log("newObjFn", newObjFn, )
				pprint.pprint(inspect.signature(newObjFn))
				raise e
		#log("visited", visited)
		print("returning", visited[0]['copyResult'])
		return visited[0]['copyResult']


	def _transformRecursiveTopDown(self,
	               parentObj,
	               visitParams:VisitPassParams,
	                               visitData:VisitObjectData=None,
	               ):
		"""doing this without recursion is beyond me for now"""
		# visitData = visitData or VisitObjectData(
		# 	base=parentObj,
		# 	visitResult=None,
		# 	copyResult=None,
		# 	childType=None,
		# 	childDatas=[],
		# 	makeNewObjFromVisitResult=True
		# )
		visitData = {}
		result = visitParams.visitFn(
			parentObj, self, visitData, visitParams)

		# get next objects to visit
		childObjectsFn = self.visitTypeFunctionRegister.getChildObjectsFnForType(
			type(result)
		)
		nextObjTies = childObjectsFn(result)
		for obj in nextObjTies:
			self._transformRecursiveTopDown(obj, visitParams, visitData)



	def dispatchPass(self,
	                 fromObj:T.Any,
	                 passParams:VisitPassParams,
	                 visitFn:visitFnType=None,
	                 **kwargs
	                 ):
		"""dispatch a single pass of the visitor
		"""
		# allowing LOADS of override levels for setting visit function here,
		# unnecessary
		if visitFn is not None: # set visit function if given
			passParams.visitFn = visitFn
		else:
			passParams.visitFn = passParams.visitFn or self.visitSingleObjectFn
		self.checkVisitFnSignature(passParams.visitFn)

		passParams.visitKwargs = passParams.visitKwargs or {}
		passParams.visitKwargs.update(kwargs)

		if passParams.transformVisitedObjects:
			result = self._transform(fromObj, passParams)
			return result
		else:
			return self._visitAll(fromObj, passParams)


visitFnType = T.Callable[
	[T.Any,
	 DeepVisitor,
	 VisitObjectData,
	 VisitPassParams],
	T.Any]



if __name__ == '__main__':

	def printArgsVisit(obj, visitor, visitData, visitParams):
		#print(obj, visitor, visitData, visitParams)
		print(obj, visitData["childType"])
		return obj

	visitor = DeepVisitor(
		visitTypeFunctionRegister=visitFunctionRegister,
		visitSingleObjectFn=printArgsVisit)

	structure = {
		"key1": "value1",
		(2, 4, "fhffhs"): ["value2", [], 3, 4, 5],
		"key3": "value3",
	}

	# visitPass = visitor._visitAll(structure, VisitPassParams())
	# for i in visitPass:
	# 	print("visited", i)


	def addOneTransform(obj, visitor, visitData, visitParams):
		#print("addOneTransform", obj)
		if isinstance(obj, int):
			obj += 1
		return obj

	visitor = DeepVisitor(
		visitTypeFunctionRegister=visitFunctionRegister,
		visitSingleObjectFn=addOneTransform)

	structure = [
		1, [2, [3, 4], 2], 1
	]
	print("structure", structure)
	newStructure = visitor.dispatchPass(structure, VisitPassParams(
		transformVisitedObjects=False))
	print("newStructure", newStructure)

	print("structure", structure)
	newStructure = visitor.dispatchPass(structure, VisitPassParams(
		transformVisitedObjects=True,
		topDown=False
	))
	print("newStructure", newStructure)



	pass






