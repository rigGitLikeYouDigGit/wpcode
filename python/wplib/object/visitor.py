
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
	"""Enum types to mark what kind of "child" each object is in a data structure"""

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


# consistent type passed to all visit functions
#ITEM_CHILD_LIST_T = T.Iterable[tuple[T.Any, ChildType.T]]

# enum / namespace was too involved, just use strings
# (childType, key, object)
CHILD_T = tuple[str, T.Any, T.Optional[str]]
ITEM_CHILD_LIST_T = T.Iterable[CHILD_T]

# test new adaptor system
class VisitAdaptor(Adaptor):
	"""adaptor for visit system - defines how to traverse and regenerate
	registered objects.

	It would make sense to combine this with pathing
	 functions for the Traversable system
	  - do this now

	for key type, consider a shorthand "k" for key, "a" for attribute

	top-level literal tuples or generators in path create object slices

	[ 0, "key", (1, 2) ] - get slice of two objects
	[ 0, ".attr", "key", ["branch", "leaf"] ] - expand branch/leaf into single path lookup for tree

	[ 0, "key", where({ "key": "value" }) ] - get slice of objects where key is value
	[ 0, "key", where(key="value") ] - get slice of objects where key is value
	[ 0, ".methodCall()" ] - call method on object with no args?
	[ 0, ".methodCall(", [1, 2], ")"] - obj[0].methodCall(1, 2) - 1, 2 as literal args
	[ 0, ".methodCall(", (1, 2), ")"] - (obj[0].methodCall(1), obj[0].methodCall(2)) - slice, one call of 1, other of 2
	this is so so stupid
	we're hitting the difficulties of control flow in expressions

	simplest version is definitely useful - smooth spectrum into crazy if/else flows, and I don't see an obvious boundary



	  """

	childTypes = ("[", ".") #????

	ITEM_CHILD_LIST_T = ITEM_CHILD_LIST_T
	ChildType = ChildType
	# new base class, declare new map
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	# declare abstract methods
	@classmethod
	def childObjects(cls, obj:T.Any) ->ITEM_CHILD_LIST_T:
		"""return iterable of (childObject, childType) pairs"""
		raise NotImplementedError

	@classmethod
	def newObj(cls, baseObj: T.Any, itemChildTypeList: ITEM_CHILD_LIST_T) ->T.Any:
		"""create new object from base object and child type item map,
		"""
		raise NotImplementedError("newObj not implemented "
		                          "for type", type(baseObj),
		                          " adaptor: ", cls)

class NoneVisitAdaptor(VisitAdaptor):
	forTypes = (type(None),)
	@classmethod
	def childObjects(cls, obj:T.Any) ->ITEM_CHILD_LIST_T:
		return ()
	@classmethod
	def newObj(cls, baseObj: T.Any, itemChildTypeList: ITEM_CHILD_LIST_T) ->T.Any:
		return None

class LiteralVisitAdaptor(VisitAdaptor):
	forTypes = LITERAL_TYPES
	@classmethod
	def childObjects(cls, obj:T.Any) ->ITEM_CHILD_LIST_T:
		return ()
	@classmethod
	def newObj(cls, baseObj: T.Any, itemChildTypeList: ITEM_CHILD_LIST_T) -> T.Any:
		return baseObj

class MapVisitAdaptor(VisitAdaptor):
	"""we DO need a special type for dict items, since
	otherwise we lose option to capture that relationship

	option would be nice to turn this off as vast majority of dicts
	don't needit
	"""
	forTypes = MAP_TYPES

	"""USE_TIES - should maps be considered a list of (key, value) ties,
	or literally as their basic relation?
	
	second is more literal and readable, but makes it more confusing to access
	dict keys specifically
	"""
	USE_TIES = True
	if USE_TIES:
		@classmethod
		def childObjects(cls, obj:T.Any)->ITEM_CHILD_LIST_T:
			return ((i, tie, "[") for i, tie in enumerate(obj.items()))

		@classmethod
		def newObj(cls, baseObj: T.Any, itemChildTypeList: ITEM_CHILD_LIST_T) -> T.Any:
			"""expects list of [
				( (key , value ), ChildType.MapItem)
				"""
			return type(baseObj)(i[1] for i in itemChildTypeList)

	else:
		@classmethod
		def childObjects(cls, obj:T.Any) ->ITEM_CHILD_LIST_T:
			result : ITEM_CHILD_LIST_T = [None] * len(obj) * 2
			result = []
			for k, v in obj.items():
				result.append((k, v, "["))

			# my lord,
			# result.append(( "keys()", obj.keys(), ".")) #is this legal??????
			return result

		@classmethod
		def newObj(cls, baseObj: T.Any, itemChildTypeList: ITEM_CHILD_LIST_T) -> T.Any:
			"""expects list of [
				( (key , value ), ChildType.MapItem)
				"""
			return type(baseObj)({i[0]: i[1] for i in itemChildTypeList if i[-1] == "["})

class SeqVisitAdaptor(VisitAdaptor):
	forTypes = SEQ_TYPES
	@classmethod
	def childObjects(cls, obj:T.Any) ->ITEM_CHILD_LIST_T:
		#return ((i, ChildType.SequenceElement) for i in obj)
		#return (("item", i, val) for i, val in enumerate(obj))
		return ((i, val, "[") for i, val in enumerate(obj))
	@classmethod
	def newObj(cls, baseObj: T.Any, itemChildTypeList: ITEM_CHILD_LIST_T) -> T.Any:
		#log("newObj", baseObj, itemChildTypeList)
		if isNamedTupleInstance(baseObj):
			return type(baseObj)(*(i[1] for i in itemChildTypeList))
		return type(baseObj)(i[1] for i in itemChildTypeList)

class Visitable:
	"""custom base interface for custom types -
	we associate an adaptor type for these later"""
	def childObjects(self)->ITEM_CHILD_LIST_T:
		raise NotImplementedError
	@classmethod
	def newObj(cls, baseObj, itemChildTypeList:ITEM_CHILD_LIST_T):
		raise NotImplementedError

class VisitableVisitAdaptor(VisitAdaptor):
	"""integrate derived subclasses with adaptor system
	"""
	forTypes = (Visitable,)
	@classmethod
	def childObjects(cls, obj:T.Any) ->ITEM_CHILD_LIST_T:
		return obj.childObjects()
	@classmethod
	def newObj(cls, baseObj: T.Any, itemChildTypeList: ITEM_CHILD_LIST_T) -> T.Any:
		return baseObj.newObj(baseObj, itemChildTypeList)


@dataclass
class VisitPassParams:
	"""Parametres governing single iteration of visit"""
	topDown:bool = True
	depthFirst:bool = True
	runVisitFn:bool = True # if false, only yield objects to visit
	transformVisitedObjects:bool = False # if true, modifies visited objects - yields (original, transformed) pairs
	visitFn:visitFnType = None # if given, overrides visitor's visit function
	visitKwargs:dict = None # if given, kwargs to pass to visit function
	yieldChildType:bool = False # if true, yield child type as well as child object


class VisitObjectData(TypedDict):
	base : T.Any # root object of the visit
	visitResult : T.Any # temp result of the visit
	#copyResult : T.Any # copy of final result
	#childType : ChildType.T # type of current object
	key : T.Optional[str, int]
	childType : str # type of current object
	visitPassParams : VisitPassParams # pass params for current object

	#childDatas : list[VisitObjectData] # tuple of child data for current object
	#makeNewObjFromVisitResult : bool # if true, make new object from visit result - if false, use visit result as is





class DeepVisitOp:
	"""helper class to define operations on visited objects"""

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

	Might also be useful to have a lazy generator / structure that evaluates only when pathed into?
	"""

	ChildType = ChildType
	VisitObjectData = VisitObjectData
	VisitPassParams = VisitPassParams
	DeepVisitOp = DeepVisitOp

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

	# separate method for every permutation of iteration direction - excessive but I can understand it
	def _iterRecursiveTopDownDepthFirst(self,
	                                    #parentObj:T.Any,
	                                    parentObjData:CHILD_T,
	                                    visitParams:VisitPassParams,
	                                    )->T.Generator[CHILD_T, None, None]:
		"""iterate over all objects top-down"""
		#yield parentObj
		parentKey, parentObj, parentChildType = parentObjData
		adaptorType = VisitAdaptor.adaptorForType(
			type(parentObj))
		assert adaptorType, f"no visit adaptor for type {type(parentObj)}"
		nextObjs : ITEM_CHILD_LIST_T = adaptorType.childObjects(parentObj)
		for key, nextObj, childType  in nextObjs:
			if visitParams.yieldChildType:
				yield key, nextObj, childType
			else:
				yield nextObj
			yield from self._iterRecursiveTopDownDepthFirst(
				#nextObj,
				(key, nextObj, childType),
				visitParams)

	def _iterRecursiveTopDownBreadthFirst(self,
	                                      #parentObj:T.Any,
	                                      parentObjData: CHILD_T,
	                                      visitParams:VisitPassParams,
	                                      )->T.Generator[CHILD_T, None, None]:
		"""iterate over all objects top-down"""
		parentKey, parentObj, parentChildType = parentObjData
		adaptorType = VisitAdaptor.adaptorForType(
			type(parentObj))
		assert adaptorType, f"no visit adaptor for type {type(parentObj)}"
		nextObjs : ITEM_CHILD_LIST_T = adaptorType.childObjects(parentObj)
		for key, nextObj, childType  in nextObjs:
			if visitParams.yieldChildType:
				yield key, nextObj, childType
			else:
				yield nextObj
		for key, nextObj, childType  in nextObjs:
			yield from self._iterRecursiveTopDownBreadthFirst(
				#nextObj,
				(key, nextObj, childType),
				visitParams)

	def _applyRecursiveTopDownDepthFirst(
			self,
			parentObj:T.Any,
			visitParams:VisitPassParams,
			)->T.Generator[tuple, None, None]:
		"""apply a function to all entries, yield input and result"""
		adaptorType = VisitAdaptor.adaptorForType(
			type(parentObj))
		assert adaptorType, f"no visit adaptor for type {type(parentObj)}"
		nextObjs : ITEM_CHILD_LIST_T = adaptorType.childObjects(parentObj)
		for key, nextObj, childType  in nextObjs:
			visitData = VisitObjectData(
				base=parentObj,
				visitResult=None,
				childType=childType,
				key=key,
				visitPassParams=visitParams,
			)
			yield nextObj, visitParams.visitFn(
				nextObj, self, visitData)
			yield from self._iterRecursiveTopDownDepthFirst(nextObj, visitParams)


	def _transformRecursiveTopDownDepthFirst(
			self,
			#parentObj:T.Any,
			parentObjData:CHILD_T,

			visitParams:VisitPassParams,
			)->T.Any:
		"""transform all objects top-down"""

		key, parentObj, childType = parentObjData

		# transform
		visitData = VisitObjectData(
			base=parentObj,
			visitResult=None,
			childType=childType,
			visitPassParams=visitParams,
			key=key,
		)
		result = visitParams.visitFn(
			parentObj, self, visitData)
		if result is None:
			return result
		#print("result", result)

		# get child objects
		adaptor = VisitAdaptor.adaptorForType(type(result))
		assert adaptor, f"no visit adaptor for type {type(result)}"
		nextObjs : ITEM_CHILD_LIST_T = VisitAdaptor.adaptorForType(
			type(result)).childObjects(result)
		resultObjs = []
		#for nextObj, childType in nextObjs:
		for  key, nextObj, childType in nextObjs:
			# transform child objects
			resultObjs.append(
				(key, self._transformRecursiveTopDownDepthFirst(
					(key, nextObj, childType),
				 visitParams),
				 childType)
				 )
		# create new object from transformed child objects
		adaptor = VisitAdaptor.adaptorForType(
			type(result))
		try:
			newObj = adaptor.newObj(result, resultObjs)
		except Exception as e:
			print("error making new object:")
			print("base", parentObj)
			print("result", result)
			print("resultObjs", resultObjs)
			raise e

		#print("newObj", newObj)
		return newObj

	def _transformRecursiveBottomUpDepthFirst(
			self,
			parentObjData:CHILD_T,
			visitParams:VisitPassParams,
			#childType:ChildType.T=None
			)->T.Any:
		"""transform all objects top-down"""
		key, parentObj, childType = parentObjData
		# transform
		visitData = VisitObjectData(
			base=parentObj,
			visitResult=None,
			childType=childType,
			key=key,
			visitPassParams=visitParams,
		)
		nextObjs : ITEM_CHILD_LIST_T = VisitAdaptor.adaptorForType(
			type(parentObj)).childObjects(parentObj)
		# get child objects
		resultObjs = []
		for  key, nextObj, childType in nextObjs:
			resultObjs.append(
				(key, self._transformRecursiveBottomUpDepthFirst(
					(key, nextObj, childType),
					visitParams),
									 childType)
				 )

		adaptor = VisitAdaptor.adaptorForType(parentObj)
		try:
			newObj = adaptor.newObj(parentObj, resultObjs)
		except Exception as e:
			print("Cannot make new object:")
			print("base", parentObj)
			print("resultObjs", resultObjs)
			raise e

		return visitParams.visitFn(
			newObj, self, visitData)




	def dispatchPass(self,
	                 fromObj:T.Any,
	                 passParams:VisitPassParams,
	                 visitFn:visitFnType=None,
	                 **kwargs
	                 ):
		"""dispatch a single pass of the visitor
		"""

		baseChildData = ("", fromObj, "")
		# if no function, just iterate over stuff
		if passParams.visitFn is None:
			return self._iterRecursiveTopDownDepthFirst(
				#fromObj,
				baseChildData,
				passParams)


		self.checkVisitFnSignature(passParams.visitFn)

		passParams.visitKwargs = passParams.visitKwargs or {}
		passParams.visitKwargs.update(kwargs)

		if not passParams.transformVisitedObjects: # apply function over structure
			return self._applyRecursiveTopDownDepthFirst(fromObj, passParams)

		# transform and return a new structure

		# switch top-down / bottom-up
		if passParams.topDown:
			return self._transformRecursiveTopDownDepthFirst(
				baseChildData,
				passParams,
			)
		else: # bottom-up
			return self._transformRecursiveBottomUpDepthFirst(
				baseChildData,
				passParams,
				)

visitFnType = T.Callable[
	[T.Any,
	 DeepVisitor,
	 VisitObjectData,
	 #VisitPassParams
	 ],
	T.Any]



if __name__ == '__main__':

	def printArgsVisit(obj, visitor, visitData, visitParams):
		#print(obj, visitor, visitData, visitParams)
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






