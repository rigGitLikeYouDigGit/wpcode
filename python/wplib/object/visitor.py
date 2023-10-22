
from __future__ import annotations
import typing as T
from wplib import log
from types import FunctionType
from collections import defaultdict
from dataclasses import dataclass
from collections import deque, namedtuple
from typing import TypedDict

#from wplib.sentinel import Sentinel
from wplib.inheritance import superClassLookup, SuperClassLookupMap
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

	class TreeBranch(_Base):
		pass
	class TreeName(_Base):
		pass
	class TreeValue(_Base):
		pass
	class TreeAuxProperties(_Base):
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
from wptree import TreeInterface




#region get child objects

MapItemTie = namedtuple("MapItemTie", "key value")

TreeTie = namedtuple("TreeTie", "name value aux")

typeChildObjectsFnMap = {
	MAP_TYPES : lambda obj: (
			(MapItemTie(*i), ChildType.MapItem)
	             for i in obj.items()
	),
	MapItemTie : lambda obj: ((obj[0], ChildType.MapKey), (obj[1], ChildType.MapValue)),

	SEQ_TYPES : lambda obj: ((i, ChildType.SequenceElement) for i in obj),

	LITERAL_TYPES : lambda obj: (),


	TreeInterface : lambda obj: (
			(obj.name, ChildType.TreeName),
			(obj.value, ChildType.TreeValue),
			(obj.auxProperties, ChildType.TreeAuxProperties),
			*((i, ChildType.TreeBranch) for i in obj.branches)
		),

}
# endregion

# region updating
# need to return values in case of immutable types
def _updateMap(parentObj:dict, childObj:tuple, childType):
	parentObj[childObj[0]] = childObj[1]
	return parentObj

typeUpdateFnMap = {
	SEQ_TYPES : lambda parentObj, childObj, childType: parentObj + (childObj,),
	MAP_TYPES : _updateMap,
}
# endregion

# region creating new
childTypeItemMapType = T.Mapping[ChildType.T, list[T.Any]]
typeNewFnMap = {
	object : lambda baseParent, childTypeItemMap: type(baseParent)(
		childTypeItemMap.popitem()[1]),
	MAP_TYPES : lambda baseParent, childTypeItemMap: dict(childTypeItemMap[ChildType.MapItem]),

}
# endregion

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

	def registerNewFnForType(self, type, fn):
		self.typeNewFnMap.updateClassMap({type:fn})

	def getChildObjectsFnForType(self, type):
		return self.typeChildObjectsFnMap.lookup(type)

	def getUpdateFnForType(self, type):
		return self.typeUpdateFnMap.lookup(type)

	def getNewFnForType(self, type):
		return self.typeNewFnMap.lookup(type)


baseVisitTypeFunctionRegister = VisitTypeFunctionRegister(
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


@dataclass
class VisitPassParams:
	"""Parametres governing single iteration of visit"""
	topDown:bool = True
	depthFirst:bool = True
	runVisitFn:bool = True # if false, only yield objects to visit
	transformVisitedObjects:bool = False # if true, modifies visited objects - yields (original, transformed) pairs


	pass




class DeepVisitor:
	"""base class for recursive visit and transform operations"""


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


	def __init__(self,
				visitSingleObjectFn:callable[
					T.Any,
					DeepVisitor,
					VisitObjectData,
					VisitPassParams,
					[]]=None,
				 visitTypeFunctionRegister: VisitTypeFunctionRegister=baseVisitTypeFunctionRegister,
	             #nextVisitDestinationsFn:callable=None,
	             ):
		"""initialize the visitor with functions
		to handle visiting single objects and
		getting next objects to visit
		"""
		self.visitTypeFunctionRegister = visitTypeFunctionRegister
		self.visitSingleObjectFn = visitSingleObjectFn or self.visitSingleObject
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
					result = self.visitSingleObjectFn(
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
				result = self.visitSingleObjectFn(
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
		)

		toVisit.append(visitData)


		while toVisit:
			#log("toVisit", toVisit)

			visitData = toVisit.pop() #type: VisitObjectData

			# visit object
			toIter = visitData["base"]
			if visitParams.topDown:
				result = self.visitSingleObjectFn(
					visitData["base"], self, visitData, visitParams)
				visitData["visitResult"] = result

				# # yield base and result
				# yield (visitData["base"], result)
				toIter = result

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
				)
				# add ref to child data
				visitData["childDatas"].append(nextVisitData)
				# add new data to be visited next
				toVisit.append(nextVisitData)

		# reconstruct objects, bottom up
		for visitData in reversed(visited):
			visitData : VisitObjectData

			# if bottom up, do visiting now
			if not visitParams.topDown:
				result = self.visitSingleObjectFn(
					visitData["base"], self, visitData, visitParams)
				visitData["visitResult"] = result

			if not visitData['childDatas']:
				visitData['copyResult'] = visitData['visitResult']
				continue

			newObjFn = self.visitTypeFunctionRegister.getNewFnForType(
				type(visitData['visitResult'])
			)

			# build map of {childType : [ list of child objects of that type]}
			childTypeToChildObjectsMap = defaultdict(list)
			for childData in visitData['childDatas']:
				childTypeToChildObjectsMap[childData['childType']].append(childData['copyResult'])

			newObjFn : callable[[T.Any, dict[ChildType.T(), T.Any]], T.Any]
			visitData['copyResult'] = newObjFn(visitData["visitResult"],
			                                   childTypeToChildObjectsMap)

		return visited[0]['copyResult']




	def dispatchPass(self,
	                 fromObj:T.Any,
	                 passParams:VisitPassParams,
	                 ):
		"""dispatch a single pass of the visitor
		"""
		if passParams.transformVisitedObjects:
			result = self._transform(fromObj, passParams)
			return result
		else:
			yield from self._visitAll(fromObj, passParams)




		pass


def listBuildRecursive(obj:list):
	result = []
	for i in obj:
		if isinstance(i, list):
			result.append(listBuildRecursive(i))
		else:
			result.append(i + 2)
	return result


def listBuildIterative(obj:list):
	result = []
	toVisit = deque()
	toVisit.append((obj, result))
	while toVisit:
		obj, result = toVisit.pop()
		for i in obj:
			if isinstance(i, list):
				result.append([])
				toVisit.append((i, result[-1]))
			else:
				result.append(i + 2)
	return result


if __name__ == '__main__':

	def printArgsVisit(obj, visitor, visitData, visitParams):
		#print(obj, visitor, visitData, visitParams)
		print(obj, visitData["childType"])
		return obj

	visitor = DeepVisitor(
		visitTypeFunctionRegister=baseVisitTypeFunctionRegister,
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
		print("addOneTransform", obj)
		if isinstance(obj, int):
			obj += 1
		return obj

	visitor = DeepVisitor(
		visitTypeFunctionRegister=baseVisitTypeFunctionRegister,
		visitSingleObjectFn=addOneTransform)

	structure = [
		1, [2, [3, 4], 2], 1
	]
	print("structure", structure)
	newStructure = visitor._transform(structure, VisitPassParams())
	print("newStructure", newStructure)

	print("structure", structure)
	newStructure = visitor._transform(structure, VisitPassParams(
		topDown=False
	))
	print("newStructure", newStructure)



	pass






