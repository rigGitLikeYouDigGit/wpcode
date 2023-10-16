
from __future__ import annotations
import typing as T
from wplib import log
from types import FunctionType
from dataclasses import dataclass
from collections import deque, namedtuple
from typing import TypedDict

#from wplib.sentinel import Sentinel
from wplib.inheritance import superClassLookup, SuperClassLookupMap
from wplib.object import TypeNamespace, Traversable
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





class TypeVisitInfo(TypedDict):
	"""information about how to visit a type"""
	childObjectsFn: callable[T.Any, [T.Iterable[T.Any, ChildType.T()]]]
	#childType: ChildType.T()

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

MapItemTie = namedtuple("MapItemTie", "key value")

TreeTie = namedtuple("TreeTie", "name value aux")

def _visitDict(obj):
	return ((MapItemTie(*i), ChildType.MapItem)
	             for i in obj.items())

typeVisitDataMap = {
	MAP_TYPES : TypeVisitInfo(
		childObjectsFn = lambda obj: (
			(MapItemTie(*i), ChildType.MapItem)
	             for i in obj.items()
		),
		# childObjectsFn = _visitDict,
		#childType = ChildType.MapItem,
	),
	MapItemTie : TypeVisitInfo(
		childObjectsFn = lambda obj: ((obj[0], ChildType.MapKey), (obj[1], ChildType.MapValue)),
	),

	SEQ_TYPES : TypeVisitInfo(
		childObjectsFn = lambda obj: ((i, ChildType.SequenceElement) for i in obj),
	),

	LITERAL_TYPES : TypeVisitInfo(
		childObjectsFn = lambda obj: (),
	),

	TreeInterface : TypeVisitInfo(
		childObjectsFn = lambda obj: (
			(obj.name, ChildType.TreeName),
			(obj.value, ChildType.TreeValue),
			(obj.auxProperties, ChildType.TreeAuxProperties),
			*((i, ChildType.TreeBranch) for i in obj.branches)
		)
	),

}

visitDataFnRegister = SuperClassLookupMap(typeVisitDataMap)

@dataclass
class VisitObjectData:
	fromRoot:T.Any = None # root object of the visit
	visitedIds:set[int] = None # set of object ids already visited
	parentObjects:tuple = None # tuple of parent objects of current object
	parentKeys:tuple = None # tuple of keys to current object from parent objects
	childType:ChildType.T = None # type of current object
	_newPassesToDispatch:tuple = () # tuple of new passes to dispatch

	def dispatchNewPass(self, passParams:VisitPassParams):
		"""dispatch new pass to visit"""
		self._newPassesToDispatch += (passParams,)


@dataclass
class VisitPassParams:
	"""Parametres governing single iteration of visit"""
	topDown:bool = True
	depthFirst:bool = True
	runVisitFn:bool = True # if false, only yield objects to visit
	copyVisitedObjects:bool = True # if false, operates in place
	transformVisitedObjects:bool = False # if true, modifies visited objects - yields (original, transformed) pairs


	pass




class RecursiveVisitor:
	"""base class for recursive visit and transform operations"""


	@classmethod
	def visitSingleObject(cls,
	                      obj,
	                      visitor:RecursiveVisitor,
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
		             RecursiveVisitor,
		             VisitObjectData,
		             VisitPassParams,
	             []]=None,
	             #nextVisitDestinationsFn:callable=None,
	             ):
		"""initialize the visitor with functions
		to handle visiting single objects and
		getting next objects to visit
		"""
		self.visitSingleObjectFn = visitSingleObjectFn or self.visitSingleObject
		#self.nextVisitDestinationsFn = nextVisitDestinationsFn or self.nextVisitDestinations


	def _iterateTopDown(self,
	             parentObj,
	             visitParams:VisitPassParams,
	             ):
		"""iterate over objects"""

		toVisit = deque()
		#toVisit.append(parentObj)

		visitData = VisitObjectData(
			fromRoot=parentObj,
			visitedIds=set(),
			parentObjects=(),
			parentKeys=(),
			childType=None)

		toVisit.append((parentObj, visitData))


		while toVisit:
			#log("toVisit", toVisit)

			obj, visitData = toVisit.pop() #type: T.Any, VisitObjectData

			# visit object
			result = self.visitSingleObjectFn(obj, self, visitData, visitParams)

			visitData.visitedIds.add(id(obj))
			visitData.parentObjects += (obj,)

			# yield visited object or (original, transformed) pair
			if visitParams.transformVisitedObjects:
				toIter = result
				#yield obj, result
			else:
				toIter = obj
				#yield obj

			# get next objects to visit
			typeVisitInfo : TypeVisitInfo = visitDataFnRegister.lookup(type(toIter))
			childObjectsFn = typeVisitInfo["childObjectsFn"]
			nextObjTies = childObjectsFn(toIter)
			#print("nextObjTies", nextObjTies, type(nextObjTies))

			# build new visit datas for next objects
			for nextObj, childType in nextObjTies:
				nextVisitData = VisitObjectData(
					fromRoot=visitData.fromRoot,
					visitedIds=visitData.visitedIds,
					parentObjects=tuple(visitData.parentObjects),
					parentKeys=tuple(visitData.parentKeys),
					childType=childType,
				)

				toVisit.append((nextObj, nextVisitData))


	def dispatchPass(self,
	                 fromObj:T.Any,
	                 passParams:VisitPassParams,
	                 )->T.Generator[
		tuple[tuple[T.Any, ChildType.T()]],
		None,
		None]:
		"""dispatch a single pass of the visitor
		"""
		#yield from self._iterateTopDown(fromObj, passParams)
		print("dispatchPass", fromObj, passParams)
		result = self._iterateTopDown(fromObj, passParams)
		print("result", result)




		pass


if __name__ == '__main__':

	def printArgsVisit(obj, visitor, visitData, visitParams):
		#print(obj, visitor, visitData, visitParams)
		print(obj, visitData.childType)
		return obj

	visitor = RecursiveVisitor(visitSingleObjectFn=printArgsVisit)

	structure = {
		"key1": "value1",
		(2, 4, "fhffhs"): ["value2", [], 3, 4, 5],
		"key3": "value3",
	}

	visitor.dispatchPass(structure, VisitPassParams())



	pass






