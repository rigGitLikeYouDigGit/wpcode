
from __future__ import annotations
import typing as T

from types import FunctionType
from dataclasses import dataclass
from collections import deque, namedtuple
from typing import TypedDict

from wplib import Sentinel
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



from wptree import TreeInterface


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

MapItemTie = namedtuple("MapItemTie", "key value")

typeVisitDataMap = {
	MAP_TYPES : TypeVisitInfo(
		childObjectsFn = lambda obj: (i, ChildType.MapItem
		                              for i in map(MapItemTie, obj.items())),
		#childType = ChildType.MapItem,
	),
	MapItemTie : TypeVisitInfo(
		childObjectsFn = lambda obj: ((obj[0], ChildType.MapKey), (obj[1], ChildType.MapValue)),
	),

	SEQ_TYPES : TypeVisitInfo(
		childObjectsFn = lambda obj: (i, ChildType.SequenceElement for i in obj),
	),

}


typeNextFnMap = {
	MAP_TYPES: lambda obj: ((ChildType.MapItem, i) for i in obj.items()),
	# TreeInterface : lambda obj: ((ChildType.TreeBranch, i) for i in obj.branches()),
	#SEQ_TYPES: lambda obj: obj,
	LITERAL_TYPES: lambda obj: (),
}

rebuildFnMap = {
	MAP_TYPES: lambda srcObj, data: type(srcObj)(data),
	TreeInterface: lambda srcObj, data: type(srcObj).fromLiteral(data),
}

def iterDict(obj:dict):
	for childType, tie in typeNextFnMap.items():
		pass

def registerNextFnForType(type:T.Type, nextFn:FunctionType):
	typeNextFnMap[type] = nextFn

def getNextFnForType(type:T.Type)->FunctionType:
	result = superClassLookup(typeNextFnMap, type)
	if result is None:
		raise TypeError(f"no next function registered for type {type}")
	return result


def getVisitDestinations(obj:T.Any)->tuple:
	"""return a list of objects to be visited
	from the given object

	might be worth unifying with Traversable
	"""

	result = None

	if isinstance(obj, LITERAL_TYPES):
		return ()

	# elif isinstance(obj, Traversable):
	# 	return obj.next

	elif isinstance(obj, MAP_TYPES):
		return tuple(obj.items())

	elif isinstance(obj, SEQ_TYPES):
		return tuple(obj)

	raise TypeError(f"cannot visit object of type {type(obj)}")


@dataclass
class VisitObjectData:
	fromRoot:T.Any = None # root object of the visit
	visitedIds:set[int] = None # set of object ids already visited
	parentObjects:tuple = None # tuple of parent objects of current object
	parentKeys:tuple = None # tuple of keys to current object from parent objects
	childType:ChildType.T = None # type of current object


@dataclass
class VisitPassParams:
	"""Parametres governing single iteration of visit"""
	topDown:bool = True
	depthFirst:bool = True
	runVisitFn:bool = True # if false, only yield objects to visit
	copyVisitedObjects:bool = True # if false, operates in place
	transformVisitedObjects:bool = False # if true, modifies visited objects - yields (original, transformed) pairs


	pass


@dataclass
class VisitResult:
	pass




class RecursiveVisitor:
	"""base class for recursive visit and transform operations"""


	@classmethod
	def visitSingleObject(cls,
	                      obj,
	                      visitor:RecursiveVisitor,
	                      visitData:VisitObjectData=None,
	                      visitParams:VisitPassParams=None,
	                      visitResult:VisitResult=None,
	                      )->None:
		"""visit a single object - if needed,
		populate VisitResult and modify VisitParams
		"""
		return obj

	@classmethod
	def nextVisitDestinations(cls,
	                          forObj,
	                          visitor:RecursiveVisitor,
	                          visitData:T.Any=None
	                          )->T.Generator[tuple[T.Any, ChildType.T()], None, None]:
		"""yield a list of objects to be visited
		from the given object
		"""

		if isinstance(forObj, (list, tuple)):
			for i, child in enumerate(forObj):
				yield child, ChildType.SequenceElement


	def __init__(self,
	             visitSingleObjectFn:callable[
		             T.Any,
		             RecursiveVisitor,
		             VisitObjectData,
	             []]=None,
	             nextVisitDestinationsFn:callable=getVisitDestinations,
	             ):
		"""initialize the visitor with functions
		to handle visiting single objects and
		getting next objects to visit
		"""
		self.visitSingleObjectFn = visitSingleObjectFn or self.visitSingleObject
		self.nextVisitDestinationsFn = nextVisitDestinationsFn


	def _iterateTopDown(self,
	             parentObj,
	             #visitData:VisitObjectData,
	             visitParams:VisitPassParams,
	             ):

		toVisit = deque()
		toVisit.append(parentObj)

		visitData = VisitObjectData(
			fromRoot=parentObj,
			visitedIds=set(),
			parentObjects=(),
			parentKeys=(),
			childType=None)


		while toVisit:
			obj = toVisit.pop()

			# visit object
			result = self.visitSingleObjectFn(obj, self, visitData, visitParams)

			# yield visited object or (original, transformed) pair
			if visitParams.transformVisitedObjects:
				toIter = result
				yield obj, result
			else:
				toIter = obj
				yield obj

			# get next objects to visit
			nextObjs = self.nextVisitDestinationsFn(toIter, self, visitData)

			toVisit.extend(nextObjs)


	def dispatchPass(self,
	                 fromObj:T.Any,
	                 passParams:VisitPassParams,
	                 )->T.Generator[
		tuple[tuple[T.Any, ChildType.T()]],
		None,
		None]:
		"""dispatch a single pass of the visitor
		"""




		pass









