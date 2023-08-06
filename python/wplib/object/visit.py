
"""test for system recursively visiting all
items within a python structure, and optionally
modifying them in place

inspired by the ast node visitor

reworking this into smaller component functions - any purpose for recursive
visiting may want bottom-up, top-down, copying, modifying, etc.

"""

from __future__ import annotations

import typing as T
from typing import TypedDict
from dataclasses import dataclass
from wplib.constant import MAP_TYPES, SEQ_TYPES, LITERAL_TYPES
from wplib.inheritance import overrideThis

DEBUG = 0

def log(*msg:str):
	if DEBUG:
		print(*msg)

class Visitable:
	"""base class for types defining custom 'visitation'
	logic -
	this is only structural, how the visitor should traverse
	the object,

	may overlap with Traversable base class used for tree and
	path objects

	"""

	def nextObjectsToVisit(self)->tuple:
		"""return a tuple of objects to be visited
		from this object"""
		return ()

	#@overrideThis
	def copyFromVisitedObjects(self, *visitedObjs)->Visitable:
		"""return new copy of this object, using
		visited objects as arguments -
		visited objects should match those returned
		by nextObjectsToVisit()"""
		return type(self)(*visitedObjs)



def getVisitDestinations(obj:T.Any)->tuple:
	"""return a list of objects to be visited
	from the given object

	might be worth unifying with Traversable
	"""

	result = None

	if isinstance(obj, LITERAL_TYPES):
		return ()

	elif isinstance(obj, Visitable):
		return obj.nextObjectsToVisit()

	elif isinstance(obj, MAP_TYPES):
		return tuple(obj.items())

	elif isinstance(obj, SEQ_TYPES):
		return tuple(obj)

	raise TypeError(f"cannot visit object of type {type(obj)}")



def recursiveVisitCopy(obj:T.Any,
                       transformFn:T.Callable[[object, TypedDict], object],
                       visitData:(TypedDict, dict))->T.Any:
	"""return a deep copy of the given object, recursively
	visiting any objects that are Visitable.

	visitData will be copied into child items

	"""
	if isinstance(obj, Visitable):
		# get list of objects to visit
		# visit them
		# copy them
		# return new copy of this object

		results = []
		for i in obj.nextObjectsToVisit():
			newData = visitData.copy()
			results.append(recursiveVisitCopy(
				transformFn(i, newData), transformFn, newData))
		return obj.copyFromVisitedObjects(*results)

	elif isinstance(obj, MAP_TYPES):
		results = []
		for k, v in obj.items():
			kData = visitData.copy()
			vData = visitData.copy()
			results.append((
				recursiveVisitCopy(transformFn(k, kData), transformFn, kData),
				recursiveVisitCopy(transformFn(v, vData), transformFn, vData)
			))
		return type(obj)({k: v for k, v in results})

	elif isinstance(obj, SEQ_TYPES):
		results = list(obj)
		for i, v in enumerate(obj):
			newData = visitData.copy()
			results[i] = (recursiveVisitCopy(
				transformFn(v, newData), transformFn, newData))

		return type(obj)(results)

	# if object not complex, just return it
	return transformFn(obj, visitData)



class Visitor(object):
	"""more simple than it could be
	(see visit() docstring)"""

	# optionally define specific functions to be called on visiting custom types

	@dataclass
	class VisitData:
		"""data object passed in to functions during visit
		provides hook for creating new visitdata """
		parentObject : object = None
		objectPath : tuple = ()

	def __init__(self,
	             skipLemma:T.Callable[[object], bool]=None,
	             visitFn:T.Callable[[Visitor, object, object], None]=None,
	             followRecursive=False,

	             ):

		# lemmas skipping object if any return true -
		# not sure if this is the best way, rather than a set of types
		self.skipLemmaSet = skipLemma or set()

		# an instance may be passed a custom visit() function,
		# saves having to redeclare new class
		if visitFn:
			self.visit = lambda obj, parent: visitFn(self, obj, parent)


		# should recursive links be followed during visit
		self.followRecursive = followRecursive

		self._visitedObjects = set()

		# map of object id to the getitem path taken to find it
		self.idPathMap : dict[int, tuple] = {}


	def log(self, *msg):
		log(*msg)



	def getVisitData(self, newObj, parentObj, parentData:VisitData=None)->VisitData:
		"""parentData is None at top level of iteration"""
		if parentData:
			newObjPath = (*parentData.objectPath, newObj)
		else:
			newObjPath = ()
		return self.VisitData(parentObject=parentObj,
		                      objectPath=newObjPath
		                      )

	def visitRecursive(self, obj, parentObj=None, visitData:VisitData=None
	                   ):
		"""controls logic of recursive iteration
		should recurse down to leaf levels, visit them,
		then join upwards

		"""
		# visitData = visitData or self.VisitData(parentObject=None,
		#                                         objectPath=())

		self.log("visitRec", obj, visitData)

		if id(obj) in self._visitedObjects and not self.followRecursive:
			self.log("id known, skipping")
			return obj

		if any(i(obj) for i in self.skipLemmaSet):
			self.log("skipping by lemma")
			return obj

		# RecLog.depth += 1
		# if RecLog.depth == 50: raise

		# check if exact type is in primitive type sets -
		# inherited types need to be saved
		result = None
		# if isinstance(obj, Visitable):
		# 	result = obj._visitTraverse(
		# 		self.visitRecursive,
		# 		visitArgsKwargs=((obj, visitData ), {})
		# 	)
		if type(obj) in MAP_TYPES:
			result = type(obj)({ self.visitRecursive(k, obj, visitData) :
				                  self.visitRecursive(v, obj, visitData)
			                  for k, v in obj.items()})

		elif type(obj) in SEQ_TYPES:
			result = [self.visitRecursive(i, obj, visitData) for i in obj]
			result = type(obj)(result)

		if result is not None:
			obj = result

		"""visitRecursive may be called from within visit() to extend
		into complex types - do not assume that directly here
		also not sure if visitData should be generated here or before recursion
		"""

		visitData = self.getVisitData(obj, parentObj, visitData)

		result = self.visit(obj, visitData)
		self._visitedObjects.add(id(result))

		#self.log("result", result)
		#RecLog.depth -= 1
		return result

	def visit(self, obj, visitData:VisitData):
		"""logic run when visiting a single object
		if an object is returned, the return value will replace
		the original in the parent


		in mapping types, obj is a list of [key, value]
		override here
		"""

		return obj



if __name__ == '__main__':

	testStructure = [
		(0, 3, {"how" : 77, "shall": ["I"],
		        ("tup", "key") : {"set", 44} # tuple as key to a dict
		        }
		 )
	]

	newVisitor = Visitor()
	visitedStructure = newVisitor.visitRecursive(testStructure)
