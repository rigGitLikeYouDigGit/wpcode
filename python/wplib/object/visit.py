
"""test for system recursively visiting all
items within a python structure, and optionally
modifying them in place

inspired by the ast node visitor

references to internals will almost certainly not be preserved
"""

from __future__ import annotations

import typing as T
from dataclasses import dataclass
from wplib.constant import MAP_TYPES, SEQ_TYPES

DEBUG = 0

class RecLog:
	depth = 0
	pass

def log(*msg:str):
	if DEBUG:
		print(*msg)


class Visitable:
	"""base class for types defining custom 'visitation'
	logic -
	this is only structural, how the visitor should traverse
	the object,
	and does not define the actual transformation run
	by visitor"""

	def _visitTraverse(self, masterVisitRecursiveFn,
	                   visitArgsKwargs=((), {})):
		"""return new copy of this object, with transformed
		result of the master visit function

		so for a list:
		- return [ masterVisitRecursiveFn(
			i, *visitArgsKwargs[0], **visitArgsKwargs[1]
				) ]
		"""
		raise NotImplementedError



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
	             skipLemmaSet:set[function]=None,
	             visitFn:T.Callable=None,
	             followRecursive=False,

	             ):

		# lemmas skipping object if any return true -
		# not sure if this is the best way, rather than a set of types
		self.skipLemmaSet = skipLemmaSet or set()

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
		log("  " * RecLog.depth, *msg)



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

		RecLog.depth += 1
		if RecLog.depth == 50: raise

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
		RecLog.depth -= 1
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
