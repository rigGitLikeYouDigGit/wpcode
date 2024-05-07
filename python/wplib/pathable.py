

from __future__ import annotations
import typing as T

from wplib.object import Adaptor
from wplib.log import log
from wplib.object import DeepVisitor, VisitAdaptor
from wplib.typelib import isImmutable
from wplib import sequence, Sentinel

"""if at first
try
try
try
try
try
try"""



class Pathable(Adaptor):
	"""pathable object - can be pathed into, and can return a path.

	immutable in the sense that whenever items change, a new object is created

	'/' is valid, can be useful in string patterns, equivalent to a space between tokens
	"""
	adaptorTypeMap = Adaptor.makeNewTypeMap()

	dispatchInit = True

	keyType = T.Union[str, int]
	keyType = T.Union[keyType, tuple[keyType, ...]]

	@classmethod
	def startPath(cls, obj):
		"""might add this back to the base class,
		to dispatch the type lookup directly on base class call,
		rather than have to do Adaptor.adaptorForObject(obj)(obj) etc
		"""
		return cls.adaptorForObject(obj)(obj)

	def __init__(self, obj:dict, parent:DictPathable=None, key:T.Iterable[keyType]=None):
		"""initialise with object and parent"""
		self.parent = parent
		self.obj = obj
		self.key = list(key or [])
		#log("init", self,)# obj, parent, key)

		self.children = self._buildChildren() # only build lazily

	def __repr__(self):
		return f"<{self.__class__.__name__}({self.obj}, {self.path()})>"

	def makeChildPathable(self, key:tuple[keyType], obj:T.Any)->Pathable:
		"""make a child pathable object"""
		pathType = self.adaptorForType(type(obj))
		assert pathType, f"no path type for {type(obj)}"
		return pathType(obj, parent=self, key=key)

	def _buildChildren(self):
		raise NotImplementedError(self, f"no buildChildren")

	def objToPathKey(self, obj:T.Any)->str:
		"""convert an object to a key"""
		return str(obj)

	# this object's path
	def path(self)->list[str]:
		if self.parent is None:
			return []
		return self.parent.path() + list(self.key)

	# path access
	@classmethod
	def access(cls, obj:(Adaptor, T.Iterable[Adaptor]), path:list[keyType], one=False, default=Sentinel.FailToFind):
		"""access an object at a path
		outer function only serves to avoid recursion on linear paths -
		DO NOT override this directly, we should delegate max logic
		to the pathable objects
		"""
		toAccess = sequence.toSeq(obj)
		while path:
			token, *path = path
			# if a slice is passed toAccess may shrink or grow -
			# unsure how to handle slice access when only one branch fails
			toAccess = sequence.flatten([i._processPathToken(token) for i in toAccess])
		return toAccess


	def _processPathToken(self, token:T.Any)->T.Any:
		"""process a path token"""
		return self._childForKey(token)
	def _childForKey(self, key:str)->Pathable:
		"""get child object for key"""
		#TODO: set NotImplemented errors to flag object type and args
		raise NotImplementedError(self, f"no child for key {key}")
	def __getitem__(self, *item):
		"""get item by path"""
		return self.access(self, item)




class DictPathable(Pathable):
	"""
	list is single path
	tuple forms a slice

	("a", "b") - slice
	"(a, b)" - call an object with args?


	if obj is string, it's a path - to an IMMUTABLE object

	if any value anywhere is a string starting with "$", it's a path?

	still wrap immutables - we need to know what each entry's path is,
	and access it with other paths. ignore editing for a second

	"""
	forTypes = (dict,)



	# cache children and path

	def _buildChildren(self):
		return [(self.makeChildPathable(("keys()", i), k),
		         self.makeChildPathable((k,), v) )
		        for i, (k, v) in enumerate(self.obj.items())]

	def _childForKey(self, key:str)->Pathable:
		if key == "keys()":




class IntPathable(Pathable):
	"""this could just be a simple one
	OR
	consider the idea of indexing into an int path,
	then doing operations on it

	a/b/a * 10?
	min( $[a, b, a] , 20 ) ?
	would need a general way to do paths within expressions within paths
	"""
	forTypes = (int,)
	def _buildChildren(self):
		return []

class StringPathable(Pathable):
	forTypes = (str,)

	def _buildChildren(self):
		return []

"""2 systems needed 
- single function to path into any random object
- persistent objects that can track their own path


pathing into an object stores its path?
"""

#KEY_T = T.Union[str, int]

def pathGet(obj:T.Any, path:list[(str, int, tuple)]) -> dict[tuple[str, int], T.Any]:
	"""get an object at a path -
	tuple denotes a slice

	:return dict of path tokens to objects
	"""
	toReturn = {}
	while path:
		token, *path = path
		if isinstance(token, tuple): # output a slice of objects
			for i, t in enumerate(token):
				toReturn[i] = pathGet(obj, [t, *path])
			return toReturn
		adaptor = Pathable.adaptorForType(type(obj))
		if adaptor is None:
			raise Exception(f"No adaptor for class {type(obj)}")
		obj = adaptor._childForToken(obj, token)
	return obj

if __name__ == '__main__':

	structure = {
		1 : {"key1" : 2,
		     "key3" : {"a" : "b"},
		},
		"b" : {}
	}

	path = Pathable(structure)

	log("r1", path[1, "key1"])

	# structure = [
	# 	"a",
	# 	{"key1" : {"nestedKey1" : "nestedValue1",
	# 	           1 : "oneA",
	# 	           3 : [4, 5]
	# 	           },
	# 	 "key2" : [4, 5, 6],
	# 	 "key3" : ["zero", "oneB", "two", [4, 5]],
	# 	},
	# ]

	# r1 = pathGet(structure, [1, "key1", "nestedKey1"])
	# print(r1)
	#
	# r2 = pathGet(structure, [1, "key1", ".keys()", 0,])
	# print(r2)
	#
	# r3 = pathGet(structure, [1, ("key1", "key3"), 1])
	# print(r3)
	#
	# r4 = pathGet(structure, [1, ("key1", "key3"), 3, (1, 0)])
	# print(r4)

"""
try
try
try
try
"""