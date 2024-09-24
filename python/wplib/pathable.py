

from __future__ import annotations
import typing as T

from copy import deepcopy
import fnmatch

from wplib.object import Adaptor
from wplib.log import log
from wplib.object import DeepVisitor, VisitAdaptor
from wplib.typelib import isImmutable
from wplib import sequence, Sentinel, TypeNamespace

"""if at first
try
try
try
try
try
try

but I think we're finally getting somewhere

move this to lib.object
"""



class Pathable(Adaptor):
	"""pathable object - can be pathed into, and can return a path.

	immutable in the sense that whenever items change, a new object is created

	'/' is valid, can be useful in string patterns, equivalent to a space between tokens

	should the same path op be able to return a single object and/or a list of objects?
	path[0] -> single object
	path[:] -> list of objects
	access flag is not optional - one=true returns a single result, one=false wraps it in a list
	no no COMBINE results based on given operators - First by default

	main pathing logic for slicing, wildcarding etc should work the same across
	any class - searching requires all entries to be known for linux-style
	root/**/leaf recursive

	how would we manage multiple wildcards in the same query, that might
	only be defined by the adaptor for one object type?

	root/**/branch/array[:4]/leaf ?
	where only the ArrayPathable adaptor has any idea what slicing means


	NOW we can start unifying all our path stuff -
	classmethod access() can look up the right adaptor class to manage the given object

	pathable handles accessing, data structures like tree deal with holding
	data


	"""

	class Combine(TypeNamespace):
		"""operators to flatten multiple results into one"""
		class _Base(TypeNamespace.base()):
			@classmethod
			def flatten(cls, results:(list[T.Any], list[Pathable]))->T.Any:
				"""flatten results"""
				raise NotImplementedError(cls, f"no flatten for {cls}")

		class First(_Base):
			"""return the first result"""
			@classmethod
			def flatten(cls, results:(list[T.Any], list[Pathable]))->T.Any:
				"""flatten results"""
				return results[0]

	adaptorTypeMap = Adaptor.makeNewTypeMap()

	dispatchInit = True # Pathable([1, 2, 3]) will return a specialised ListPathable object

	keyT = T.Union[str, int]
	keyT = T.Union[keyT, tuple[keyT, ...]]
	pathT = T.List[keyT]


	@classmethod
	def searchChars(cls)->list[str]:
		"""return characters to look for in queries that
		require"""


	def __init__(self, obj, parent:DictPathable=None, key:T.Iterable[keyType]=None):
		"""initialise with object and parent"""
		self.parent = parent
		self.obj = obj
		self.key = list(key or [])
		#log("init", self,)# obj, parent, key)

		self.children = self._buildChildren() # only build lazily

	def __repr__(self):
		try:
			return f"<{self.__class__.__name__}({self.obj}, {self.path()})>"
		except:
			return f"<{self.__class__.__name__}({self.obj}, (PATH ERROR))>"

	@property
	def root(self)->Pathable:
		"""get the root
		the similarity between this thing and tree is not lost on me,
		I don't know if there's merit in looking for yet more unification
		"""
		test = self
		while test.parent:
			test = test.parent
		return test

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
	def access(cls, obj:(Adaptor, T.Iterable[Adaptor]), path:pathT, one:(bool, None)=True,
	           values=True, default=Sentinel.FailToFind,
	           combine:Combine.T()=Combine.First)->(T.Any, list[T.Any],
	                                                Pathable, list[Pathable]):
		"""access an object at a path
		outer function only serves to avoid recursion on linear paths -
		DO NOT override this directly, we should delegate max logic
		to the pathable objects

		feed copy of whole path to each pathable object - let each consume the
		first however many tokens and return the result

		track which object returned which path, and match the next chunk of path
		to the associated result

		if values, return actual result values
		if not, return Pathable objects


		"""
		toAccess = sequence.toSeq(obj)

		foundPathables = [] # end results of path access - unstructured
		paths = [deepcopy(path) for i in range(len(toAccess))]
		depthA = 0
		while paths:
			#newPaths = [None] * len(toAccess)
			newPaths = []
			newToAccess = []
			#log( " "* depthA + "outer iter", paths)
			depthA += 1
			depthB = 1
			for pathId, (path, pathable) \
					in enumerate(zip(paths, toAccess)):

				#log((" " * (depthA + depthB)), "path iter", path, pathable)
				depthB += 1

				newPathables, newPath = pathable._consumeFirstPathTokens(path)
				if not newPath:
					foundPathables.extend(newPathables)
					continue
				newPaths.append(newPath)
				newToAccess.extend(newPathables)
			paths = newPaths
			toAccess = newToAccess

		# combine / flatten results
		results = foundPathables
		# check if needed to error
		if not results:
			# format of default overrides one/many, since it's provided directly
			if default is not Sentinel.FailToFind:
				return default
			raise KeyError(f"Path not found: {path}")

		if values:
			results = [r.obj for r in results]

		if one is None: # shape explicitly not specified, return natural shape of result
			# why would you ever do this
			if len(results) == 1:
				return results[0]
			return results

		if one:
			return combine.flatten(results)
		return results


	def _consumeFirstPathTokens(self, path:pathT)->tuple[list[Pathable], pathT]:
		"""process a path token"""
		raise NotImplementedError(self, f"no _processPathToken for ", path)

	def __getitem__(self, item):
		"""get item by path -
		getitem is faster and less safe version of access,
		may variably return a single result or a list from a slice -
		up to user to know what they're putting.

		If any part of the path is not known, and may contain a slice,
		prefer using access() to define return format explicitly
		"""
		#log("getitem", item)

		return self.access(self, list(sequence.toSeq(item)), one=None)




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

	def _buildChildren(self):
		# return [(self.makeChildPathable(("keys()", ), k),
		#          self.makeChildPathable((k,), v) )
		#         for i, (k, v) in enumerate(self.obj.items())]
		items = { k : self.makeChildPathable((k,), v) for k, v in self.obj.items()}
		items["keys()"] = self.makeChildPathable(("keys()",), list(self.obj.keys()))
		return items

	def _consumeFirstPathTokens(self, path:pathT) ->tuple[list[Pathable], pathT]:
		"""process a path token"""
		token, *path = path
		if token == "keys()":
			return [self.children["keys()"]], path
		return [self.children[token]], path


class SeqPathable(Pathable):
	forTypes = (list, tuple)
	def _buildChildren(self):
		return [self.makeChildPathable((i,), v) for i, v in enumerate(self.obj)]
	def _consumeFirstPathTokens(self, path:pathT) ->tuple[list[Pathable], pathT]:
		"""process a path token

		how to do slicing, if a slice is given as
		"array[2:5]"
		or "array", "2:5"

		SEPARATE matching and checking against path, IN SEQUENCE for each
		separate path token?

		a/b/**/e/f[2:5]/g/**/leaf

		return all paths matching a
		of those, all then matching b
		of those, all matching a big wildcard

		3 values for each anim clip
		weight, speed, offset

		consider single little widget for each? for later,
		channelbox is cool for now
		literally change speed by visualising length of clip as a box,
		offset by moving clip back and forth,
		careful that it doesn't confuse with the actual graph profile of the attributes,
		don't try and represent timing, blending etc

		"""
		token, *path = path
		# if isinstance(token, int):
		# 	return [self.children[token]], path
		return [self.children[token]], path



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



if __name__ == '__main__':

	structure = {
		1 : {"key1" : 2,
		     "key3" : {"a" : "b"},
		},
		"b" : {}
	}

	path = Pathable(structure)

	log("r1", path[1, "key1"])
	log("r2", path["keys()", 0])

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