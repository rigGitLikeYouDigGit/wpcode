

from __future__ import annotations

import pathlib
import typing as T

from copy import deepcopy
import fnmatch
from pathlib import Path

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



class Pathable(#Adaptor
               ):
	"""pathable object - can be pathed into, and can return a path.

	immutable in the sense that whenever items change, a new object is created
	store no state
	store a bit of state

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
	data. pathable can add extra syntax beyond the keys we get from Visitable

	Pathable is an adaptor and a valid class to inherit from in custom objects

	TODO: we specialise Pathable for primitives, but then we INHERIT from pathable in WpDex, and specialise THAT for primitives too. MADNESS
	for now use WpDex as the all-in-one toolbox wrapper

	TODO: builtins obviously don't know anything about their parent,
		so there is no concept of a consistent path for them.
		Wherever you start the path access is the root of that operation

	NO TUPLE KEYS. get that jank outta here

	renaming "key" to "name" - keeps consistent with the original tree interface
	do we rename "obj" to "value"? closer still, but less clear?
	would mean you only have to check "isinstance(branch, Tree)" to change any processing
	logic, access would be indentical

	"""

	@classmethod
	def getPathAdaptorType(cls)->type[PathAdaptor]:
		"""TODO: should we pass in the value for this here?
			unsure what the point of this method is
		"""
		return PathAdaptor

	class PathKeyError(Exception):
		pass

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

	#adaptorTypeMap = Adaptor.makeNewTypeMap()

	#dispatchInit = True # Pathable([1, 2, 3]) will return a specialised ListPathable object

	keyT = T.Union[str, int]
	#keyT = T.Union[keyT, tuple[keyT, ...]]
	pathT = T.Sequence[keyT]

	def __init__(self, obj,
	             parent:Pathable=None,
	             name:keyT=None
	             # parents:(Pathable, T.Sequence[Pathable])=None,
	             # key:(keyT, T.Sequence[keyT])=None
	             ):
		"""getters and setters here are excessive
		TODO: if you need complex logic like for tree, make properties

		this could be a full dag if we allow multiple parents for multiple
		paths leading to the same object
		LATER

		TODO: breakpoints (blast from like 5 years ago) - best left to WpDex
			dex: roots are any object with a flag set-
			pathable: _ root _ is _ root _
		"""
		#if name is None:
		#	t = name
		#assert name is not None
		self._obj = None
		self._parent = None
		self._name = None
		self.setObj(obj)
		self.setName(name)
		self._setParent(parent)

		self._overrides = {} # we're doing it

		self._branchMap = None # built on request

		self.isRoot = False #TEST for breakpoints without excessive tooling

	def __hash__(self):
		return hash((self.parent, self.keyT))

	@property
	def obj(self):
		return self._obj
	def setObj(self, obj:T.Any):
		self._obj = obj
	@property
	def parent(self)->(Pathable, None):
		return self._parent
	def _setParent(self, parent:Pathable):
		"""private as you should use addBranch to control hierarchy
		from parent to child - addBranch will call this internally"""
		self._parent = parent
	@property
	def name(self)->keyT:
		return self._name
	def setName(self, name:keyT):
		self._name = name

	#region comparing
	def __eq__(self, other):
		if not isinstance(other, Pathable):
			#raise NotImplementedError(self, other)
			return False
		return tuple(self.path) == tuple(other.path)
	#endregion

	# region display
	_autoReprAttrs = ["path"]
	def __repr__(self):
		attrDisplay = ",".join(f"{i}={getattr(self, i)}" for i in self._autoReprAttrs)
		return f"{self.__class__}({self.name}, {attrDisplay})"
	#endregion

	#region treelike methods

	def absoluteRoot(self)->Pathable:
		"""same as self.root but without
		the check for defined breakpoint"""

	# @classmethod
	# def checkIsRoot(cls, obj:Pathable):
	# 	"""even if a function, how do we work with this?
	# 	"""

	@property
	def root(self)->Pathable:
		"""get the root
		"""
		test = self
		while test.parent:
			test = test.parent
			if getattr(test, "isRoot", False):
				break
		return test

	#@classmethod
	def _buildChildPathable(self, obj:T.Any, name:keyT)->Pathable:
		if isinstance(obj, type(self)):
			return obj
		pathType : type[PathAdaptor] = type(self).adaptorForType(type(obj))
		assert pathType, f"no path type for {type(obj)}"
		return pathType(obj, parent=self, name=name)

	def _buildBranchMap(self, **kwargs) ->dict[keyT, Pathable]:
		"""return a dict of the immediate branches of this pathable, and
		their keys
		OVERRIDE if desired
		by DEFAULT we reuse logic in Visitor, but be aware the keys
		may not be the nicest
		:param **kwargs: """
		log("_buildBranchMap", self)
		adaptor = VisitAdaptor.adaptorForObject(self.obj)
		if adaptor is None:
			log("no visitAdaptor for type", self.obj, type(self.obj))
			return {}
		branches = {}
		for childData in adaptor.childObjects(
				self.obj,
				params=VisitAdaptor.PARAMS_T()):
			branches[childData[0]] = self._buildChildPathable(
				obj=childData[1], name=childData[2]
			)
		return branches

	def updateBranchMap(self, **kwargs):
		self._branchMap = self._buildBranchMap(**kwargs)
	def branchMap(self)->dict[keyT, Pathable]:
		"""get dict of immediate branches below this pathable"""
		#log("branchMap")
		if self._branchMap is None:
			self.updateBranchMap()
		return self._branchMap


	def addBranch(self, branch:Pathable, name:keyT=None):
		"""JANK as we shouldn't need to add child objects one by one,
		but there are situations in WpDex and WpDexProxy that seem to work
		better with it - leaving it for now
		don't call this from internal functions"""
		if name:
			branch.setName(name)
		name = branch.name
		assert branch.name
		self.branchMap()
		self._branchMap[name] = branch
		branch._setParent(self)



	@property
	def branches(self):
		return list(self.branchMap().values())

	def allBranches(self, includeSelf=True, depthFirst=True, topDown=True) -> list[Pathable]:
		""" returns list of all child objects
		depth first
		if not topDown, reverse final list
		we avoid recursion here, don't insert any crazy logic in this
		"""
		toIter = [self]
		found = []
		while toIter:
			current = toIter.pop(0)
			found.append(current)
			if depthFirst:
				toIter = current.branches + toIter
			else:
				toIter = toIter + current.branches
		if not includeSelf:
			found.pop(0)
		if not topDown:
			found.reverse()
		return found

	def trunk(self, includeSelf=True, includeRoot=True)->list[Pathable]:
		"""return sequence of ancestor trees in descending order to this tree"""
		branches = []
		current = self
		while current.parent:
			branches.insert(0, current)
			current = current.parent
			# check if a custom "isRoot" breakpoint attribute has been defined on it
			if getattr(current, "isRoot", False):
				break
		if includeRoot:
			branches.insert(0, current)
		if branches and not includeSelf:
			branches.pop(-1)
		return branches

	def depth(self) -> int:
		"""return int depth of this tree from root"""
		return len(self.trunk(includeSelf=True, includeRoot=False))

	@property
	def siblings(self)->list[Pathable]:
		if self.parent:
			l = self.parent.branches
			l.remove(self)
			return l
		return []

	def commonParent(self, otherBranch: Pathable)->(Pathable, None):
		""" return the lowest common parent between given branches
		or None
		if one branch is direct parent of the other,
		that branch will be returned
		"""
		#TODO: the line below is suspicious
		if self.root is not otherBranch.root:
			return None
		otherTrunk = set(otherBranch.trunk(includeSelf=True, includeRoot=True))
		test = self
		while test not in otherTrunk:
			test = test.parent
		return test

	def relativePath(self, fromBranch:Pathable)->list[str]:
		""" retrieve the relative path from the given branch to this one"""
		fromBranch = fromBranch or self.root

		# check that branches share a common tree (root)
		#print("reladdress", self, self.trunk(includeSelf=True, includeRoot=True))
		common = self.commonParent(fromBranch)
		if not common:
			raise LookupError("Branches {} and {} "
			                  "do not share a common root".format(self, fromBranch))

		addr = []
		commonDepth = common.depth()
		# parent tokens to navigate up from other
		for i in range(commonDepth - fromBranch.depth()):
			addr.append("..")
		# add address to this node
		addr.extend(
			self.path[commonDepth:])
		return addr

	# addresses
	def address(self, includeSelf=True, includeRoot=False, uid=False)->list[str]:
		"""if uid, return path by uids
		else return nice string paths
		recursive since different levels of tree might format their addresses
		differently"""
		trunk = self.trunk(includeSelf=includeSelf,
		                   includeRoot=includeRoot,
		                   )
		if uid:
			tokens = [i.uid for i in trunk]
		else:
			tokens = [i.name for i in trunk]
		return tokens


	# def stringAddress(self, includeSelf=True, includeRoot=False) -> str:
	# 	""" returns the address sequence joined by the tree separator """
	# 	trunk = self.trunk(includeSelf=includeSelf,
	# 	                   includeRoot=includeRoot,
	# 	                   )
	# 	s = ""
	# 	for i in range(len(trunk)):
	# 		s += trunk[i].name
	# 		if i != (len(trunk) - 1):
	# 			s += trunk[i].separatorChars["child"]
	#
	# 	return s


	def _ownIndex(self)->int:
		if self.parent:
			return self.parent.index(self.name)
		else: return -1

	def index(self, lookup=None, *args, **kwargs)->int:
		if lookup is None: # get tree's own index
			return self._ownIndex()
		if lookup in self.branchMap().keys():
			return list(self.branchMap().keys()).index(lookup, *args, **kwargs)
		else:
			return -1


	#endregion

	# region actual path things
	@property
	def path(self)->pathT:
		"""return path to this object"""
		if not self.parent: return []
		if self.parent is self: raise RuntimeError("PATHABLE PARENT IS SELF", self.obj)
		return self.parent.path + [self.name, ]

	def strPath(self, root=False)->str:
		tokens = [self.root.name] + list(self.path) if root else self.path
		return "/".join(map(str, tokens))

	def _consumeFirstPathTokens(self, path:pathT, **kwargs
	                            )->tuple[list[Pathable], pathT]:
		"""process a path token
		OVERRIDE to implement custom syntax - really this is the ONLY function
		that has to be swapped out.

		leave it as method on this class for now, but things like plugins for
		different syntax in different cases wouldn't be difficult
		:param **kwargs:
		"""
		token, *path = path
		if not token:
			return [self], path
		try:
			return [self.branchMap()[token]], path
		except KeyError:
			raise Pathable.PathKeyError(f"Invalid token {token} for {self} branches:\n{self.branchMap()}")

	@classmethod
	def toPath(cls, arg:(keyT, pathT))->pathT:
		"""check that given argument is a sequence"""
		if not isinstance(arg, sequence.SEQUENCE_TYPES):
			return sequence.toSeq(arg)
		return arg

	@classmethod
	def access(cls, obj:(Pathable, T.Iterable[Pathable]), path:pathT, one:(bool, None)=True,
	           values=True, default=Sentinel.FailToFind,
	           combine:Combine.T()=Combine.First,
	           **kwargs
	           )->(T.Any, list[T.Any], Pathable, list[Pathable]):
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

		:raises Pathable.PathKeyError

		TODO: how to integrate "false" children like Dict["keys()"] ?
			need to dynamically create new pathables during this call

		TODO: error management - makes sense to me to throw an error if
			a pathable gets a token it can't interpret
		"""
		# catch the case of access(obj, [])
		if not path: return obj
		path = cls.toPath(path)
		toAccess = list(sequence.toSeq(obj))
		for i, val in enumerate(toAccess):
			if not isinstance(val, (cls, Pathable)):
				toAccess[i] = cls.getPathAdaptorType()(val) # create new root objects
		# toAccess = [cls.getPathAdaptorType()(i) if not isinstance(i, Pathable) else i for i in toAccess ]
		#log("ACCESS", obj, toAccess)


		foundPathables = [] # end results of path access - unstructured
		paths = [deepcopy(path) for i in range(len(toAccess))]
		#log("access paths", paths)
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
				#log("path", path)

				####### DEFEAT ######
				path = sequence.flatten(path)

				if not path: # if you pass an empty tuple path
					foundPathables.append(pathable)
					continue

				#log((" " * (depthA + depthB)), "path iter", path, pathable)
				depthB += 1

				newPathables, newPath = pathable._consumeFirstPathTokens(path)
				#log("found", newPathables, newPath)
				# TODO: EDGE CASE when we need to wrap in a temp thing like a string,
				#  path[0] NOT GUARANTEED to be the same as the first path tokens
				newPathables = [i if isinstance(i, Pathable)
				                else cls.getPathAdaptorType()(i, parent=pathable, name=path[0])
				                for i in newPathables]
				if not newPath: # terminate
					foundPathables.extend(newPathables)
					continue
				newPaths.append(newPath)
				newToAccess.extend(newPathables)
			paths = newPaths
			toAccess = newToAccess
			#log("end paths access", paths, toAccess, foundPathables)

		# combine / flatten results
		results = foundPathables
		#log("results", results)
		# check if needed to error
		if not results:
			# format of default overrides one/many, since it's provided directly
			if default is not Sentinel.FailToFind:
				return default
			raise Pathable.PathKeyError(f"Path not found: {path}")

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

	# def ref(self, path:DexPathable.pathT="")->DexRef:
	# 	return DexRef(self, path)

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


keyT = Pathable.keyT
pathT = Pathable.pathT

class PathAdaptor(Pathable, Adaptor):
	adaptorTypeMap = Adaptor.makeNewTypeMap()

class DictPathable(PathAdaptor):
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

	def _buildBranchMap(self, **kwargs) ->dict[keyT, Pathable]:
		"""visitAdaptor creates a list of ties for dict items -
		we skip that for paths and use dict keys as normal keys,
		more intuitive that way
		:param **kwargs:
		"""

		items = { k : self._buildChildPathable(
			obj=v,
			#parent=self,
			name=k)
			for k, v in self.obj.items()}
		items["keys()"] = self._buildChildPathable(
			obj=list(self.obj.keys()),
			name="keys()",
		)
		return items

	def _consumeFirstPathTokens(self, path: pathT, **kwargs) ->tuple[list[Pathable], pathT]:
		"""process a path token
		:param **kwargs:
		"""
		token, *path = path
		if token == "keys()":
			return [list(self.obj.keys())], path
		return [self.branchMap()[token]], path


class SeqPathable(PathAdaptor):
	forTypes = (list, tuple)

class IntPathable(PathAdaptor):
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

class StringPathable(PathAdaptor):
	forTypes = (str,)

	def _buildChildren(self):
		return []

from pathlib import Path, PurePath
class PathPathable(PathAdaptor):
	"""TODO: I don't know how this should interact
	with the separate type hierarchy below
	"""
	forTypes = (PurePath, Path)
	def _buildBranchMap(self, **kwargs) ->dict[keyT, Pathable]:
		return {i : self._buildChildPathable(
			part, i
		) for i, part in enumerate(self.obj.parts)}


# class PathableVisitAdaptor(VisitAdaptor):
# 	forTypes = (Pathable, )
# 	@classmethod
# 	def childObjects(cls, obj:Pathable, params:PARAMS_T) ->CHILD_LIST_T:
# 		return [VisitAdaptor.ChildData(k, v) for k, v in obj.branchMap()] + [
# 			VisitAdaptor.ChildData("OBJ", obj.obj)
# 		]
# 		#return [VisitAdaptor.ChildData("s", str(obj), {})]
# 	@classmethod
# 	def newObj(cls, baseObj: Pathable, childDatas:CHILD_LIST_T, params:PARAMS_T) ->T.Any:
# 		raise NotImplementedError()
# 		# new = baseObj( childDatas[-1], )
# 		# return type(baseObj)(childDatas[0][1])

# TODO: gets super tangled if we start making adaptor links between the core types



# import pathlib
# getType = PathAdaptor.adaptorForType(pathlib.WindowsPath)
# assert getType

# class ObjectPathable(Pathable):
# 	forTypes = (object, )
# 	def _buildBranchMap(self) ->dict[keyT, Pathable]:
# 		return {}
### test for abstracting this to use in file folders
class DirPathable(Pathable):
	"""

	TODO: how should we integrate this with smartFolder
	"""
	
	def __init__(self, name, parent:DirPathable):
		super().__init__(obj=self, parent=parent, name=name)
		self._diskPath = self.parent.diskPath() / self.name

	def diskPath(self) -> Path:
		return Path(self._diskPath)

	# def file(self, name):
	# 	"""unsure how to do individual files"""
	#

	def _buildBranchMap(self, **kwargs) ->dict[keyT, Pathable]:
		"""look at top-level folders under this folder,
		:param **kwargs:
		"""
		children = {}
		for childDir in self.diskPath().glob("*"):

			#log("childDir", childDir)
			if not childDir.is_dir(): continue
			child = self._buildChildPathable(
				childDir, name=childDir.name)
			if child is None: continue
			children[childDir.name] = child
		return children

	def _buildChildPathable(self, obj:Path, name:keyT)->(DirPathable, None):
		"""we pass a Path object as obj, check if that should be a full
		Asset wrapper or not"""
		if not obj.is_dir(): return
		return DirPathable(name=name, parent=self)

	@classmethod
	def isValidDir(cls, path:Path):
		""" OVERRIDE
		check if the given folder is a valid source for this class
		"""
		return True

class RootDirPathable(DirPathable):
	"""same as above, but acting as a local root
	TODO: this is just temp, all tied up with how we manage local
		roots and overrides in Pathable
	"""
	def __init__(self, path:Path, name=""):
		#super().__init__(name, parent=None)
		Pathable.__init__(self, name, parent=None) # jank af
		self._diskPath = path


if __name__ == '__main__':

	s = [1, 2, 3]
	print(next(iter(s)))
	s = []
	print(next(iter(s)))

	# structure = {
	# 	1 : {"key1" : 2,
	# 	     "key3" : {"a" : "b"},
	# 	},
	# 	"b" : {}
	# }
	#
	# path = Pathable(structure)
	#
	# log("r1", path[1, "key1"])
	# log("r2", path["keys()", 0])

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