
from __future__ import annotations
import typing as T

from copy import deepcopy
from functools import cache, cached_property
import weakref

from wplib import log, Sentinel, sequence
from wplib.object import Adaptor, TypeNamespace, HashIdElement, ObjectReference
from wplib.object.visitor import VisitAdaptor, Visitable, CHILD_LIST_T


class DexRef(ObjectReference):
	"""reference to a dex object"""
	def __init__(self, obj:WpDex, path:DexPathable.pathT=""):
		self.obj = obj
		self.path = path
	def resolve(self) ->WpDex:
		return self.obj.access(self.path)

class DexPathable:
	"""base class for objects that can be pathed to"""
	keyT = T.Union[str, int]
	keyT = T.Union[keyT, tuple[keyT, ...]]
	pathT = T.List[keyT]

	# region combine operators
	#TODO: pass in full call info to combine operators
	class Combine(TypeNamespace):
		"""operators to flatten multiple results into one"""
		class _Base(TypeNamespace.base()):
			@classmethod
			def flatten(cls, results:(list[T.Any], list[DexPathable]))->T.Any:
				"""flatten results"""
				raise NotImplementedError(cls, f"no flatten for {cls}")

		class First(_Base):
			"""return the first result"""
			@classmethod
			def flatten(cls, results:(list[T.Any], list[DexPathable]))->T.Any:
				"""flatten results"""
				return results[0]
	# endregion

	# def path(self)->pathT:
	# 	"""return path to this object"""
	# 	raise NotImplementedError(self)

	def _consumeFirstPathTokens(self, path:pathT)->tuple[list[Pathable], pathT]:
		"""process a path token"""
		raise NotImplementedError(self, f"no _processPathToken for ", path)

	@classmethod
	def access(cls, obj:(Adaptor, T.Iterable[Adaptor]), path:pathT, one:(bool, None)=True,
	           values=True, default=Sentinel.FailToFind,
	           combine:Combine.T()=Combine.First)->(T.Any, list[T.Any], Pathable, list[Pathable]):
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
				if not newPath: # terminate
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

	def ref(self, path:DexPathable.pathT="")->DexRef:
		return DexRef(self, path)

class DexValidator:
	"""base class for validating dex objects"""
	def validate(self):
		"""validate this object"""
		raise NotImplementedError(self)


class OverrideMap:
	"""sketch for storing layered overrides that can
	apply to one or more paths"""
	def __init__(self):
		self.overrides : dict[str, T.Any] = {}

	def __setitem__(self, key, value):
		self.overrides[key] = value

	def getMatches(self, seq:T.Iterable[str])->list:
		"""return all overrides that match this path"""
		return [v for k, v in self.overrides.items() if k in seq]

	def applyOverrides(self, matches, baseData:dict):
		"""no idea where this should go or how it should work -
		given base data, apply overrides to it in sequence"""



class WpDex(Adaptor,  # integrate with type adaptor system
            HashIdElement,  # each wrapper has a transient hash for caching

            # interfaces that must be implemented
            Visitable,  # compatible with visitor pattern
            DexPathable,
            #DexValidator,
            ):
	"""base for wrapping arb structure in a
	WPDex graph, allowing pathing, metadata, UI generation,
	validation, etc

	Core data in WPDex structure is IMMUTABLE - whenever anything changes,
	WPDex is regenerated. Detecting these changes from the outside is
	another issue

	Regardless of how we structure it, there's going to be a lot of
	stuff here. Do we prefer a few large blocks of stuff, or many small pieces
	of stuff

	Could split up classes per-type, per-behaviour - DictDexVisitable, ListDexValidator, etc

	let's just have single objects and interfaces - DictDex, ListDex, etc

	THIS base class collects together the interfaces that need to be fulfilled, and subclasses
	each implement them differently.
	This actually doesn't seem too bad


	for defining different behaviour in different regions of structure, different areas
	of program, etc,
	use paths to set overrides rather than defining new classes

	if these wrappers are regenerated, we shouldn't store state in them
	do we get much from having this as a separate class to DexPathable?

	"""
	# adaptor integration
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	dispatchInit = True # WPDex([1, 2, 3]) will return a specialised ListDex object

	# list of methods that can mutate state of data object
	# updated automatically with all defined setter properties
	mutatingMethodNames : set[str] = set()

	def __init__(self, obj:T.Any, parent:WpDex=None, key:T.Iterable[DexPathable.keyT]=None,
	             overrideMap:OverrideMap=None, **kwargs):
		"""initialise with object and parent"""
		# superclass inits
		HashIdElement.__init__(self)
		#Adaptor.__init__(self)

		# should these attributes be moved into pathable? probably
		self.parent = parent
		self.obj = obj
		self.key = list(key or [])

		# keep map of {child object id : WpDex holding that object}
		#self.childIdDexMap : dict[int, WpDex] = weakref.WeakValueDictionary()
		# goes against pathable to keep store of child objects
		self.childIdDexMap : dict[int, WpDex] = {}
		self.keyDexMap : dict[DexPathable.keyT, WpDex] = {}

	# 	# overrides
	# 	self._overrideMap = overrideMap
	#
	# def getOverrideMap(self)->OverrideMap:
	# 	"""get the override map for this object"""
	# 	if self._overrideMap is None:
	# 		self._overrideMap = OverrideMap()
	# 	return self._overrideMap
	#
	# def getOverrides(self, forPath=None, root=None):
	# 	"""retrieve metadata registered against a certain path
	# 	(or this object's path by default"""
	# 	path = forPath or self.path
	# 	root = root or self.root
	# 	return root.getOverrideMap().getMatches(path)



	def makeChildPathable(self, key:tuple[keyType], obj:T.Any)->WpDex:
		"""make a child pathable object"""
		pathType : type[WpDex] = self.adaptorForType(type(obj))
		assert pathType, f"no path type for {type(obj)}"
		return pathType(obj, parent=self, key=key)
		# dex = pathType(obj, parent=self, key=key)
		# self.childIdDexMap[id(obj)] = dex
		# return dex

	def _buildChildren(self)->dict[DexPathable.keyT, WpDex]:
		"""build child objects, return keyDexMap"""
		raise NotImplementedError(self, f"no _buildChildren")

	def updateChildren(self):
		self.childIdDexMap.clear()
		self.keyDexMap.clear()

		self.keyDexMap.update(self._buildChildren())
		self.childIdDexMap.update({id(v.obj) : v for v in self.keyDexMap.values()})

	def children(self)->dict[WpDex.keyT, WpDex]:
		"""return child objects
		maybe move to pathable superclass"""
		#return self._buildChildren()
		if not self.keyDexMap:
			self.updateChildren()
		return self.keyDexMap
		# return { v.key : v for k, v in self.childIdDexMap.items()
		#          if v is not None
		# }

	@cached_property
	def root(self)->WpDex:
		"""get the root
		the similarity between this thing and tree is not lost on me,
		I don't know if there's merit in looking for yet more unification
		"""
		test = self
		while test.parent:
			test = test.parent
		return test

	@cached_property
	def path(self)->list[str]:
		if self.parent is None:
			return []
		return self.parent.path + list(self.key)

	def __repr__(self):
		try:
			return f"<{self.__class__.__name__}({self.obj}, path={self.path})>"
		except:
			return f"<{self.__class__.__name__}({self.obj}, (PATH ERROR))>"


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


	# serialisation
	def asStr(self)->str:
		"""return a string representation of this object"""
		#TODO: replace with visitor for serialisation
		return str(self.obj)

