
from __future__ import annotations

import copy
import pickle
import typing as T

from copy import deepcopy
from collections import defaultdict
from functools import cache, cached_property
import weakref

import deepdiff

from wplib import log, Sentinel, sequence
from wplib.object import Adaptor, TypeNamespace, HashIdElement, ObjectReference, EventDispatcher
from wplib.serial import serialise, deserialise
from wplib.object.visitor import VisitAdaptor, Visitable, CHILD_LIST_T, DeepVisitor
from wplib.object.proxy import Proxy, FlattenProxyOp


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

	def path(self)->pathT:
		"""return path to this object"""
		raise NotImplementedError(self)

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
		# catch the case of access(obj, [])
		if not path: return obj
		toAccess = sequence.toSeq(obj)
		#log("ACCESS", obj, toAccess)



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
            EventDispatcher, # can send events
            DexPathable,
            #DexValidator,
            ):
	"""base for wrapping arb structure in a
	WPDex graph, allowing pathing, metadata, UI generation,
	validation, etc

	define a separate adaptor for each primitive type, and implement
	all special treatment there

	if we wanted to properly police a structure, we might do a kind of
	client/server model, where a WpDex sends the desired change and path to
	a server through events, and a new event is sent back with the result
	and permissions?
	YES this way we avoid storing parametres and state in wpdex
	the ROOT of any wpdex hierarchy is considered the manager
	so data has to persist and maintain a link to its object, but logic/interface doesn't

	TODO:
	 to gather deltas, not enough to compare serialised dictionaries, and it's
	 restrictive to only rely on child objects from visitable -
	 consider copying out a static "dissected" version of the hierarchy, held entirely
	 in the dex structure?

	"""
	# adaptor integration
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	dispatchInit = True # WPDex([1, 2, 3]) will return a specialised ListDex object

	# list of methods that can mutate state of data object
	# updated automatically with all defined setter properties
	mutatingMethodNames : set[str] = {
		"__setattr__", "__setitem__", "__iadd__", "__imul__", "__delitem__", "__delattr__", "__setslice__",

		"insert", "append", "extend", "pop", "remove", "clear", "reverse", "sort",
		"update", "add", "discard",
		"split",
	}

	def _newPersistData(self)->dict:
		"""return a new dict to store data"""
		return {"deltaBase" : None}

	def __init__(self, obj:T.Any, parent:WpDex=None, key:T.Iterable[DexPathable.keyT]=None,
	             overrideMap:OverrideMap=None, **kwargs):
		"""initialise with object and parent"""
		# superclass inits
		HashIdElement.__init__(self)
		EventDispatcher.__init__(self)

		# should these attributes be moved into pathable? probably
		self.parent = parent
		self.obj = obj
		self.key = list(key or [])

		# keep map of {child object id : WpDex holding that object}
		# goes against pathable to keep store of child objects
		self.childIdDexMap : dict[int, WpDex] = {}
		self.keyDexMap : dict[DexPathable.keyT, WpDex] = {}

		# save data against paths to persist across
		# destroying and regenerating dex structure
		self._rootData : dict[tuple[str], dict[str]] = {}
		self._persistData = self._newPersistData()

		self.isPreppedForDeltas = False

		# do we build on init?
		#self.updateChildren()


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

	def _gatherRootData(self):
		"""iter over all children, save persist data to this
		dex root data by path"""
		for i in self.allBranches(includeSelf=False):
			self._rootData[tuple(i.path)] = i._persistData

	def _restoreChildDatasFromRoot(self):
		"""restore child data from root data"""
		for i in self.allBranches(includeSelf=False):
			i._persistData.update(self._rootData.get(tuple(i.path), {}))

	def updateChildren(self):
		self._gatherRootData()
		self.childIdDexMap.clear()
		self.keyDexMap.clear()

		self.keyDexMap.update(self._buildChildren())
		self.childIdDexMap.update({id(v.obj) : v for v in self.keyDexMap.values()})
		for v in self.keyDexMap.values():
			self.childIdDexMap.update(v.childIdDexMap)
		self._restoreChildDatasFromRoot()


	# TODO TODO TODO TODO TODO
	#  UNIFY THIS WITH TREE AND PATHABLE

	def children(self)->dict[WpDex.keyT, WpDex]:
		"""return child objects
		maybe move to pathable superclass"""
		#return self._buildChildren()
		# if not self.keyDexMap:
		# 	self.updateChildren()
		return self.keyDexMap
		# return { v.key : v for k, v in self.childIdDexMap.items()
		#          if v is not None
		# }

	@property
	def branches(self):
		return list(self.children().values())

	def allBranches(self, includeSelf=True, depthFirst=True, topDown=True)->list[WpDex]:
		""" returns list of all child objects
		depth first
		if not topDown, reverse final list
		we avoid recursion here, don't insert any crazy logic in this
		"""

		#found = [ self ] if includeSelf else []
		toIter = [ self ]
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

	def trunk(self, includeSelf=True, includeRoot=True)->list[WpDex]:
		"""return sequence of ancestor trees in descending order to this tree"""
		branches = []
		current = self
		while current.parent:
			branches.insert(0, current)
			current = current.parent
		if includeRoot:
			branches.insert(0, current)
		if branches and not includeSelf:
			branches.pop(-1)
		return branches


	def commonParent(self, otherBranch: TreeType)->(TreeType, None):
		""" return the lowest common parent between given branches
		or None
		if one branch is direct parent of the other,
		that branch will be returned
		"""
		# #print("commonParent")
		if self.root is not otherBranch.root:
			return None
		otherTrunk = set(otherBranch.trunk(includeSelf=True, includeRoot=True))
		# otherTrunk.add(otherBranch)
		test = self
		while test not in otherTrunk:
			test = test.parent
		return test

	def relativePath(self, fromDex:WpDex)->list[str]:
		""" retrieve the relative path from the given branch to this one"""
		fromDex = fromDex or self.root

		# check that branches share a common tree (root)
		#print("reladdress", self, self.trunk(includeSelf=True, includeRoot=True))
		common = self.commonParent(fromDex)
		if not common:
			raise LookupError("Branches {} and {} "
			                  "do not share a common root".format(self, fromDex))

		addr = []
		commonDepth = common.depth()
		# parent tokens to navigate up from other
		for i in range(commonDepth - fromDex.depth()):
			addr.append("..")
		# add address to this node
		addr.extend(
			self.path[commonDepth:])
		return addr




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

	# events
	def _nextEventDestinations(self, forEvent:EventBase, key:str)->list[EventDispatcher]:
		"""
		OVERRIDE
		return a list of objects to pass this event
		unsure if we should allow filtering here (parent deciding
		which child should receive event)
		"""
		if self.parent:
			return [self.parent]
		return []

	def _handleEvent(self, event:dict, key:str= "main"):
		"""test a more flexible way of identifying by paths -
		as an event propagates from its source dex, it prepends its
		own key to the event path

		that way it even works for a diamond shape, without worrying
		about unique vs alias paths, I think it's cool"""
		log("handleEvent", self)
		log(event)
		if "path" in event:
			event["path"].insert(0, self.key)
		else:
			event["path"] = []
		return super()._handleEvent(event, key)
			
	def staticCopy(self)->WpDex:
		"""return a fully separate hierarchy, wrapped in a separate
		network of WpDex objects"""
		return WpDex(deserialise(serialise(Proxy.flatten(
			self.obj
		), serialParams={"PreserveUid" : True}),
			serialParams={"PreserveUid" : True}
		))


	def getStateForDelta(self)->dict:
		"""return a representation suitable to extract deltas
		doing this all the way down gives obscene slowdown,
		every wpdex has to recurse into the entire data
		structure below it

		remove any proxies from structure first?
		"""
		#state = serialise(self.obj)
		state = serialise(Proxy.flatten(self.obj),
		                  serialParams={"PreserveUid" : True})
		#log(" getState", self)
		#log(state)
		return state

	def extractDeltas(self, baseState, endState)->list[dict]:
		"""
		extract changes between two states
		if list is empty, no changes done -

		should compare only this level of dex, children will extract
		their own deltas and emit events with their own path
		"""
		log("extract deltas", self.path, "states:", )
		log(baseState)
		log( endState)
		if baseState is None:
			return [{"added" : endState}]

		# check for what could have changed - might switch to deepdiff here,
		# as fully diffing 2 objects is very involved
		# changes = []
		# baseMap = {i[0] : i[1] for i in VisitAdaptor.adaptorForObject(baseState).childObjects(baseState, {})}
		if pickle.dumps(baseState) != pickle.dumps(endState):
			return [{"change": None}]
		return []

	def prepForDeltas(self):
		"""check for deltas below this dex -
		if called by outside process, be aware that internal
		effects may also change the state of the other structure,
		so it may be best to always call this on the root"""

		# for i in self.allBranches(includeSelf=True):
		# 	log("prep deltas ", i)
		# 	i._persistData["deltaBase"] = i.getStateForDelta()
		self._persistData["deltaBase"] = self.staticCopy()
		self.isPreppedForDeltas = True

	def compareState(self, newObj, baseObj=None):
		"""by default, compare newObj against this Dex object"""
		raise NotImplementedError

	def gatherDeltas(self, #startState, endState
	                 ):
		deltas = {}
		log("GATHER", self)
		self.isPreppedForDeltas = False
		baseState : WpDex = self._persistData["deltaBase"]
		assert baseState is not None
		log("base", baseState)

		# we compare FROM the base state to this dex as the live current one
		"""iter top-down -
		if type has changed, full change
		if length or number of children changed, continue going
		
		delegate to specific dex classes to process deltas between two 
		objects of their type, where they match
		"""
		toIter = [baseState]
		while toIter:
			baseDex = toIter.pop(0)
			basePath = tuple(baseDex.path)
			try:
				liveDex = self.access(self, basePath, values=False)
			except KeyError: # if a path is missing
				deltas[basePath] = {"remove" : basePath}
				continue

			if not isinstance(liveDex.obj, type(baseDex.obj)):
				# type entirely changed
				deltas[basePath] = {"change" : "type"}
				"""later consider catching edge case where new value is subclass of
				original type - that way there could be a change in type, but other
				relevant deltas as well"""
				continue

			try:
				log("compare", baseDex.obj, liveDex.obj)
				deltas[basePath] = baseDex.compareState(liveDex.obj)
			except NotImplementedError:
				log("no delta compare implemented for", self)
				deltas[basePath] = {"change" : "any"}
				continue

			toIter.extend(baseDex.branches)



		return deltas

	# serialisation
	def asStr(self)->str:
		"""return a string representation of this object"""
		#TODO: replace with visitor for serialisation
		return str(self.obj)

