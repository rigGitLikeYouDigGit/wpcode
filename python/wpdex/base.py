
from __future__ import annotations

import copy
import pickle
import pprint
import traceback
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
from wplib.pathable import Pathable, PathAdaptor
from wplib.delta import DeltaAtom, DeltaAid, SetValueDelta

if T.TYPE_CHECKING:
	from .proxy import WpDexProxy

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
			Pathable,
            #HashIdElement,  # each wrapper has a transient hash for caching

            # interfaces that must be implemented
            Visitable,  # compatible with visitor pattern
            EventDispatcher, # can send events
            #DexPathable,

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
	objIdDexMap : dict[int, WpDex] = {}
	writeDefault = "setItem" # or "setAttr"

	parent : WpDex
	branches : list[WpDex]

	def _newPersistData(self)->dict:
		"""return a new dict to store data"""
		return {"deltaBase" : None}

	@classmethod
	def getPathAdaptorType(cls) ->type[PathAdaptor]:
		return WpDex

	def __init__(self, obj:T.Any,
	             parent:WpDex=None,
	             #name:T.Iterable[DexPathable.keyT]=None,
	             name:Pathable.keyT=None,
	             **kwargs):
		"""initialise with object and parent"""
		# superclass inits
		#HashIdElement.__init__(self)
		EventDispatcher.__init__(self)
		Pathable.__init__(self,
		                  obj,
		                  parent=parent,
		                  name=name)

		if obj is self:
			w = "a"
			raise RuntimeError("dex object is self, do not do this")

		# should these attributes be moved into pathable? probably
		# self.parent = parent
		# self.obj = obj
		# self.key = tuple(key or [])
		self.objIdDexMap[id(obj)] = self

		#self.keyDexMap : dict[DexPathable.keyT, WpDex] = {}

		# save data against paths to persist across
		# destroying and regenerating dex structure
		self._rootData : dict[tuple[str], dict[str]] = {}
		self._persistData = self._newPersistData()

		self.isPreppedForDeltas = False

		# do we build on init?
		self.updateChildren(recursive=0)

	if T.TYPE_CHECKING:
		def branches(self)->list[WpDex]:...
		def branchMap(self)->dict[Pathable.keyT, WpDex]:...

	def getValueProxy(self)->WpDexProxy:
		"""return a WpDexProxy initialised on the value of this dex
		"""
		from .proxy import WpDexProxy
		return WpDexProxy(self.obj,
		                  wpDex=self)

	def writeChildToKey(self, key:Pathable.keyT, value):
		"""OVERRIDE -
		manage the process of writing a value to child more closely
		"""
		setAttr = False # by default
		if setAttr:
			setattr(self.obj, key, value)
		else:
			self.obj[key] = value


	def write(self, value):
		"""rudimentary support for writing values back into the structure
		this is split across this method and writeChildToKey() on the parent

		if no parent, we can't really write
		"""
		log("WRITE", self, value)
		if not self.parent:
			self.setObj(value)
			return
		self.parent.prepForDeltas()

		self.parent.writeChildToKey(self.name, value)

		# trigger delta / event on parent
		#self.parent.gatherDeltas()

		# self.parent._gatherRootData()
		self.parent.updateChildren(recursive=1)

		self.parent.gatherDeltas()

		# event = {"type":"deltas",
		#          "paths" : [self.path]}
		# self.sendEvent(event)
		# self.parent._restoreChildDatasFromRoot()


	@classmethod
	def dexForObj(cls, obj)->WpDex:
		"""if object is immutable, it gets super annoying, since those won't have unique ids across all of the interpreter -
		consider passing in a known parent object to narrow down?"""
		return cls.objIdDexMap.get(id(obj))

	def _buildChildPathable(self, obj:T.Any, name:keyT)->WpDex:
		"""redeclaring default method because otherwise tracking the inheritance
		gets a bit scary"""
		if isinstance(obj, WpDex):
			return obj
		pathType : type[WpDex] = WpDex.adaptorForType(type(obj))
		assert pathType, f"no wpdex type for {type(obj)}"
		return pathType(obj, parent=self, name=name)

	def _buildBranchMap(self)->dict[Pathable.keyT, WpDex]:
		"""build child objects, return keyDexMap
		check for existing dex objects and add them if found

		overriding might be illegal - maybe dex can add additional,
		false children on top for pathing syntax"""
		#log("BUILD branch map", self, self.obj)
		#raise NotImplementedError(self, f"no _buildChildren")
		children = {}
		adaptor = VisitAdaptor.adaptorForObject(self.obj)
		assert adaptor, f"Base dex branchMap implementation relies on VisitAdaptor;\n no adaptor found for obj {self.obj}, type {type(self.obj)} "
		#log("adaptor", adaptor, self.obj)
		#raise RuntimeError
		childObjects = list(adaptor.childObjects(self.obj, {}))
		#log(     "child objects for ", self.obj, childObjects, adaptor)
		# returns list of 3-tuples (index, value, childType)
		for i, t in enumerate(childObjects):
			#if t[1] is None: continue # maybe
			key = t[0]
			foundDex = self.dexForObj(t[1])
			#log("id dex map", self.objIdDexMap)

			#log("found dex for", t[1], foundDex)
			if foundDex:
				foundDex._name = key  # repair dex name if it was previously an unparented root
				#self.addBranch(foundDex, key)
				children[key] = foundDex
				foundDex._setParent(self)
			else:
				children[key] = self._buildChildPathable(
					obj=t[1],

				name=key)
		return children

	def _gatherRootData(self):
		"""iter over all children, save persist data to this
		dex root data by path"""
		for i in self.allBranches(includeSelf=False):
			self._rootData[tuple(i.path)] = i._persistData

	def _restoreChildDatasFromRoot(self):
		"""restore child data from root data"""
		for i in self.allBranches(includeSelf=False):
			i._persistData.update(self._rootData.get(tuple(i.path), {}))


	def updateChildren(self, recursive=False):
		"""todo: this could be moved to pathable
		    but for now it's not needed
		    """
		#log("update children", self, self.obj, recursive)
		#self._gatherRootData()


		#self.branchMap().update(self._buildBranchMap())
		self.updateBranchMap()
		#self._buildBranchMap()
		for v in self.branchMap().values():
			if recursive:
				v.updateChildren(recursive=recursive)
		#self._restoreChildDatasFromRoot()

	def __repr__(self):
		if self.obj is self:
			w = "SOMETHING IS VERY WRONG"
			raise RuntimeError
		try:
			return f"<{self.__class__.__name__}({self.obj}, path={self.path})>"
		except:
			return f"<{self.__class__.__name__}({self.obj}, (PATH ERROR))>"


	# events
	def _nextEventDestinations(self,
	                           forEvent:dict,
	                           key:str)->list[EventDispatcher]:
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
		#log("handleEvent", self)
		#log(event)
		if "path" in event:
			event["path"].insert(0, self.name)
		else:
			event["path"] = []
		return super()._handleEvent(event, key)
			
	def staticCopy(self)->WpDex:
		"""return a fully separate hierarchy, wrapped in a separate
		network of WpDex objects"""
		serialParams = {"PreserveUid" : True}
		dex = WpDex(deserialise(serialise(Proxy.flatten(
			self.obj, serialParams=serialParams,
		), serialParams=serialParams),
			serialParams=serialParams
		))
		dex.updateChildren(recursive=1)
		return dex

	def prepForDeltas(self):
		"""check for deltas below this dex -
		if called by outside process, be aware that internal
		effects may also change the state of the other structure,
		so it may be best to always call this on the root"""

		# for i in self.allBranches(includeSelf=True):
		# 	log("prep deltas ", i)
		# 	i._persistData["deltaBase"] = i.getStateForDelta()
		self._persistData["deltaBase"] = self.staticCopy()
		#log("static copy", self._persistData["deltaBase"], self._persistData["deltaBase"].branches)
		self.isPreppedForDeltas = True

	def compareState(self, newDex:WpDex, baseDex:WpDex=None)->(dict, list[DeltaAtom]):
		"""by default, compare newObj against this Dex object
		TODO: still a lot to work on here with the deltas, just a first pass

		here we directly compare object values with the delta library, I
		don't think there's any point in adding extra logic in the dex layer
		"""
		baseDex = baseDex or self
		baseObj = baseDex.obj
		newObj = newDex.obj
		deltaAid : DeltaAid = DeltaAid.adaptorForType(type(baseObj))
		#log("base", baseObj, deltaAid)

		if deltaAid:
			return deltaAid.gatherDeltas(baseObj, newObj)
		log("no aid for ", newObj, DeltaAid.adaptorTypeMap)
		raise NotImplementedError()


	def gatherDeltas(self, #startState, endState
	                 emit=True,
	                 )->dict[Pathable.pathT, (list, dict)]:
		deltas = {}
		#log("GATHER", self, self.branches)
		self.isPreppedForDeltas = False
		baseState : WpDex = self._persistData["deltaBase"]
		if baseState is None:
			log("nooo")
		assert baseState is not None
		#log("base", baseState)

		# we compare FROM the base state to this dex as the live current one
		"""iter top-down -
		if type has changed, full change
		if length or number of children changed, continue going
		
		delegate to specific dex classes to process deltas between two 
		objects of their type, where they match
		"""
		#liveRoot = self.staticCopy()
		liveRoot = self
		toIter = [baseState]
		while toIter:
			baseDex = toIter.pop(0)
			basePath = tuple(baseDex.path)[1:]
			#log(" base", baseDex, basePath, baseDex.branches)
			try:
				#liveDex : WpDex = self.access(self, basePath, values=False)
				liveDex : WpDex = liveRoot.access(liveRoot, basePath, values=False)
			except (KeyError, Pathable.PathKeyError) as e: # if a path is missing
				#log("keyError", )
				#raise e
				deltas[basePath] = {"remove" : basePath}
				continue
			#log(" live", liveDex, liveDex.path, liveDex.branches)

			if not isinstance(liveDex.obj, type(baseDex.obj)):
				# type entirely changed
				deltas[basePath] = {"change" : "type"}
				"""later consider catching edge case where new value is subclass of
				original type - that way there could be a change in type, but other
				relevant deltas as well"""
				continue

			# get any added dex paths
			for i, dex in liveDex.branchMap().items():
				if not i in baseDex.branchMap():
					deltas[tuple(dex.path)] = {"added" : dex}

			try:
				#log("compare", baseDex.obj, liveDex.obj)
				itemDeltas = baseDex.compareState(liveDex)

			except NotImplementedError:
				#log("no delta compare implemented for", baseDex)
				itemDeltas = {"change" : "any"}
				if itemDeltas:
					deltas[basePath] = itemDeltas
				continue

			except Exception as e: # could be any error, serialise it here and get on with program
				itemDeltas = {"error" : traceback.format_exc(),
				              "change" : "any"}
				deltas[basePath] = itemDeltas
				continue


			if itemDeltas:
				deltas[basePath] = itemDeltas
			toIter.extend(baseDex.branches)

		#log("deltas"); pprint.pp(deltas)
		if emit:
			if deltas:
				event = {"type":"deltas",
				         "paths" : deltas}
				self.sendEvent(event)

		return deltas

	if T.TYPE_CHECKING:
		from .proxy import WX, WpDexProxy

	def ref(self, *path:WpDex.pathT
	        )->WX:
		"""convenience - sanity to get a reference to this dex
		value directly, without acrobatics through WpDexProxy
		not on the calling end at least
		"""
		proxy = self.getValueProxy()
		return proxy.ref(*path)

	# type "summary" string systems
	def getTypeSummary(self):
		"""
		return "tuple[list[(str, dict[str, int])]] = (["ey", {"a" : 2}])
		for now just do top level and string rep of what's contained -
		this is NOT suitable for eval-ing
		"""
		name = str(type(self.obj).__name__)
		return f"{name} = {str(self.obj)}"



	# serialisation
	def asStr(self)->str:
		"""return a string representation of this object"""
		#TODO: replace with visitor for serialisation
		return str(self.obj)

