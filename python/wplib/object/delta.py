from __future__ import annotations

import typing as T

from dataclasses import dataclass, asdict

#from tree.lib.object.decorator import UserDecorator
#from tree.lib.object.framecontext import FrameContext, FrameContextVendor

#from tree.lib.object.proxy import Proxy

#from tree.signal import Signal

if T.TYPE_CHECKING:
	pass

"""base classes for tracking atomic deltas to object state
subclass for specific behaviour in user objects.

There is a WHOLE LOT of relic code here - tests for making objects automatically emit
deltas, hook up signals, inject control for delta events etc. Some of the ideas are
interesting, but if we ever need that behaviour, use this as REFERENCE ONLY. Start
from scratch with cleaner structure.


"""


class DeltaAtom:
	"""single atomic, ideally reversible change to state of object"""

	def __init__(self):
		"""add attribute to check if delta is done or undone"""
		self._done = True

	def isDone(self):
		return self._done
	def setDone(self, state:bool):
		self._done = state

	def doDelta(self, target):
		"""pass in the object to which to apply the delta
		need to return target in case of immutable types"""
		self.setDone(True)
		return target
		pass

	def undoDelta(self, target):
		"""return target in case of immutable types"""
		self.setDone(False)
		return target
		pass

	def serialise(self):
		return asdict(self)
		pass

	@classmethod
	def fromDict(cls, dataDict:dict):
		return cls(**dataDict)
		pass

	@classmethod
	def combined(cls, deltas:list[cls])->list[cls]:
		"""combine a list of deltas into as few as possible"""
		return deltas

# agnostic delta objects (should suffice for primitive types)
# we special-case string deltas later on?
primTypes = (tuple, list, set, dict)

@dataclass
class PrimDeltaAtom(DeltaAtom):
	"""container for delta functions on prim types"""

	@classmethod
	def _insertDict(cls, target:dict, key:str, index=None, value=None ):
		items = list(target.items())
		tie = (key, value)
		items.insert(index if index is not None else -1, tie)
		target.clear()  # preserve original object references
		target.update({**items})

	@classmethod
	def _insertTuple(cls, target:tuple, index:int, value)->tuple:
		l = list(target)
		l.insert(index, value)
		return type(target)(l)

	@classmethod
	def _popTuple(cls, target:tuple, index:int):
		l = list(target)
		value = l.pop(index)
		return type(target)(l), value

@dataclass
class InsertDelta(PrimDeltaAtom):
	"""insert value at key or index - signifies value has been created,
	with no prior source within delta scope"""
	key : str = None
	index : int = None
	value : object = None


	def doDelta(self, target:primTypes):
		"""this could probably be more agnostic"""

		if isinstance(target, tuple):
			target = self._insertTuple(target, self.index, self.value)
		elif isinstance(target, T.Mapping):
			self._insertDict(target, self.key, self.index, self.value )
		elif self.index is None:  # we assume set?
			target.add(self.value)
		else:
			target.insert(self.index, self.value)
		super(InsertDelta, self).doDelta(target)
		return target

	def undoDelta(self, target:primTypes):
		if isinstance(target, tuple):
			target, value = self._popTuple(target, self.index)
		elif isinstance(target, T.Mapping):
			target.pop(self.key)
		elif self.index is None: # must be set
			target.remove(self.value)
		else:
			target.pop(self.index)
		super(InsertDelta, self).undoDelta(target)
		return target


@dataclass
class MoveDelta(PrimDeltaAtom):
	"""moving value between positions, both within delta scope-
	this delta doesn't even save the value itself"""
	oldKey : str = None
	oldIndex : str = None
	newKey : str = None
	newIndex : str = None

	def doDelta(self, target:primTypes):
		if isinstance(target, tuple):
			target, value = self._popTuple(target, self.oldIndex)
			target = self._insertTuple(target, self.newIndex, value)
		elif isinstance(target, T.Mapping):
			self._insertDict(target, self.newKey,self.newIndex, target.pop(self.oldKey))
		else:
			target.insert(self.newIndex, target.pop(self.oldIndex))
		super(MoveDelta, self).doDelta(target)
		return target

	def undoDelta(self, target:primTypes):
		"""literally reverse of above"""
		if isinstance(target, tuple):
			target, value = self._popTuple(target, self.newIndex)
			target = self._insertTuple(target, self.oldIndex, value)
		elif isinstance(target, T.Mapping):
			self._insertDict(target, self.oldKey,self.oldIndex, target.pop(self.newKey))
		else:
			target.insert(self.oldIndex, target.pop(self.newIndex))
		super(MoveDelta, self).doDelta(target)
		return target




#
# class DeltaContext(FrameContext):
# 	"""we delegate responsibility to the context object to hold its
# 	own base state
#
# 	BUT do we allow the delta tracker object to EXTRACT that state?
#
# 	the delta tracker component holds one or more of these context objects,
# 	then processes their stored data when a context returns to its top level
#
# 	it's cleaner ( and maybe equally correct? idk) to put delta extraction in here?
# 	this lets us directly call stateChanged() without needing an external
# 	object
#
# 	issue is that a context is unique per object,
# 	so should ideally handle stateChanged() itself -
#
# 	while a custom tracker class might want to derive its own rules for finding deltas -
# 	but then those rules by definition impact what is considered a stateChanged() event
# 	anyway?
#
# 	context is unique per tracker
# 	for now only one all-purpose tracker per object
#
# 	technically a ui should define and attach its own tracker, but since trackers already
# 	detect separate changes in delta and state, IT'S ALL FINE
# 	MOVE ON
#
# 	"""
# 	vendor : DeltaEmittingMixin
# 	def __init__(self, vendor:DeltaEmittingMixin, name=FrameContext.DEFAULT_CONTEXT_NAME):
# 		super(DeltaContext, self).__init__(vendor, name)
#
# 		# compare states of attached structure before and after context
# 		self.baseState = None
# 		self.endState = None
# 		#self.boundStructure : DeltaEmittingMixin = None
#
# 		self._setup()
#
# 	def _setup(self):
# 		"""called when context is first created
# 		smaller, simpler version of init"""
# 		pass
#
# 	def getStructureState(self, interface:DeltaEmittingMixin):
# 		"""return a static representation of the attached structure at
# 		this point in time
#
# 		this will be used to compare start and end states
# 		"""
# 		raise NotImplementedError
#
#
# 	@classmethod
# 	def extractDeltas(cls, startState, endState, targetObject=None)->T.List[DeltaAtom]:
# 		"""structure-specific implementation to extract deltas from given states
# 		keeping this as a classmethod to encourage cleanliness
# 		"""
# 		raise NotImplementedError
#
#
# 	def onDeltasChanged(self, changeObject, deltas:list[DeltaAtom]):
# 		"""on resolution, deltasChanged is called before stateChanged"""
# 		pass
#
# 	def onStateChanged(self, changeObject):
# 		"""called whenever changeObject state has changed within context"""
# 		pass
#
#
#
# 	def onTopEnter(self):
# 		#print("delta on top enter", self.vendor)
# 		self.baseState = self.getStructureState(self.vendor)
# 		self.endState = None
#
# 	def onDeltasFound(self, deltas:list[DeltaAtom]):
# 		"""define what to do with deltas once gathered"""
# 		raise NotImplementedError
#
# 	def onTopExit(self, excType, excVal, excTb):
# 		"""check if any deltas exist between start and end states"""
# 		#print("delta on top exit")
# 		self.endState = self.getStructureState(self.vendor)
#
# 		deltas = self.extractDeltas(self.baseState, self.endState, targetObject=self.vendor)
#
# 		# if any deltas detected, delegate to function
# 		if deltas:
# 			self.onDeltasFound(deltas)
#
#
# class deltaFn(UserDecorator):
# 	"""decorator for classes to mark their own delta functions or something"""
#
# 	def wrapFunction(self, targetFunction:callable,
# 	                 decoratorArgsKwargs=(None, tuple)) ->function:
# 		"""first check if valid tracker is found
#
# 		then initialise a DeltaContext around function call
# 		"""
# 		#print("delta wrapFunction", targetFunction, decoratorArgsKwargs)
#
# 		if decoratorArgsKwargs is None:
# 			decoName = FrameContext.DEFAULT_CONTEXT_NAME
# 		else:
# 			try:
# 				decoName = decoratorArgsKwargs[1]["name"]
# 			except KeyError:
# 				raise TypeError("If called with arguments, @withFrameContext() must be passed 'name' keyword argument")
#
# 		@wraps(targetFunction)
# 		def _wrapperDeltaFn(boundSelf:DeltaEmittingMixin, *args, **kwargs):
#
# 			if args:
# 				if isinstance(args[0], FrameContextVendor) and self.boundInstance is None:
# 					self.boundInstance = args[0]
#
# 			#state = boundSelf.frameContextEnabled()
# 			#if not state: state = str(state).upper()
# 			#print("wrapper delta fn", state,  (boundSelf), targetFunction)
#
# 			if not self.boundInstance.frameContextEnabled():
# 				##print("frame context disabled on ", self.boundInstance, "for", targetFunction)
#
# 				return targetFunction(boundSelf, *args, **kwargs)
# 			with boundSelf.getFrameContext() as ctx:
# 				result = targetFunction(boundSelf, *args, **kwargs)
# 			return result
#
# 		return _wrapperDeltaFn
#
#
#
# class disableDeltas(UserDecorator):
#
# 	def wrapFunction(self, targetFunction:callable, decoratorArgsKwargs:(None, tuple)=None) ->function:
# 		"""prevent delta context from being invoked in this function"""
#
# 		@wraps(targetFunction)
# 		def _wrapperNoDeltaFn(boundSelf: DeltaEmittingMixin, *args, **kwargs):
# 			baseState = boundSelf.frameContextEnabled()
# 			boundSelf.setFrameContextEnabled(False)
# 			try:
# 				result = targetFunction(boundSelf, *args, **kwargs)
# 			finally:
# 				boundSelf.setFrameContextEnabled(baseState)
# 			return result
#
#
# """
# COMMIT vs INSTANT
#
# stateChanged() is instant, any state changed
# deltasChanged() only activates on full deltas changed
#
# """
#
#
# class DeltaEmittingMixin(FrameContextVendor):
# 	"""base class mixin for objects emitting deltas when their state changes
# 	SHOULD BE MINIMAL, do any complex processing in
# 	the Tracker class
#
# 	"""
# 	deltaFn = deltaFn
#
# 	defaultFrameContextCls = DeltaContext
# 	def __init__(self, deltasActive=False):
# 		super(DeltaEmittingMixin, self).__init__()
#
#
# 	def onFrameContextTopEnter(self, contextObj:DeltaContext):
# 		pass
#
# 	def onFrameContextTopExit(self, contextObj:DeltaContext):
# 		pass
#
# 	def onStateChanged(self):
# 		"""called whenever instant state of this object changes"""
# 		pass
#
# 	def onDeltasChanged(self):
# 		"""called whenever object deliberate deltas change"""
# 		pass
#
# 	@deltaFn
# 	def testFn(self):
# 		pass
#
# 	deltasEnabled = FrameContextVendor.frameContextEnabled
# 	setDeltasEnabled = FrameContextVendor.setFrameContextEnabled
#
#
#
# defaultMutatorNames = [
# 		'__delitem__', '__delslice__', '__iadd__', '__iand__',
# 		'__idiv__', '__idivmod__', '__ifloordiv__', '__ilshift__', '__imod__',
# 		'__imul__', '__invert__', '__ior__', '__ipow__', '__irshift__',
# 		'__isub__', '__itruediv__', '__ixor__', '__lshift__',
# 		'__reduce__',
# 		'__setitem__', '__setslice__', '__setattr__',
# 		'next',
#
# 	"append", "remove", "pop", "insert", "reverse", "clear", "update", "merge",
# 	"popitem", "setdefault",
# 	# sets
# 	"add", "discard", "intersection_update", "difference_update", "symmetric_difference_update",
# ]
#
# class DeltaHelper:
# 	""" extensible plugin-like delta support, in same vein as Serialisable
# 	intended to help cut down import spam when dealing
# 	with deltas for different types,
# 	and to allow defining deltas for custom types in uniform way"""
#
# 	def __init__(self):
# 		self.extractStateTypeFnMap : dict[tuple[type], callable] = {}
#
# 		# map of mutating functions for different types -
# 		# these can prevent firing deltas around functions that only read
# 		self.typeMutatorFnNameMap : dict[tuple[type], set[str]] = {}
# 		#self.typeDeltaMap : dict[tuple[type], tuple[T.Type[DeltaAtom]]]
# 		pass
#
# 	def registerExtractDeltaFnForTypes(self, fn:callable, types:T.Sequence[T.Type]):
# 		self.extractStateTypeFnMap[tuple(types)] = fn
#
# 	def setMutatorFnNamesForTypes(self, names:set[str], types:T.Sequence[T.Type]):
# 		self.typeMutatorFnNameMap[tuple(types)] = set(names)
#
# 	def getMutatorFnNames(self, forObject):
# 		return superClassLookup(self.typeMutatorFnNameMap, type(object), default=set(defaultMutatorNames))
#
# 	def _fallbackGetState(self, target):
# 		"""if no custom functions are given for target type state extraction,
# 		just use deepcopy"""
# 		return copy.deepcopy(target)
#
# 	def getStateForObject(self, target):
# 		"""extract static state of object - used to store a base state against
# 		which to compare deltas"""
# 		fn = superClassLookup(self.extractStateTypeFnMap, type(target), default=self._fallbackGetState)
# 		return fn(target)
#
# 	@classmethod
# 	def extractDeltasTemplateFn(cls, stateA, stateB)->list[DeltaAtom]:
# 		"""example parametre template for any registered delta extraction functions
# 		"""
# 		pass
#
# mainDeltaHelper = DeltaHelper()
#
#
# # proxy wrapper allowing extracting deltas on arbitrary primitives
# class DeltaProxy(Proxy):
# 	"""pre- and post-call hooks to run delta checks,
# 	but I don't like the idea of putting full delta infrastructure
# 	in proxy -
# 	stateChanged signal emits only (stateA, stateB)-
# 	exterior systems should accept and process this as necessary
# 	"""
#
# 	_proxyAttrs = ("_proxyStateChanged", )
#
# 	def __init__(self, obj, _proxyLink=None, proxyData=None):
# 		"""set up proxyData"""
# 		Proxy.__init__(self, obj, _proxyLink, proxyData=proxyData)
# 		self._proxyData.update( {
# 			"baseHash" : None,
# 			"baseState" : None,
# 			# allow changing this at some point per instance if needed
# 			"deltaHelper" : mainDeltaHelper,
# 			"mutatorNames" : mainDeltaHelper.getMutatorFnNames(forObject=obj),
# 			# safeguard to stop recursion
# 			"callDepth" : 0
# 		})
# 		self._proxyStateChanged = Signal()
#
#
# 	def _beforeProxyCall(self, methodName:str,
# 	                     methodArgs:tuple, methodKwargs:dict,
# 	                     targetInstance:object
# 	                     ) ->tuple[str, tuple, dict, object]:
# 		"""save base state and base hash"""
# 		if methodName in self._proxyData["mutatorNames"]:
# 			self._proxyData["callDepth"] += 1
# 			self._proxyData["baseHash"] = DeepHash(targetInstance)[targetInstance]
# 			self._proxyData["baseState"] = self._proxyData["deltaHelper"].getStateForObject(targetInstance)
#
# 		return methodName, methodArgs, methodKwargs, targetInstance
#
# 	def _onProxyCallException(self,
# 	                          methodName: str,
# 	                          methodArgs: tuple, methodKwargs: dict,
# 	                          targetInstance: object,
# 	                          exception:BaseException) ->(None, object):
# 		"""reset count if exception in function occurs"""
# 		self._proxyData["callDepth"] = 0
# 		raise exception
#
# 	def _afterProxyCall(self, methodName:str,
# 	                     methodArgs:tuple, methodKwargs:dict,
# 	                    targetInstance: object,
# 	                    callResult:object,
# 	                    ) ->object:
# 		"""run delta comparisons"""
# 		if methodName not in self._proxyData["mutatorNames"]:
# 			return callResult
#
# 		self._proxyData["callDepth"] -= 1
# 		if self._proxyData["callDepth"] == 0: # only check deltas at top frame
# 			newHash = DeepHash(targetInstance)[targetInstance]
# 			if newHash != self._proxyData["baseHash"]:
# 				# hashes differ, emit state through signal
# 				newState = self._proxyData["deltaHelper"].getStateForObject(targetInstance)
#
# 				self._proxyStateChanged.emit(self._proxyData["baseState"], newState)
#
#






