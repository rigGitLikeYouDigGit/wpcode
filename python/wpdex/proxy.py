
from __future__ import annotations

import copy
import typing as T

from collections import defaultdict, namedtuple
import weakref

from deepdiff import Delta, DeepDiff, DeepHash

from wplib import log
from wplib.object import DeltaAtom, Proxy, ProxyData
from wplib.constant import SEQ_TYPES, MAP_TYPES, STR_TYPES, LITERAL_TYPES, IMMUTABLE_TYPES
from wplib.sentinel import Sentinel

from wpexp.match import stringMatch

from wpdex.base import WpDex



"""
test for wrapping a WHOLE datastructure
in layers of proxy objects

purpose is to watch for changes - either simple events or full deltas


copied the old proxy system from wplib for testing - once stable
maybe combine them somehow

myStruct = [ 1, [ 2, [ 3, 4 ] ] ]
proxy = WpDexProxy(myStruct)

myStruct[1][1] -> [3, 4] (list)
proxy[1][1] -> WpDexProxy([3, 4])


"""

"""
TODO:
might be cleaner to emit events into an intermediate queue,
and then start working through them after the call stack is clear - 
that could also be a way to async the ui updates.

for now the tight coupling seems to work
"""

class Reactive:
	"""

	bind(source, target)
	not as flashy as wrapping every function to watch for calling
	with a live source, but more readable


	TODO: check if it's more work to just use Param for
	 this
	 -
	 Param's main model of declaring class-level attributes on objects
	 doesn't seem extensible enough for the flexible params in chimaera -
	 the reactive stuff is rad though

	"""

class WpDexProxyData(ProxyData):
	parent : weakref.ref
	wpDex : WpDex
	deltaStartState : T.Any
	deltaCallDepth : int

class WpDexProxy(
    Proxy
            ):
	"""
	proxy object for watching changes to a data structure
	initialise at top level of structure to watch -
	accessing structure through proxy returns a fully wrapped
	WpDex structure
	"""
	_classProxyCache = defaultdict(dict) # { class : { class cache } }
	_objProxyCache = {} #NB: weakdict was giving issues, this might chug memory
	_proxyAttrs = ("_proxyWpDex", "_proxyDeltaStartState")

	_proxyData : WpDexProxyData


	def __init__(self, obj, proxyData: WpDexProxyData=None,
	             wpDex:WpDex=None, parentDex:WpDex=None,
	             **kwargs):
		"""create a WpDex object for this proxy, add
		weak sets for children and parent"""
		#log("super cls", self._proxyParentCls, self._proxySuperCls)
		self._proxySuperCls.__init__(self, obj, proxyData, **kwargs)


		self._proxyData["deltaStartState"] = None
		self._proxyData["deltaCallDepth"] = 0

		# wpdex set up
		if wpDex is not None: # pre-built dex passed in
			self._proxyData["wpDex"] = wpDex
		else:
			self._proxyData["wpDex"] = WpDex(obj)
			# link parent dex if given
		if parentDex is not None:
			self.dex().parent = parentDex

	def dex(self)->WpDex:
		return self._proxyData["wpDex"]

	def _openDelta(self):
		"""open a delta for this object -
		if mutating functions are reentrant, only open one delta"""
		if self._proxyData["deltaCallDepth"] == 0:
			self._proxyData["deltaStartState"] = copy.deepcopy(self._proxyData["target"])
		self._proxyData["deltaCallDepth"] += 1

	def _emitDelta(self):
		"""emit a delta for this object"""
		self._proxyData["deltaCallDepth"] -= 1
		if self._proxyData["deltaCallDepth"] == 0:
			# delta = DeltaAtom(self._proxyData["deltaStartState"], self._proxyData["target"])
			self._proxyData["deltaStartState"] = None
			# send out delta event
			# TODO: make an actual delta event
			self.dex().sendEvent({"type":"delta", "delta":None,
			                      "path" : self.dex().path
			                      })


	def _beforeProxyCall(self, methodName:str,
	                     methodArgs:tuple, methodKwargs:dict,
	                     targetInstance:object
	                     ) ->tuple[T.Callable, tuple, dict, object]:
		"""open a delta if mutating method called"""
		#log(f"before proxy call {methodName}, {methodArgs, methodKwargs}", vars=0)
		fn, args, kwargs, targetInstance = super()._beforeProxyCall(methodName, methodArgs, methodKwargs, targetInstance)

		if not self.dex().childIdDexMap:
			self.dex().updateChildren()

		# check if method will mutate data - need to open a delta
		if methodName in self.dex().mutatingMethodNames:
			self._openDelta()

		return fn, args, kwargs, targetInstance

	def _afterProxyCall(self, methodName:str,
	                    method:T.Callable,
	                     methodArgs:tuple, methodKwargs:dict,
	                    targetInstance: object,
	                    callResult:object,
	                    ) ->object:
		"""return result wrapped in a wpDex proxy, if it appears in
		main dex children
		"""
		#log(f"after proxy call {methodName}, {methodArgs, methodKwargs}", vars=0)
		callResult = super()._afterProxyCall(methodName, method, methodArgs, methodKwargs, targetInstance, callResult)
		toReturn = callResult

		# if mutating method called, rebuild WpDex children
		if methodName in self.dex().mutatingMethodNames:
			self.dex().updateChildren()

		#log("wp children", self.dex().childIdDexMap)
		#log(self.dex(), self.dex().obj)

		# only wrap containers, not literals?
		if id(callResult) in self.dex().childIdDexMap:
			if isinstance(callResult, LITERAL_TYPES) or callResult is None:
				pass
			else:
				proxy = WpDexProxy(callResult, parent=self,
				                   wpDex=self.dex().childIdDexMap[id(callResult)],
				                   parentDex=self.dex()
				                   )
				toReturn = proxy

		# if mutating method called, finallyemit delta
		if methodName in self.dex().mutatingMethodNames:
			self._emitDelta()

		return toReturn

	def _onProxyCallException(self,
	                          methodName: str,
	                          method:T.Callable,
	                          methodArgs: tuple, methodKwargs: dict,
	                          targetInstance: object,
	                          exception:BaseException) ->(None, object):
		"""ensure we still emit / clear deltas if an exception is raised"""
		log(f"on proxy call exception {methodName}, {methodArgs, methodKwargs}")
		if methodName in self.dex().mutatingMethodNames:
			self._emitDelta()
		raise exception


	# reactive
	def bind(self, dataPath:str, target:callable, callNow=True):
		"""bind a reactive target to a path in this object -
		when the path data changes, target is called with
		that data"""

		def onEvent(event):
			print("onEvent", event)
			print("dataPath", dataPath)
			print("dataToSet", self.dex().access(self.dex(), dataPath))
			print("sender rel path", event["sender"].relativePath(fromDex=self.dex()))

			# check that the relative path is equal or less than the data path
			# if event["sender"] == dataPath:
			target(self.dex().access(self.dex(), dataPath,
			                         values=True, one=True
			                         )
			       )

		self.dex().getEventSignal("main", create=True).connect(onEvent)

		if callNow:
			target(self.dex().access(self.dex(), dataPath,
			                         values=True, one=True
			                         )
			       )


		pass




if __name__ == '__main__':

	s = [1, [2, [3, 4]]]
	proxy = WpDexProxy(s)

	print(type(proxy), isinstance(proxy, Proxy))

	print(proxy, isinstance(proxy, list))
	item = proxy[1][1]

	print(item, isinstance(item, WpDexProxy))

	print(type(item))
	print(type(item).__mro__)

	# test that we can still access the clean structure
	print(proxy._proxyTarget())
	cleanResult = proxy._proxyTarget()[1][1]
	print(cleanResult, type(cleanResult))

	# test that events work at root
	def printEventFn(event, *args, **kwargs):
		log("event", event, args, kwargs)

	proxy.dex().getEventSignal(create=True).connect(printEventFn)
	# change a leaf value
	proxy[1][1] = [5, 6]

	# check that it stuck
	print(proxy[1][1])
	print(s[1][1])

	# check if setting on the base object triggers signal (probably won't)
	s[1][1] = [7, 8]
	print("nowt")




