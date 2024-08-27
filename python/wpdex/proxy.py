
from __future__ import annotations

import copy
import typing as T

from collections import defaultdict, namedtuple
import weakref

from wplib.object import DeltaAtom, Proxy, ProxyData

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

	def __init__(self, obj, proxyData: WpDexProxyData, **kwargs):
		"""create a WpDex object for this proxy, add
		weak sets for children and parent"""
		self.psuper().__init__(self, obj, proxyData, **kwargs)
		self._proxyData["parent"] : weakref.ref = None
		# find better way to set up wpdex
		self._proxyData["wpDex"] = WpDex(obj)
		self._proxyData["deltaStartState"] = None
		self._proxyData["deltaCallDepth"] = 0

	def _openDelta(self):
		"""open a delta for this object -
		if mutating functions are reentrant, only open one delta"""
		if self._proxyData["deltaCallDepth"] is 0:
			self._proxyData["deltaStartState"] = copy.deepcopy(self._proxyData["target"])
		self._proxyData["deltaCallDepth"] += 1

	def _emitDelta(self):
		"""emit a delta for this object"""
		self._proxyData["deltaCallDepth"] -= 1
		if self._proxyData["deltaCallDepth"] is 0:
			# delta = DeltaAtom(self._proxyData["deltaStartState"], self._proxyData["target"])
			self._proxyData["deltaStartState"] = None


	def _beforeProxyCall(self, methodName:str,
	                     methodArgs:tuple, methodKwargs:dict,
	                     targetInstance:object
	                     ) ->tuple[T.Callable, tuple, dict, object]:
		fn, args, kwargs, targetInstance = super()._beforeProxyCall(methodName, methodArgs, methodKwargs, targetInstance)

		# check if method will mutate data - need to open a delta
		if methodName in self._proxyWpDex.mutatingMethodNames:
			self._openDelta()

		return fn, args, kwargs, targetInstance

	def _afterProxyCall(self, methodName:str,
	                    method:T.Callable,
	                     methodArgs:tuple, methodKwargs:dict,
	                    targetInstance: object,
	                    callResult:object,
	                    ) ->object:
		callResult = super()._afterProxyCall(methodName, method, methodArgs, methodKwargs, targetInstance, callResult)
		if methodName in self._proxyWpDex.mutatingMethodNames:
			self._emitDelta()
		return callResult

	def _onProxyCallException(self,
	                          methodName: str,
	                          method:T.Callable,
	                          methodArgs: tuple, methodKwargs: dict,
	                          targetInstance: object,
	                          exception:BaseException) ->(None, object):
		"""ensure we still emit / clear deltas if an exception is raised"""
		if methodName in self._proxyWpDex.mutatingMethodNames:
			self._emitDelta()
		raise exception




if __name__ == '__main__':

	t = [1, 2, 3]
	t.append = lambda x: print("nope")

	t.append(4)
	print(t)
