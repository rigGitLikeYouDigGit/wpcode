
from __future__ import annotations
import typing as T

import pickle # for fast deep copy

from wplib.object import Adaptor, Signal

"""after several years of messing around with wrappers, deltas, events,
signals etc, where are we?

- any arbitrary python structure
ui needs to detect any changes to any level
- structure can be any type, builtins, custom, 3rd party etc

should be invisible to effects editing the structure

try building separate watcher hierarchy and patching mutation methods

adaptor stuff has worked well for serialisation, maybe here too

don't worry about tracking deltas and undoing for now, though this
could be a good base to build that

similar ideas became wpDex, still working on that

I'll delete this soon, I had the ideas here but they were too compressed - 
the more I've split out, the easier it's become to breathe

- before - method - after wrappers
	implemented in proxy for any method on a given object
- mutating methods
	on dex
- event emission
	gathered by proxy, emitted by dex
- watchObject( object )
	now equivalent to WpDexProxy( object ) 
"""

class WatcherAdaptor(Adaptor):
	adaptorTypeMap = Adaptor.makeNewTypeMap()

	_BEFORE_KEY = "before"
	_AFTER_KEY = "after"
	_METHOD_KEY = "method"


	@classmethod
	def objMutationMethodNames(cls, obj:T.Any)->T.Set[str]:
		"""return the names of methods that mutate the object"""
		raise NotImplementedError

	def __init__(self, obj:T.Any):
		self.obj = obj
		self._patchObjMethods(obj)

	def emitChange(self, changeDict:dict[str, T.Any]):
		"""emit a change signal or pass an event, not sure what yet"""
		raise NotImplementedError

	def _patchObjMethod(self, methodName:str):
		"""wrap a method"""
		def wrappedMethod(*args, **kwargs):
			beforeState = pickle.dumps(self.obj)
			result = getattr(self.obj, methodName)(*args, **kwargs)
			#afterState = pickle.dumps(self.obj)
			self.emitChange({
				self._BEFORE_KEY: beforeState,
				self._AFTER_KEY: self.obj,
				self._METHOD_KEY: methodName,
			})
			return result
		#setattr(self.obj, methodName, wrappedMethod)
		self.obj.__dict__[methodName] = wrappedMethod
		pass

	def _patchObjMethods(self, obj:T.Any):
		"""wrap an object"""
		for methodName in self.objMutationMethodNames(obj):
			self._patchObjMethod(methodName)

	pass


class ListWatcher(WatcherAdaptor):
	"""wrap a list to detect changes"""
	forTypes = [list]
	@classmethod
	def objMutationMethodNames(cls, obj:T.Any) ->T.Set[str]:
		return {
			"append",
			"extend",
			"insert",
			"remove",
			"pop",
			"clear",
			"reverse",
			"sort",
			"__setitem__",
			#"__delitem__",
		}

def watchObject(obj):
	return WatcherAdaptor.adaptorForType(type(obj))(obj)

if __name__ == '__main__':

	l = [1, 2, 3]
	watcher = watchObject(l)
	l.append(4)
	pass

