from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wplib.object import Sentinel



""" similar to proxy but focused on copying calls
to a set of objects, rather than disguising as them


goes hand in hand with broadcast

"""

class WrapAccessor:
	"""holds operator functions for wrap objects, without
	going through their attribute access every time"""
	def __init__(self, obj:WrapList):
		self.obj = obj

	def __getitem__(self, key):
		return list(self.obj).__getitem__(key)

	def __call__(self, *args, **kwargs):
		return list(self.obj)

l = [1, 2, 3]
r = l[:] # compatible with both wrapper and normal list
l[::] # flatten recursively? seems silly

class WrapList(list):
	"""

	flatten to normal list by
	>>>multi[:]
	?
	get exact list attr by
	>>>multi._[3]

	wunky little test that grew from broadcasting/slicing maya plugs:
	when you run a filter operation on plugs/pathables,
	you get a selection of objects matching that filter.
	this allows further operations to be run, splitting in a more readable way

	this should ALSO allow copying any method lookups done on this object to
	all referenced objects
	somehow get type hinting for that, we're into c++ templating
	with every pathable returning a PathableSelection[Plug]
	from expression lookups

	`selection.method(arg)` is of course 2 calls - one getattr to get the method object,
	and a __call__ on that object
	this is inconvenient as we would need to return a second selection of just the methods,
	and then perform the call on that

	how do we actually resolve the result of these calls?

	myMultiObject.value() -> [ list of values ] # this one sparks joy
	myMultiObject.value() -> MultiObject( [ list of values ] )# this one does not spark joy

	myMultiObject.intermediateFn().value() -> [ list of deep values ]
	# we just check if it's top level somehow?

	intermediateObjs = myMultiObject.intermediateFn() # MultiObject
	resultList = intermediateObjs.value() # return list

	that classic indicator I'm wasting my time -
	sometimes you would want it, and sometimes you would not


	...
	do we just inherit from list?

	IMMEDIATE access on this object should ALWAYS DISTRIBUTE
	access list and selection itself by .toList() ?
	by list(multiObj) ?

	override logic operators here to use this for intersections,
	filters

	obj._.unwrap() -> access an actual operator method on this list

	__iter__ goes through all contents

	"""

	def __init__(self, *args, distributeGetAttr=True, distributeGetItem=True):
		super().__init__(*args)
		self._distributeGetAttr = distributeGetAttr
		self._distributeGetItem = distributeGetItem


	def _(self):
		"""MAYBE TEMP - just use this for now, rename if a better idea
		occurs"""
		return WrapAccessor(self)

	def _makeChildWrapList(self, objs)->WrapList:
		return type(self)(objs, distributeGetAttr=self._distributeGetAttr,
		                  distributeGetItem=self._distributeGetItem)


	def __str__(self):
		return f"<{type(self).__name__}({list(self)})>"

	# steal callables of all objects in list?
	# nah way too slow

	def __getattr__(self, item):
		return self._makeChildWrapList(filter(
			lambda x: x is not Sentinel.FailToFind,
			(getattr(i, item, Sentinel.FailToFind) for i in self)
		))


	def __tryGetItem(self, obj, *key):
		try:
			return obj.__getitem__(*key)
		except TypeError as e:
			return Sentinel.FailToFind

	def __getitem__(self, *key):
		"""run filter over this MultiObject, return matching elements
		multiObj[:] -> return a list copy

		logic here of whether to return a new list of this exact type, or
		just a vanilla WrapList - seen this before, consider a base for it
		"""
		if key[0] == slice(None):
			return list(self)
		return self._makeChildWrapList(filter(
			lambda x: x is not Sentinel.FailToFind,
			(self.__tryGetItem(i, *key) for i in self)
		))

	def __tryCall(self, obj, *args, **kwargs):
		try:
			return obj(*args, **kwargs)
		except TypeError as e:  # should we filter this more, check message for "is not callable" etc?
			return Sentinel.FailToFind

	def __call__(self, *args, **kwargs):
		"""if you try and call a non-callable object, just strip out result
		"""

		return self._makeChildWrapList(
			filter(
				lambda x: x is not Sentinel.FailToFind,
				(self.__tryCall(i, *args, **kwargs) for i in self)
			)
		)

	def __multiSet(self, key, value):
		"""OVERRIDE THIS for logic setting multiple items at once"""
		for i in self:
			try:
				i.__setitem__(key, value)
			except TypeError as e:
				pass


	def __setitem__(self, key, value):
		self.__multiSet(key, value)
