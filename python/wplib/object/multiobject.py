from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wplib.object import Sentinel



""" similar to proxy but focused on copying calls
to a set of objects, rather than disguising as them


goes hand in hand with broadcast

"""


#class MultiObject(list):
class MultiObject:
	"""wunky little test that grew from broadcasting/slicing maya plugs:
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
	"""

	def __init__(self, objs: T.Iterable[object]):
		self.objs = list(objs)


	# steal callables of all objects in list?
	# nah way too slow

	def __getattr__(self, item):
		return MultiObject(filter(
			lambda x: x is not Sentinel.FailToFind,
			(getattr(i, item, Sentinel.FailToFind) for i in self.objs)
		))


	def __tryGetItem(self, obj, key):
		try:
			return obj.__getitem__(key)
		except TypeError as e:
			return Sentinel.FailToFind

	def __getitem__(self, key):
		"""run filter over this MultiObject, return matching elements
		multiObj[:] -> return a list copy
		"""
		for i in self.objs:
			MultiObject(
				filter(
					lambda x: x is not Sentinel.FailToFind,
					(self.__tryGetItem(i, key) for i in self.objs)
				)
			)

	def __tryCall(self, obj, *args, **kwargs):
		try:
			return obj(*args, **kwargs)
		except TypeError as e:  # should we filter this more, check message for "is not callable" etc?
			return Sentinel.FailToFind

	def __call__(self, *args, **kwargs):
		"""if you try and call a non-callable object, just strip out result
		"""

		return MultiObject(
			filter(
				lambda x: x is not Sentinel.FailToFind,
				(self.__tryCall(i, *args, **kwargs) for i in self.objs)
			)
		)

	def __multiSet(self, key, value):
		"""OVERRIDE THIS for logic setting multiple items at once"""
		for i in self.objs:
			try:
				i.__setitem__(key, value)
			except TypeError as e:
				pass


	def __setitem__(self, key, value):
		self.__multiSet(key, value)
