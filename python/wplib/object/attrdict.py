from __future__ import annotations
import types, typing as T
import pprint
from wplib import log, EVAL

import os, sys
from dataclasses import dataclass
"""fancier way to predict dict items"""

class AttrDict:

	def __setattr__(self, key, value):
		if key in self.__dict__:
			self.__dict__[key] = value
		else:
			self[key] = value

	def __getattr__(self, item):
		if item in self.__dict__:
			return self.__dict__[item]
		return self[item]


"""TypedDict is very useful for giving structure to shallow and simple structures,
but it's a pain not to be able to specify instance / classmethods,
 NOT TO MENTION default values.
 
Dataclasses are cool for defaults and associated methods, but exquisitely annoying for
__init__ methods and serialisation, not to mention packing
into custom python objects rather than dicts.

Try to combine benefits of these two below:

- hinting on creation
- default values
- full class and instance methods
- overrideable init


so, slightly annoying, pycharm is special-cased to the exact TypedDict function in
the STL to apply its hints - 
no problem, just means we have to monkey-patch around it
 """


_tdOld = T.TypedDict # store actual T.TypedDict while loading this module

T.TypedDict = dict # nothing to see here officer

class _TDMeta(type):

	def __call__(cls, *args, **kwargs):
		#log("_TDMeta _call_", args, kwargs)
		hints = T.get_type_hints(cls)
		b : dict = super().__call__( *args, **kwargs)
		for k, v in hints.items():
			if hasattr(cls, k):
				b.setdefault(k, EVAL(getattr(cls, k)))
		return b


class TDefDict(T.TypedDict, metaclass=_TDMeta):

	def __str__(self):
		return f"<{self.__class__.__name__}{str(dict(self))}>"

T.TypedDict = _tdOld # restore original TypedDict to module

if __name__ == '__main__':
	class Test(TDefDict):
		"""order not guaranteed (yet), haven't found it useful"""
		a: int = 22
		b: str = "hello"
		pass


	t = Test(a=3, b="ey")
	print(t, type(t))
	t = Test(a=3)
	print(t, type(t))
	t = Test( b="ey")
	print(t, type(t))

	t = Test({"randomKey" : 77, "a" : "wot"})
	print(t, type(t))

