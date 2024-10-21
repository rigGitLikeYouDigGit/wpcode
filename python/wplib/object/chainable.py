
from __future__ import annotations

import types
import typing as T

from functools import partial, partialmethod
from wplib import log

class Chainable:
	"""
	chain attribute and method access into a pipeline, drawing from a base
	getting on a Chainable does not modify that object, but returns a new Chainable,
	whose latest op is that lookup.
	That way,
	chainableA.name.upper()
	chainableA.split(",").filter(None)
	doesn't mutate chainableA

	If we were to integrate this with the proxy system, the danger is in Chainable pipelines
	spreading throughout our code,
	"""

	def __init__(self, base, ops=None,
	             fn=None,
	             isCall=False,
	             ):
		self.base = base
		#self._instance = _instance
		self.fn : callable = fn
		self.ops : list[Chainable] = ops or []
		self.isCall = isCall

		# if you wrap a function as a chainable
		self._isRawFn = isinstance(self.base, types.FunctionType)

	def __str__(self):
		fnStr = "None"
		if self.fn:
			fnStr = self.fn.__qualname__.split(" at ")[0]
		opStr = ""
		if self.ops:
			opStr = f"ops={self.ops[-1]}"
		return f"<Chainable(fn={fnStr}, base={self.base} {opStr})>"

	def __repr__(self):
		return str(self)

	def __getattr__(self, item):
		if self._isRawFn: # attributes of functions aren't dynamic
			return getattr(self.base, item)
		def chainGetAttr(obj):
			try: return obj.__getattr__(item)
			except AttributeError:
				return type(obj).__dict__[item]
		op = chainGetAttr
		chainGetAttr.__name__ = f"getAttr({item})"
		chainGetAttr.__qualname__ = f"getAttr({item})"
		#self.fn = op
		return Chainable(self.base, ops=self.ops + [self],
		                         fn=op)

	def __getitem__(self, item):
		def chainGetItem(obj):
			return obj.__getitem__(item)
		op = chainGetItem
		chainGetItem.__name__ = f"getItem({item})"
		chainGetItem.__qualname__ = f"getItem({item})"
		#self.fn = op
		return Chainable(self.base, ops=self.ops + [self],
		                         fn=op)

	def __call__(self, *args, **kwargs):
		"""janky to call the looked up attribute, while still passing
		the last found instance as self argument"""

		if self._isRawFn:
			def _op():
				fnArgs, fnKwargs = EVAL(*args, **kwargs)
				return self.base(*fnArgs, **fnKwargs)
		else:
			def _op(obj, instance):
				fnArgs, fnKwargs = EVAL(*args, **kwargs)
				try:
					return obj.__call__(instance, *fnArgs, **fnKwargs)
				except TypeError: # it's a class or static method
					return obj.__call__(*fnArgs, **fnKwargs)
		op = _op
		op.__qualname__ = "call()"
		return Chainable(self.base,
		                 ops=self.ops + [self],
		                 fn=op,
		                 isCall=True)

	def __eval__(self):
		result = self.base
		instance = self.base
		#log("EVAL")
		prev = None
		for i in self.ops + [self]:
			#log("op", i)
			if i.fn is None:
				continue
			#log(instance, result)
			if i.isCall:
				#log("do call", i.fn, result, instance)
				result = i.fn(result, instance)
				instance = result
			else:
				#log("do noncall", i.fn, result)
				result = i.fn(result)
			prev = i

		return result

	EVAL = __eval__

def _evalSingle(i):
	"""this will have to be integrated with expressions later, somehow"""
	if isinstance(i, types.FunctionType):
		return i()
	if isinstance(i, Chainable):
		return i.__eval__()
	return i

def EVAL(*args, **kwargs):
	"""call this on all args of all functions in live networks to check
	if we have anything live inserted there"""
	return (
		*map(_evalSingle, args),
		{k : _evalSingle(v) for k, v in kwargs.items()}
	)



def myFn(inStr:str):
	return inStr + "_addendum"

if __name__ == '__main__':
	s = "hello"

	# print(type(s).__dict__["upper"].__call__(s))
	#
	# c = Chainable(s)
	# upper = c.upper().title()
	# print(upper)
	# print(upper.__eval__())
	#
	#
	# t = dict
	# ct = Chainable(t)
	# emptyItems = ct().items()
	# print(emptyItems)
	# print(emptyItems.__eval__())
	#
	#

	baseDict = {"a" : [0, 1, 2]}
	cDict = Chainable(baseDict)
	log("get item", cDict["a"][1], cDict["a"][1].EVAL() )
