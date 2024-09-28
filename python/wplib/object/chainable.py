
from __future__ import annotations
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

	TODO: janky handling of keeping track of instance arguments
		to pass as self - if you have a callable base object, this
		will freak out


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

	def __str__(self):
		return f"<Chainable(fn={self.fn}, base={self.base}, ops={self.ops})>"

	def __repr__(self):
		return str(self)

	def __getattr__(self, item):
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

	def __call__(self, *args, **kwargs):
		"""janky to call the looked up attribute, while still passing
		the last found instance as self argument"""
		op = lambda obj, instance : obj.__call__(instance, *args, **kwargs)
		op.__qualname__ = "call()"
		return Chainable(self.base,
		                 ops=self.ops + [self],
		                 fn=op,
		                 isCall=True)

	def __eval__(self):
		result = self.base
		i = 0
		instance = self.base
		log("EVAL")
		#while i < len(self.ops):
		for i in self.ops + [self]:
			log("op", i)
			if i.fn is None:
				continue
			# if self.fn is None:
			# 	i += 1
			# 	continue
			log(instance, result)
			if i.isCall:
				log("do call", i.fn, result, instance)
				result = i.fn(result, instance)
				instance = result
			else:
				log("do noncall", i.fn, result)
				result = i.fn(result)

		return result


if __name__ == '__main__':
	s = "hello"

	print(type(s).__dict__["upper"].__call__(s))

	c = Chainable(s)
	upper = c.upper().title()
	print(upper)
	print(upper.__eval__())

	"""
	t = dict
	ct = Chainable(t)
	emptyItems = ct().items()
	print(emptyItems)
	print(emptyItems.__eval__())
	"""
	#TODO: see above, passing a callable base object freaks out

