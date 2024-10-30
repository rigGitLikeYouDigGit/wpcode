
from __future__ import  annotations
import typing as T
import types
from param.reactive import rx
from types import FunctionType, MethodType
from functools import partial, partialmethod
from wplib import log

class WX:
	"""
	chain attribute and method access into a pipeline, drawing from a base
	getting on a WX does not modify that object, but returns a new WX,
	whose latest op is that lookup.
	That way,
	chainableA.name.upper()
	chainableA.split(",").filter(None)
	doesn't mutate chainableA

	If we were to integrate this with the proxy system, the danger is in WX pipelines
	spreading throughout our code,
	"""

	def __init__(self, base, ops=None,
	             fn=None,
	             isCall=False,
	             ):
		self.base = base
		#self._instance = _instance
		self.fn : callable = fn
		self.ops : list[WX] = ops or []
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
		return f"<WX(fn={fnStr}, base={self.base} {opStr})>"

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
		return WX(self.base, ops=self.ops + [self],
		          fn=op)

	def __getitem__(self, item):
		def chainGetItem(obj):
			return obj.__getitem__(item)
		op = chainGetItem
		chainGetItem.__name__ = f"getItem({item})"
		chainGetItem.__qualname__ = f"getItem({item})"
		#self.fn = op
		return WX(self.base, ops=self.ops + [self],
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
		return WX(self.base,
		          ops=self.ops + [self],
		          fn=op,
		          isCall=True)

	def __eval__(self):
		result = self.base
		instance = self.base
		#log("EVALAK")
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



def myFn(inStr:str):
	return inStr + "_addendum"


def BIND(subject, outputFn:callable):
	pass

from PySide2 import QtCore, QtWidgets, QtGui
from threading import Thread
import time, math

from wplib import log

def sinTime():
	return math.sin(time.time())

class WRX(rx):
	def __str__(self):
		return f"WRX({self.rx.value})"

run = True

val = "INIT VAL"
rxval = WRX(val)
def task():
	while run:
		print("time", sinTime())
		rxval.rx.value = str(sinTime())
		#rxval.rx = str(sinTime())
		print("val", rxval.rx.value)
		time.sleep(0.2)
	pass


if __name__ == '__main__':
	s = "hello"

	# print(type(s).__dict__["upper"].__call__(s))
	#
	# c = WX(s)
	# upper = c.upper().title()
	# print(upper)
	# print(upper.__eval__())
	#
	#
	# t = dict
	# ct = WX(t)
	# emptyItems = ct().items()
	# print(emptyItems)
	# print(emptyItems.__eval__())
	#
	#

	baseDict = {"a" : [0, 1, 2]}
	cDict = WX(baseDict)
	log("get item", cDict["a"][1], cDict["a"][1].EVAL() )









