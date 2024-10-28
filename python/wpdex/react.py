
from __future__ import annotations
import typing as T, types
import functools

from param import rx, Parameter

from wplib import log
"""
like the third time in the past week I've got rid of and brought back 
this file

now we're on to something, put functions here to integrate the
reference and rx proxies with other systems (namely UI)

"""

liveT = (types.FunctionType, rx)


def rxAllowsSet(obj):
	"""copy the logic from the original .rx.value =
	function to check if base rx would allow the given object
	to be set
	"""
	obj = obj.rx
	if isinstance(obj._reactive, Parameter): return False
	elif not isinstance(obj._reactive, rx): return False
	elif obj._reactive._root is not obj._reactive: return False
	if obj._reactive._wrapper is None: return False
	return True

def isRxRoot(obj):
	"""check if the given object is the root of an rx chain -
	if not, it'll error if you try to set its value"""
	#log("isRxRoot", isinstance(obj, rx), obj._compute_root() is obj)
	if not isinstance(obj, rx):
		return False
	#return obj._compute_root() is obj
	return rxAllowsSet(obj)

def canBeSet(obj):
	from wpdex.proxy import WX
	#log("canbeset", isinstance(obj, liveT))
	if not isinstance(obj, liveT):
		return True
	if isinstance(obj, types.FunctionType): return False
	if "_dexPath" in obj._kwargs and hasattr(obj, "WRITE"):
		log("obj has path", obj._kwargs["_dexPath"])
		return True
	return isRxRoot(obj)

def EVAL1(i):
	"""this will have to be integrated with expressions later, somehow
	evaluate a single input and return a single result
	"""
	if isinstance(i, rx):
		return i.rx.value
	if isinstance(i, types.FunctionType):
		return i()
	return i

def EVAL(*args, **kwargs):
	"""call this on all args of all functions in live networks to check
	if we have anything live inserted there
	need to call as *EVAL(*args), **EVAL(**kwargs)
	since there's no easy way to inline-unpack a tuple of ( tuple, dict )

	at scale this might turn a lot of our UI code into some godforsaken
	C-macro java lovechild, but let's stick with it for now

	to eval a single value, use EVAL1
	"""
	if args:
		return tuple(map(EVAL1, args))
	if kwargs:
		return {k: EVAL1(v) for k, v in kwargs.items()}

def wrapEval(fn):
	@functools.wraps(fn)
	def _evalArgsKwargs(*args, **kwargs):
		return fn(*EVAL(*args), **EVAL(**kwargs))
	return _evalArgsKwargs


class _ReactPatchTracker:
	"""
	TODO: this is for later, get the more verbose stuff working for now
	"""
	@staticmethod
	def __reactGetAttr__(selfObj, name):
		assert hasattr(selfObj, "_preRCache")
		baseResult = selfObj.__dict__["preRCache"]["__orig_getattr__"](name)
		if not isinstance(baseResult, types.FunctionType):
			return baseResult
		d = object.__getattribute__(selfObj, "__dict__")
		if not baseResult in d["preRCache"]["origFnReactMap"]:
			# wrap function to eval arguments
			wrapped = wrapEval(baseResult)
			d["preRCache"]["origFnReactMap"][baseResult] = wrapped
		return d["preRCache"]["origFnReactMap"][baseResult]

	@classmethod
	def wrap(cls, obj):
		obj.__dict__["preRCache"] = {
			"__orig_getattr__" : obj.__getattr__,
			"preRCache" : {
				"origFnReactMap" : {}
			}
		}

def WRAP_MEMBERS(obj): # good idea
	return _ReactPatchTracker.wrap(obj)

def BIND(src:(rx, T.Any), fn:callable, *args, **kwargs):
	"""
	all-in-one driving system - if source is a live reactive
	object, drive fn live, else just set it with a static value

	args and kwargs are passed in the same way as a partial,
	and eval'd themselves

	BIND(myReactiveSource, function, fnArg1, fnKwarg2=maybeReactiveKwarg)

	"""
	if isinstance(src, rx):
		def _watchWithEval(v):
			fargs, fkwargs = EVAL(*args), EVAL(**kwargs)
			return fn(v, *fargs, **fkwargs)
		src.rx.watch(_watchWithEval, onlychanged=False)
		return
	if isinstance(src, types.FunctionType):
		"""this isn't as strong a relation since we can't guarantee all the
		inputs to the raw function even with rx, """
		raise TypeError(f"cannot bind a raw function {src} to {fn},\n use EVAL synchronously instead")
		src.rx = rx(src)

	fn(src, *EVAL(*args), **EVAL(**kwargs))


class Reactive:
	""""""
	def __init__(self):
		self.dirtyFlag = rx(False)
