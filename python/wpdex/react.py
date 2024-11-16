
from __future__ import annotations
import typing as T, types
import functools

from param import rx, Parameter, Event

from wplib.object import DeepVisitor

from wplib import log
"""
like the third time in the past week I've got rid of and brought back 
this file

now we're on to something, put functions here to integrate the
reference and rx proxies with other systems (namely UI)

"""

liveT = (types.FunctionType, rx)

def hasRx(obj):
	return hasattr(obj, "rx")

def getRx(obj):
	return getattr(obj, "rx", None)


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
		#log("obj has path", obj._kwargs["_dexPath"])
		return True
	return isRxRoot(obj)

def EVAL(i, *args, **kwargs):
	"""this will have to be integrated with expressions later, somehow
	evaluate a single input and return a single result
	"""
	#if isinstance(i, rx):
	if hasRx(i):
		return i.rx.value
	if isinstance(i, (functools.partial, functools.partialmethod)):
		# evaluate all arguments, then function itself
		fargs, fkwargs = EVALA(*i.args), EVALK(**i.keywords)
		return i.func(*fargs, **fkwargs)
	if isinstance(i, types.FunctionType):
		return i(*EVALA(*args), **EVALK(**kwargs))
	return i

def EVALA(*args):
	return map(EVAL, args)

def EVALK(**kwargs):
	return {k: EVAL(v) for k, v in kwargs.items()}

def EVALAK(*args, **kwargs):
	"""call this on all args of all functions in live networks to check
	if we have anything live inserted there
	need to call as *EVALAK(*args), **EVALAK(**kwargs)
	since there's no easy way to inline-unpack a tuple of ( tuple, dict )

	at scale this might turn a lot of our UI code into some godforsaken
	C-macro java lovechild, but let's stick with it for now

	to eval a single value, use EVAL
	"""
	if args:
		return tuple(map(EVAL, args))
	if kwargs:
		return {k: EVAL(v) for k, v in kwargs.items()}

def EVALRecursive(obj):
	"""the nuclear option"""
	v = DeepVisitor()
	result = v.dispatchPass(obj,
	               passParams=v.VisitPassParams(
		               topDown=True, depthFirst=True,
		               visitFn=EVAL,
		               transformVisitedObjects=True
	               ))
	return result

def wrapEval(fn):
	@functools.wraps(fn)
	def _evalArgsKwargs(*args, **kwargs):
		return fn(*EVALAK(*args), **EVALAK(**kwargs))
	return _evalArgsKwargs


# if we had some medium short-hand way to declare attributes on init?

class LiveAttr:
	"""like an instance descriptor? idk
	need to avoid double-declaration
	"""

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

class QtWidget:

	def setName(self, s):
		"""a normal imperative object method, as god intended"""

def wrapFn(fn:T.Callable):
	"""sketch to watch rx values implicitly when passed
	still to early to implement everywhere, but consider it for the future

	if called with all
	"""
	rxFn = rx(fn)
	fn.rx = rxFn
	@functools.wraps(fn)
	def _reactiveFn(*args, **kwargs):
		liveArgs = False
		for i in args:
			if isinstance(i, liveT):
				liveArgs = True
				rx(i).rx.watch(lambda *_ : fn(*EVALA(*args), **EVALK(**kwargs)),
				               onlychanged=False)
		for k, v in kwargs:
			if isinstance(v, liveT):
				liveArgs = True
				rx(v).rx.watch(lambda *_ : fn(*EVALA(*args), **EVALK(**kwargs)),
				               onlychanged=False)
		#### TODO:
		###     check that the above doesn't work,
		#       fires function for every live argument

		if not liveArgs: # just a function call
			return fn(*args, **kwargs) # just an innocent function call

		rxFn(*args, **kwargs)
	return _reactiveFn

"""
is there a world where we try and do some kind of static analysis on
functions, to find if they rely on object attributes that could be
reactive?

"""

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
			fargs, fkwargs = EVALAK(*args), EVALAK(**kwargs)
			return fn(v, *fargs, **fkwargs)
		src.rx.watch(_watchWithEval, onlychanged=False)
		return
	if isinstance(src, types.FunctionType):
		"""this isn't as strong a relation since we can't guarantee all the
		inputs to the raw function even with rx, """
		raise TypeError(f"cannot bind a raw function {src} to {fn},\n use EVALAK synchronously instead")
		src.rx = rx(src)

	fn(src, *EVALAK(*args), **EVALAK(**kwargs))

def PING(val):
	if not isinstance(val, liveT): return
	val.rx._reactive._wrapper.trigger()
	#val._wrapper.trigger()
	#val.rx.trigger()
	# log("PING", val, type(val))
	# val = rx(val)
	# root = val._compute_root()


	#val._invalidate_obj()
	#val._invalidate_current()
	#val._setup_invalidations()

	# if obj is None:
	# 	watchers = self.watchers.get("value")
	"""
	elif name in obj._param__private.watchers:
		watchers = obj._param__private.watchers[name].get('value')
		if watchers is None:
			watchers = self.watchers.get("value")
	else:
		watchers = None

	obj = self.owner if obj is None else obj

	if obj is None or not watchers:
		return

	event = Event(what='value', name="value", obj=val, cls=rx,
	              old=_old, new=val, type=None)

	# Copy watchers here since they may be modified inplace during iteration
	for watcher in sorted(watchers, key=lambda w: w.precedence):
		obj.param._call_watcher(watcher, event)
	if not obj.param._BATCH_WATCH:
		obj.param._batch_call_watchers()


	def __set__(self, obj, val):
		# from wplib import log
		name = self.name
		if obj is not None and self.allow_refs and obj._param__private.initialized:
			syncing = name in obj._param__private.syncing
			ref, deps, val, is_async = obj.param._resolve_ref(self, val)
			refs = obj._param__private.refs
			if ref is not None:
				self.owner.param._update_ref(name, ref)
			elif name in refs and not syncing:
				del refs[name]
				if name in obj._param__private.async_refs:
					obj._param__private.async_refs.pop(name).cancel()
			if is_async or val is Undefined:
				return

		# Deprecated Number set_hook called here to avoid duplicating setter
		if hasattr(self, 'set_hook'):
			val = self.set_hook(obj, val)
			if self.set_hook is not _identity_hook:
				# PARAM3_DEPRECATION
				warnings.warn(
					'Number.set_hook has been deprecated.',
					category=_ParamDeprecationWarning,
					stacklevel=6,
				)

		self._validate(val)

		_old = NotImplemented
		# obj can be None if __set__ is called for a Parameterized class
		if self.constant or self.readonly:
			if self.readonly:
				raise TypeError("Read-only parameter '%s' cannot be modified" % name)
			elif obj is None:
				_old = self.default
				self.default = val
			elif not obj._param__private.initialized:
				_old = obj._param__private.values.get(self.name, self.default)
				obj._param__private.values[self.name] = val
			else:
				_old = obj._param__private.values.get(self.name, self.default)
				if val is not _old:
					raise TypeError("Constant parameter '%s' cannot be modified" % name)
		else:
			if obj is None:
				_old = self.default
				self.default = val
			else:
				# When setting a Parameter before calling super.
				if not isinstance(obj._param__private, _InstancePrivate):
					obj._param__private = _InstancePrivate(
						explicit_no_refs=type(obj)._param__private.explicit_no_refs
					)
				_old = obj._param__private.values.get(name, self.default)
				# old keeps the original value from rx.value, since it counts as a new param
				obj._param__private.values[name] = val
		self._post_setter(obj, val)

		if obj is not None:
			if not hasattr(obj, '_param__private') or not getattr(obj._param__private, 'initialized', False):
				return
			obj.param._update_deps(name)

		if obj is None:
			watchers = self.watchers.get("value")
		elif name in obj._param__private.watchers:
			watchers = obj._param__private.watchers[name].get('value')
			if watchers is None:
				watchers = self.watchers.get("value")
		else:
			watchers = None

		obj = self.owner if obj is None else obj

		if obj is None or not watchers:
			return

		event = Event(what='value', name=name, obj=obj, cls=self.owner,
		              old=_old, new=val, type=None)

		# Copy watchers here since they may be modified inplace during iteration
		for watcher in sorted(watchers, key=lambda w: w.precedence):
			obj.param._call_watcher(watcher, event)
		if not obj.param._BATCH_WATCH:
			obj.param._batch_call_watchers()

	"""
class Reactive:
	""""""
	def __init__(self):
		self.dirtyFlag = rx(False)



if __name__ == '__main__':

	print(all([]))
