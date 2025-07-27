from __future__ import annotations
import types, typing as T
import pprint

from param import rx, Parameter
from param.parameterized import resolve_value
from param.reactive import reactive_ops

from wplib import log
from wplib.object import Signal


class Wreactive_ops(reactive_ops):
	"""test overriding reactive_ops,
	more specific support for writing back to refs
	using
	.rx.value =
	syntax, since otherwise you have to do some type checking to
	tell when to use BIND(), etc
	"""
	@property
	def value(self):
		"""
		Returns the current state of the reactive expression by
		evaluating the pipeline.
		"""
		if isinstance(self._reactive, rx):
			return self._reactive._resolve()
		elif isinstance(self._reactive, Parameter):
			return getattr(self._reactive.owner, self._reactive.name)
		else:
			return self._reactive()

	@value.setter
	def value(self, new):
		"""
		Allows overriding the original input to the pipeline.
		"""
		# log("set value", new,
		#     self._reactive, self._reactive._wrapper)
		#root = self._reactive._compute_root()
		#log("root", root)

		#print("")
		#log("set value")

		# wx = WX.getWXRoot(self._reactive)
		# if wx is not None:
		# 	#assert wx
		# 	#if "_dexPath" in wx._kwargs:
		# 	#if "_dexPath" in root._kwargs:
		# 	#root.WRITE(resolve_value(new))
		# 	#log("EMITTING")
		# 	reactive_ops.value.fset(self, new)
		# 	wx.WRITE(resolve_value(new))
		# 	return
		reactive_ops.value.fset(self, new)
		if isinstance(self._reactive, WX):
			resolved = resolve_value(value=new)
			self._reactive.WRITE(resolved)

		# try:
		#
		# except AttributeError as e:
		# 	#return
		#
		#
		# 	hasPath = "_dexPath" in root._kwargs
		# 	log("root", root, hasPath)
		# 	if hasPath: return
		# 	raise e



	# @value.setter
	# def value(self, new):
	# 	"""
	# 	Allows overriding the original input to the pipeline.
	# 	"""
	#
	# 	log("set value", new,
	# 	    type(self._reactive), self._reactive._wrapper)
	# 	if "_dexPath" in self._reactive._kwargs:
	# 		self._reactive.WRITE(resolve_value(new))
	# 		return
	# 	rootHasPath = "_dexPath" in self._reactive._compute_root()._kwargs
	# 	if isinstance(self._reactive, Parameter):
	# 		raise AttributeError(
	# 			"`Parameter.rx.value = value` is not supported. Cannot override "
	# 			"parameter value."
	# 		)
	# 	elif not isinstance(self._reactive, rx):
	# 		raise AttributeError(
	# 			"`bind(...).rx.value = value` is not supported. Cannot override "
	# 			"the output of a function."
	# 		)
	# 	# elif "_dexPath" in self._reactive._kwargs:
	# 	# 	self._reactive.WRITE(resolve_value(new))
	# 	# 	return
	# 	elif self._reactive._root is not self._reactive:
	# 		if rootHasPath: return
	# 		raise AttributeError(
	# 			"The value of a derived expression cannot be set. Ensure you "
	# 			"set the value on the root node wrapping a concrete value, e.g.:"
	# 			"\n\n    a = rx(1)\n    b = a + 1\n    a.rx.value = 2\n\n "
	# 			"is valid but you may not set `b.rx.value = 2`."
	# 		)
	# 	if self._reactive._wrapper is None:
	# 		if rootHasPath: return
	# 		raise AttributeError(
	# 			"Setting the value of a reactive expression is only "
	# 			"supported if it wraps a concrete value. A reactive "
	# 			"expression wrapping a Parameter or another dynamic "
	# 			"reference cannot be updated."
	# 		)
	# 	self._reactive._wrapper.object = resolve_value(new)

	def watch(self, fn=None, onlychanged=True, queued=False, precedence=0, fire=False):
		"""
		Adds a callable that observes the output of the pipeline.
		If no callable is provided this simply causes the expression
		to be eagerly evaluated.

		Added "fire" flag to call given function once, after bind
		"""
		if precedence < 0:
			raise ValueError("User-defined watch callbacks must declare "
							 "a positive precedence. Negative precedences "
							 "are reserved for internal Watchers.")
		self._watch(fn, onlychanged=onlychanged, queued=queued, precedence=precedence)
		if fire:
			fn(self.value)

class WX(rx):
	"""By default, printing an rx object freaks out because it returns
	an implicit __str__() call, not an actual string that can be printed

	putting in caps so that it's obvious when we do stuff with it
	"""
	rx : Wreactive_ops
	def __repr__(self):
		try:
			return f"WX({repr(self.rx.value)})"
		except Exception as e:
			return f"WX(ERROR GETTING REPR)"
	def __str__(self):
		return f"WX({repr(self.rx.value)})"
	def __init__(self, *args,# path:WpDex.pathT=(),
	             **kwargs):
		if kwargs.get("_writeSignal", None) is None:
			kwargs["_writeSignal"] = Signal("WX-write")
		super().__init__(*args,# _dexPath=path,
		                 **kwargs
		                 )
		# pack path in _kwargs to ensure it gets copied on _clone()
		# also signal, they're expensive to build

	# TODO###: patch this in properly
	#  currently we have a filthy untracked patch in rx, to paste this block on the end of
	#   rx.__getattribute__
	#   instead of erroring straight out - waits to error until a reference actually computes

	# def __getattribute__(self, name):
	# 	try:
	# 		return super().__getattribute__(name)
	# 	except AttributeError:
	# 		#selfName = str(self).rx.value
	# 		# selfName = str(self)
	# 		# log("attribute error getting", name, "from", selfName)
	# 		new = self._resolve_accessor()
	# 		new._method = name
	# 		return new

	@classmethod
	def getWXRoot(cls, rxInstance:rx):
		while not isinstance(rxInstance, WX):
			# try:
			# 	log("discount rx", rxInstance)
			# except:
			# 	log("discount rx", type(rxInstance))
			rxInstance = rxInstance._prev
			if rxInstance is None: return None
		return rxInstance if isinstance(rxInstance, WX) else None

	def WRITE(self, val):
		"""emit (path, value)
		"""
		if "_dexPath" in self._kwargs:
			self._kwargs["_writeSignal"].emit(self._kwargs["_dexPath"], val)
		else: # WOOPS
			self._kwargs["_writeSignal"].emit((), val)

	# def _resolveRef():
	# 	rootDex = self.dex()
	# 	foundDex: WpDex = rootDex.access(rootDex, path, values=False, one=True)
	# 	return foundDex.getValueProxy()

	def RESOLVE(self, dex=False, proxy=False, value=False):
		assert dex or proxy or value
		rootDex = self._kwargs["_dex"]
		path = self._kwargs["_dexPath"]
		if dex:
			return rootDex.access(rootDex, path, values=False)
		if proxy:
			return rootDex.access(rootDex, path, values=False).getValueProxy()
		if value:
			return self.rx.value


