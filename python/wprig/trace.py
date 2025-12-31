from __future__ import annotations
import types, typing as T
import pprint

import inspect
import ast
import textwrap

from dataclasses import dataclass

import jax
from jax import numpy as jnp

"""
'wouldn't it be nice to use strings and named values when authoring
constraint gpu code, and have those resolve to indices before
jit-compiling the function?'

and from that, this monstrosity:

basically we do a python version of the C preprocessor, embedded within
any python function - 
in order to do that without ast stuff (which I still think could have
worked),
we all but reimplement the tracing stuff that JAX does to freeze a python 
call into a full dag graph.

"""

@dataclass
class Ref:
	"""
	Symbolic reference node.
	TODO: make this more general and inherit a JaxRef for jax domain-specific help, base class should only support recording tracing.
	TODO: actually move all active interpretation out of this object

	op:
		- "measure"
		- "const"
		- arithmetic op name ("add", "mul", ...)
		- math function name ("abs", ...)
	args:
		input Refs
	value:
		for constants
	index:
		for measured values (filled after resolution)
	"""
	op: str
	args: tuple["Ref", ...] = ()
	value: T.Any = None
	index: int | None = None

	def __hash__(self):
		"""explicit for clarity - value has to be hashable
		TODO: maybe we have a toHashable() fn to accept lists
		"""
		return hash(
			(self.op, self.args, hash(self.value), self.index)
		)
	# ---------- constructors ----------

	@staticmethod
	def const(value):
		return Ref(op="const", value=value)

	@staticmethod
	def measure(index: int):
		return Ref(op="measure", index=index)


	# ---------- operator overloads ----------

	def __add__(self, other):
		return Ref("add", (self, ensureRef(other)))

	def __radd__(self, other):
		return Ref("add", (ensureRef(other), self))

	def __mul__(self, other):
		return Ref("mul", (self, ensureRef(other)))

	def __rmul__(self, other):
		return Ref("mul", (ensureRef(other), self))

	def __sub__(self, other):
		return Ref("sub", (self, ensureRef(other)))

	def __rsub__(self, other):
		return Ref("sub", (ensureRef(other), self))

	def __truediv__(self, other):
		return Ref("div", (self, ensureRef(other)))

	def __rtruediv__(self, other):
		return Ref("div", (ensureRef(other), self))

	def __neg__(self):
		return Ref("neg", (self,))

def ensureRef(x):
	if isinstance(x, Ref):
		return x
	return Ref.const(x)

@dataclass
class FunctionRef(Ref):
	fn: T.Callable = None
	args: tuple[Ref, ...] = ()
	kwargs: tuple[tuple[str, Ref], ...] = ()
	name: str | None = None

	def __hash__(self):
		return hash(
			super().__hash__() +
			hash((self.fn, self.args, self.kwargs, self.name))
		)

	def __init__(self, fn, args, kwargs=None):
		super().__init__(
			op="call",
			#args=tuple(args),
		)
		self.fn = fn
		self.args = tuple(args)
		self.kwargs = tuple(kwargs.items()) if kwargs else ()
		# object.__setattr__(self, "fn", fn)
		# object.__setattr__(self, "kwargs",
		# 	tuple(kwargs.items()) if kwargs else ()
		# )
		#object.__setattr__(self, "name", getattr(fn, "__name__", "<anon>"))
		self.name = getattr(fn, "__name__", "<anon>")


def lowerRef(root: Ref, lowerRefOpFn):
	"""
	Lowers a Ref DAG to a callable, made of lambdas.
	Calls lowerRefOpFn for any op it does not handle internally.

	lowerRefOpFn(op, lowered_args, ref) ->
		callable(buf) -> value
	"""

	cache = {}  # Ref -> lowered callable (CSE)

	def transform(ref: Ref):
		if ref in cache:
			return cache[ref]

		# Built-in leaves
		if ref.op == "const":
			fn = lambda buf: ref.value

		elif ref.op == "measure":
			i = ref.index
			fn = lambda buf, i=i: buf[i]

		# Built-in arithmetic
		elif ref.op in {"add", "sub", "mul", "div", "neg", "abs", "min"}:
			lowered = [transform(a) for a in ref.args]

			if ref.op == "add":
				fn = lambda buf: lowered[0](buf) + lowered[1](buf)
			elif ref.op == "sub":
				fn = lambda buf: lowered[0](buf) - lowered[1](buf)
			elif ref.op == "mul":
				fn = lambda buf: lowered[0](buf) * lowered[1](buf)
			elif ref.op == "div":
				fn = lambda buf: lowered[0](buf) / lowered[1](buf)
			elif ref.op == "neg":
				fn = lambda buf: -lowered[0](buf)
			elif ref.op == "abs":
				fn = lambda buf: jnp.abs(lowered[0](buf))
			elif ref.op == "min":
				fn = lambda buf: jnp.minimum(lowered[0](buf), lowered[1](buf))
			else:
				raise NotImplementedError

		elif isinstance(ref, FunctionRef):
			fn = ref.fn
			def lowered(buf):
				args = [f(buf) for f in lowered_args]
				kwargs = {k: f(buf) for k, f in ref.kwargs}
				return fn(*args, **kwargs)
			return lowered

		# Everything else: delegate
		else:
			lowered_args = [transform(a) for a in ref.args]
			fn = lowerRefOpFn(ref.op, lowered_args, ref)

		cache[ref] = fn
		return fn

	return transform(root)


class TraceError(Exception):
	pass

class SideEffectDetector(ast.NodeVisitor):
	"""TODO: maybe later try and detect invalid functions """
	def visit_Global(self, node): raise TraceError("don't do globals fool")
	def visit_Nonlocal(self, node): raise TraceError("Nonlocal not valid in "
	                                                 "jit code")
	def visit_Assign(self, node):
		if is_nonlocal_target(node): raise TraceError("jit functions must not have side effects, don't modify outside state")

IGNORE_CALLS = {
	"getMeasuredValue",
	"Ref",
	"ensure_ref",
	"wrapFunctionRefCall",
	# also ignore builtin constructors,
	# find a way to update with Ref known functions
	# and functions in a namespace (like jax)
	"tuple", "list", "dict"
}

class CallWrappingTransformer(ast.NodeTransformer):
	def __init__(self, ignoreNames: set[str]):
		self.ignoreNames = ignoreNames
		super().__init__()

	def visit_Call(self, node: ast.Call):
		# First transform children (arguments may themselves contain calls)
		self.generic_visit(node)

		# Only wrap normal calls: f(...)
		# Ignore:
		#  - attribute calls whose base is ignored (optional)
		#  - calls to whitelisted names
		calleeName = self._getCalleeName(node.func)

		if calleeName is None:
			# dynamic calls like (f())() –
			# TODO: probably do some jank stuff to trace this back to
			#  variable assignment but ONLY
			#   this,
			return node

		if calleeName in self.ignoreNames:
			return node

		# Rewrite f(a, b, ...) → wrapFunctionRefCall(f, a, b, ...)
		return ast.Call(
			func=ast.Name(id="wrapFunctionRefCall", ctx=ast.Load()),
			args=[node.func, *node.args],
			keywords=node.keywords,
		)

	def _getCalleeName(self, func):
		"""
		Returns function name if statically identifiable, else None.
		"""
		if isinstance(func, ast.Name):
			return func.id

		if isinstance(func, ast.Attribute):
			# foo.bar(...)
			return func.attr

		return None

def transformFunction(fn, ignoreNames):
	# Get source
	source = inspect.getsource(fn)
	source = textwrap.dedent(source)

	# Parse
	tree = ast.parse(source)

	# Transform
	transformer = CallWrappingTransformer(ignoreNames)
	newTree = transformer.visit(tree)
	ast.fix_missing_locations(newTree)

	# Compile into a new function
	code = compile(newTree, filename="<wrapped>", mode="exec")
	env = fn.__globals__.copy()
	env["wrapFunctionRefCall"] = wrapFunctionRefCall
	exec(code, env)

	# Replace function
	return env[fn.__name__]

def traceThrough(fn:T.Callable):
	fn._traceThrough = True
	return fn

def wrapFunctionRefCall(fn, cctx, *args, **kwargs):
	"""only recursively follow through functions explicitly decorated
	with @traceThrough - defaulting to tracing through everything gets
	dodgy if something somewhere happens to call open() or print() for
	example.
	should be fine to run over all functions in a namespace though, like
	util functions for the sim
	"""
	# If any arg is a Ref → return FunctionRef
	if hasattr(fn, "_traceThrough"):
		# Inline expansion
		return inlineFunction(fn, cctx, args, kwargs)
	else:
		return FunctionRef(fn, args, kwargs)
	# # Else: normal call
	# return fn(*args, **kwargs)

def inlineFunction(fn, cctx, args, kwargs):
	# Create new CompileContext
	cctx = CompileContext()

	# Substitute args into symbolic scope
	def boundFn():
		return fn(*args, **kwargs)

	rootRef = boundFn()   # build DAG for function body
	return rootRef

class CompileTimeBlock:
	"""code to be run only at compile time,
	nothing in here should get picked up in tracing.
	should fully resolve to static refs, generated only
	by this code
	"""
	def __init__(self, cctx:CompileContext):
		self.cctx = cctx

	def __enter__(self):
		self.cctx._compileOnlyBlockDepth += 1
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.cctx._compileOnlyBlockDepth -= 1

class CompileContext:
	"""
	overall object to use for compile-time ops in user functions -
	should be relatively readable:
	>>> mySimValue = cctx.getVal("measured", "mySimValue")
	in jax call, replace this with just an empty tuple or something.

	We could/should have a rule forbidding access to context values in
	anything other than outermost scope
	"""
	def __init__(self):
		self._compileOnlyBlockDepth = 0


	def atCompileTime(self)->CompileTimeBlock:
		return CompileTimeBlock(self)

	def ref(self, value):
		return Ref("const", value=value)

class JaxCompileContext(CompileContext):

	def __init__(self):
		super().__init__()
		self.keys = {}

	def getMeasuredValue(self, name):
		assert self._compileOnlyBlockDepth == 0, "wrapper functions not to be used in compile-time blocks"
		if name not in keys:
			index = len(self.keys)
			self.keys[name] = Ref(op="get", index=index)
		return self.keys[name]


def buildGraph(fn, compileContext=None):
	ctx = compileContext or CompileContext()

	wrapped = transformFunction(fn, IGNORE_CALLS)
	root = wrapped()   # author-time execution → Ref DAG

	keys = ctx.keys
	_measureCtx = None
	return root, keys

def userCurve(x):
	return x * x + 1.0
def userMap():
	a = getMeasuredValue("a")
	b = getMeasuredValue("b")
	c = userCurve(a)
	return c + b

if __name__ == '__main__':
	root, keys = buildGraph(userMap)
	print("Measured keys:", keys)
	print("Root Ref:", root)
	fn = lowerRef(root)

	# measurementBuffer matches keys order: ['a', 'b']
	measurementBuffer = jnp.array([3.0, 5.0])

	result = fn(measurementBuffer)
	print("Result:", result)
