from __future__ import annotations

import typing as T

from dataclasses import dataclass, is_dataclass, fields

import jax_dataclasses as jdc

from wpsim.kine.constraint import ConstraintState
from wpsim.kine.state import SubstepBoundData, SimStaticData
from wpsim.trace import CompileContext, RuntimeContext, LowerRefOpFn


def generate_pytree_indices(obj: Any):
	"""
	Generate flattened indices and index paths for a nested pytree.

	Returns:
		dict[str, (int, tuple[int, ...])]
	"""

	result: dict[str, tuple[int, tuple[int, ...]]] = {}
	flat_counter = 0

	def recurse(current, name_path, index_path):
		nonlocal flat_counter

		# Dataclass: recurse through fields
		if is_dataclass(current):
			for i, f in enumerate(fields(current)):
				value = getattr(current, f.name)
				recurse(
					value,
					name_path + (f.name,),
					index_path + (i,),
				)
			return

		# Dict: recurse through sorted keys
		if isinstance(current, dict):
			for key in sorted(current.keys()):
				value = current[key]
				recurse(
					value,
					name_path + (str(key),),
					index_path + (key,),
				)
			return

		# List or tuple: recurse by index
		if isinstance(current, (list, tuple)):
			for i, value in enumerate(current):
				recurse(
					value,
					name_path + (str(i),),
					index_path + (i,),
				)
			return

		# Leaf node: record entry
		path_str = ":".join(name_path)
		result[path_str] = (flat_counter, index_path)
		flat_counter += 1

	# Start recursion
	recurse(obj, (), ())

	return result


@jdc.pytree_dataclass(frozen=True)
class FixedLengthBuffer:
	vals : jnp.ndarray
	length : int

	def entry(self, index:int)->jnp.ndarray:
		return self.vals[index : index + self.length]

@jdc.pytree_dataclass(frozen=True)
class SpanBuffer:
	"""value buffer with indirection array of start and end indices"""
	vals : jnp.ndarray
	indices : jnp.ndarray # (N+1, start at 0)

	def start(self, index:int)->int:
		return self.indices[index]
	def end(self, index:int)->int:
		return self.indices[index + 1]
	def entry(self, index:int)->jnp.ndarray:
		return self.vals[self.start(index) : self.end(index)]

@dataclass # nb: NOT a jax dataclass, do NOT pass this into final jax code
class NamedEntryBuffer:
	buf : FixedLengthBuffer | SpanBuffer
	names : dict[str, int] # {name : index}


class JaxCompileContext(CompileContext):
	"""context for compile-time flattening in jax operations -
	retrieving named values and spans

	consider 2 layers of lookup : type name, entry name
	type name can be namespaced in string somehow, like
	"body:orientation", "constraint:hinge:alpha"

	"""
	def __init__(self):
		super().__init__()
		self.keys = {}



@jdc.pytree_dataclass(frozen=True)
class JaxRuntimeContext(RuntimeContext):
	"""
	keep tuple of buffer types passed in
	we should actually just have a flat map of all buffers across sim
	"""
	buffers : tuple[FixedLengthBuffer|SpanBuffer, ...]


class JaxLowerRefOpFn(LowerRefOpFn):
	"""finally bridge gap between string lookups and flat indices
	into buffer types and buffer entries
	"""
	def __init__(self, ):
		super().__init__()


"""valueRefs resolve to flat jax arrays (even if only single length)
"""
if T.TYPE_CHECKING:
	from jax import numpy as jnp
	ValueRef = jnp.ndarray

class NameBuffer:
	"""system to allow named values use at authoring time,
	but flattening to buffer indices before compilation.

	each name has a global index, and reserves a static number of float
	values
	"""
	def __init__(self):
		self.nameIndexMap : dict[str, int] = {}
		self.nameNValuesMap : dict[str, int] = {}

	def get(self, name:str, nValues:int, create=True) ->ValueRef[float,
	nValues]:
		if name in self.nameNValuesMap:
			if not create:
				raise KeyError(name + " not already found in string buffer")
			self.nameIndexMap[name] = len(self.nameNValuesMap)
			self.nameNValuesMap[name] = nValues
		return ValueRef(nValues, len(self.nameNValuesMap))


"""
individual functions passed params
as {localParamName : measuredValue or paramValue}

we don't allow chaining different measure tasks together,
instead we rely on XLA working out where code is duplicated within
separate functions and reusing it.

Freezing and caching function calls also naturally gives us cached values for duplicate calls

more generally we need an interface to get any named span of information-

we also need a general way to do "compile-time" operations, between
what you write in a function and what gets compiled , specifically
regarding strings

start, end = getSpan(spanName, typeName="ramp")

rampData = getRamp(rampName)
internally returns a tuple of all this ramp data

"""

@jdc.pytree_dataclass(frozen=True)
class MeasureFn:
	name : str
	resultSize : int = 1
	#params : dict[str, str] = jdc.field(default_factory=dict)
	params : tuple[str, ...] = ()

	@classmethod
	def measure(cls,
	            bs: SubstepBoundData,
	            cs: ConstraintState,
	            ss:SimStaticData,
	            struct: MeasureFn,
	            resultBuffer: jnp.ndarray,

	            )->jnp.ndarray:
		""""""


class ConstraintFn:
	def __init__(
		self, paramMap:dict[str, str],
	             ):
		self.paramMap = paramMap








