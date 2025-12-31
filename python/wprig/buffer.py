from __future__ import annotations
import types, typing as T
import pprint
from os import name

from wplib import log

import jax
from jax import numpy as jnp
import jax_dataclasses as jdc

from wprig.constraint import ConstraintState
from wprig.state import BodyState, SimSettings


class ValueRef:
	def __init__(self, index:int, length:int) -> None:
		self.index = index
		self.length = length
	def resolve(self, buffer) -> list[float]:
		return buffer[self.index:self.index+self.length]

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
separate functions and reusing it

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
	            bs: BodyState,
	            cs: ConstraintState,
	            ss:SimSettings,
	            struct: MeasureFn,
	            resultBuffer: jnp.ndarray,

	            )->jnp.ndarray:
		""""""


class ConstraintFn:
	def __init__(
		self, paramMap:dict[str, str],
	             ):
		self.paramMap = paramMap








