from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
from dataclasses import dataclass, field, asdict

import jax
from jax import numpy as jnp, jit
import jax_dataclasses as jdc

from wpsim.kine import state, constraint

# @dataclass(frozen=True)
# class UserFnType:
# 	"""arbitrary function to be jitted from simulation
# 	"""
# 	typeName : str
# 	fn : T.Callable
# 	nParams : int
# 	nResults : int

@jdc.pytree_dataclass
class UserConstraintType:
	"""each constraint must touch a static (maximum) number
	of bodies, sim params, measured values, and params"""
	typeName : str
	nBodies : int
	nParams : int


	def compute(self)->float|jnp.array:
		"""override this in your constraint - lambda has one dimension for
		each measured value"""
		raise NotImplementedError()


