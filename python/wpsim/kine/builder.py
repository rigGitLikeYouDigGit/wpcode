from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from dataclasses import dataclass, field

import jax
from jax import numpy as jnp, jit

"""author-facing side of the rig - store rich representation, then compile"""

@dataclass
class UserSimParamType:
	"""arbitrary simulation parameter -
	realistically typenames will just be Float, Vec3 or something
	"""
	typeName : str
	size : int = 1 # number of floats

@dataclass
class UserMeasureFnType:
	"""arbitrary function that measures a set of params and returns a set of
	results back into simulation -
	number of results returned must be static
	"""
	typeName : str
	fn : T.Callable
	nParams : int
	nResults : int

@dataclass
class UserConstraintFnType:
	"""each constraint must touch a static (maximum) number
	of bodies, sim params, measured values, and params"""
	typeName : str
	fn : T.Callable
	nBodies : int
	nSimParams : int
	nMeasuredValues : int
	nParams : int

@dataclass
class Body:
	"""rigid body definition"""
	name : str
	restPos : jnp.ndarray
	restQuat : jnp.ndarray


class SimBuilder:
	"""user-facing side of rig - store rich representation, then compile

	simName will usually just be the name of the character -
	2 named sims can never interact, but can run in parallel
	"""

	def __init__(self, simName:str):
		self.simName = simName
		self.simParams : dict[UserSimParamType, dict[str, jnp.ndarray]] = {}
		self.measuredFns : dict[UserMeasureFnType, dict[str, jnp.ndarray]] = {}
		self.constraints : dict[UserConstraintFnType, dict[str, jnp.ndarray]] = {}


	def include(self, other:SimBuilder):
		"""merge other sim into this one
		"""



