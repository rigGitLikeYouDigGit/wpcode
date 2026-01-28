from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from dataclasses import dataclass, field, asdict

import jax
from jax import numpy as jnp, jit

from wpsim.kine import state, constraint

"""author-facing side of the rig - store rich representation, then compile.
we still need to generate linearised structures as fast as possible
TODO: unify these data classes with the
plugin-side data classes 
"""

@dataclass(frozen=True)
class UserSimParamType:
	"""arbitrary simulation parameter -
	realistically typenames will just be Float, Vec3 or something
	"""
	typeName : str
	size : int = 1 # number of floats

@dataclass(frozen=True)
class UserMeasureFnType:
	"""arbitrary function that measures a set of params and returns a set of
	results back into simulation -
	number of results returned must be static
	"""
	typeName : str
	fn : T.Callable
	nParams : int
	nResults : int

@dataclass(frozen=True)
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
class WeightData:
	"""weight data for LBS on point-based geometry"""
	indices : jnp.ndarray # (nPoints, nWeights) may include bodies and
	# external joints
	weights : jnp.ndarray # (nPoints, nWeights) float16
	nWeights : int = 4

@dataclass
class BuilderMesh:
	"""mesh definition, assume TRI MESH ONLY
	vertices, points, faces
	hash used to check if different parts of mesh are dirty,
	then update only those regions
	"""
	name : str
	parent : str # always held under a body
	points : jnp.ndarray # (P, 3)
	indices : jnp.ndarray # (F, 3)
	driverIndex : int | None = None
	weightData : WeightData | None = None
	com : jnp.ndarray | None = None
	inertia : jnp.ndarray | None = None
	mass : float | None = None
	metaHash : int = 0


@dataclass
class BuilderNurbsCurve:
	"""nurbs curve definition"""
	name : str
	parent : str
	points : jnp.ndarray # (P, 3)
	degree : int
	knots : jnp.ndarray
	closed : bool = False
	driverIndex : int | None = None
	nPoints : int = 0
	nKnots : int = 0
	weightData : WeightData | None = None


@dataclass
class BuilderTransform:
	"""transform definition - use to define aux transforms
	in space of bodies
	"""
	name : str
	parent : str | None
	mat : jnp.ndarray # (4,4) rest matrix


@dataclass
class BuilderBody:
	"""rigid body definition -
	we assume for the purpose of finding inertia axes, collision etc, each
	body has
	one main 'representative' mesh to define its centre of mass,
	inertia tensor, etc
	"""
	name : str
	restPos : jnp.ndarray
	restQuat : jnp.ndarray
	meshMap : dict[str, BuilderMesh] = field(default_factory=dict)
	curveMap : dict[str, BuilderNurbsCurve] = field(default_factory=dict)
	transformMap : dict[str, BuilderTransform] = field(default_factory=dict)
	com : jnp.ndarray = jnp.zeros((3,))
	inertia : jnp.ndarray = jnp.eye(3)
	mass : float = 1.0,
	active : int = 1 # 0 disabled, 1 active, 2 static
	damping : float = 0.0 # linear damping factor



class SimBuilder:
	"""user-facing side of rig - store rich representation, then compile

	simName will usually just be the name of the character -
	2 named sims can never interact, but can run in parallel

	should be able to update only single span in buffer when input changes -


	"""

	def __init__(self, name:str):
		self.name = name
		# self.simParams : dict[UserSimParamType, dict[str, jnp.ndarray]] = {}
		# self.measuredFns : dict[UserMeasureFnType, dict[str, jnp.ndarray]] = {}
		# self.constraints : dict[UserConstraintFnType, dict[str, jnp.ndarray]] = {}
		# self.meshes : dict[str, BuilderMesh] = {}
		# self.weightedMeshes : dict[str, BuilderMesh] = {}
		self.builderBodyMap = {} # type: dict[str, BuilderBody]

		# compiled function building blocks

		# DATA TO UPLOAD
		self.meshData : state.MeshBuffers = None
		self.simStaticData : state.SimStaticData = None
		self.frameData : state.FrameBoundData = None
		self.substepData : state.SubstepBoundData = None
		self.dynamicData : state.DynamicData = None


	def bind(self):
		"""build all the data structures needed for sim run -
		mesh buffers, body state buffers, constraint plans etc
		run once at bind frame to allocate buffers and topology of
		simulation
		"""
		log(f"Binding sim '{self.name}' with {len(self.builderBodyMap)} bodies")

	def simFrame(self):
		"""simulate a single frame - run substeps as needed
		"""
		log(f"Simulating frame for sim '{self.name}'")


if __name__ == '__main__':

	"""test sketch for building a sim system and then linearising - 
	still on the knee, it's the whole point of this project
	
	bodies:
	bFemur
	bTibia
	
	can we just do a single way of constraining?
	
	orientation constraint between them, 2 axes stiff, one controlled with 
	constraint ramp - relative rotation in X is measured.
	feed that into mutual point-on-curve constraint on 2 nurbs curves around  
	"""






