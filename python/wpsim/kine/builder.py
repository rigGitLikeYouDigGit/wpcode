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
class UserMesh:
	"""mesh definition, assume TRI MESH ONLY
	vertices, points, faces
	hash used to check if different parts of mesh are dirty,
	then update only those regions
	"""
	name : str
	points : jnp.ndarray # (P, 3)
	indices : jnp.ndarray # (F, 3)
	pointHash: int = 0
	indicesHash: int = 0
	driverIndex : int | None = None
	weightData : WeightData | None = None
	com : jnp.ndarray | None = None
	inertia : jnp.ndarray | None = None
	mass : float | None = None
	metaHash : int = 0



@dataclass
class UserNurbsCurve:
	"""nurbs curve definition"""
	name : str
	points : jnp.ndarray # (P, 3)
	degree : int
	knots : jnp.ndarray
	closed : bool = False
	driverIndex : int | None = None
	weightData : WeightData | None = None


@dataclass
class Body:
	"""rigid body definition -
	we assume for the purpose of finding inertia axes, collision etc, each
	body has
	one main 'representative' mesh to define its centre of mass,
	inertia tensor, etc
	"""
	name : str
	restPos : jnp.ndarray
	restQuat : jnp.ndarray
	meshName : str



class SimBuilder:
	"""user-facing side of rig - store rich representation, then compile

	simName will usually just be the name of the character -
	2 named sims can never interact, but can run in parallel

	should be able to update only single span in buffer when input changes -


	"""

	def __init__(self, simName:str):
		self.simName = simName
		self.simParams : dict[UserSimParamType, dict[str, jnp.ndarray]] = {}
		self.measuredFns : dict[UserMeasureFnType, dict[str, jnp.ndarray]] = {}
		self.constraints : dict[UserConstraintFnType, dict[str, jnp.ndarray]] = {}
		self.meshes : dict[str, UserMesh] = {}
		self.weightedMeshes : dict[str, UserMesh] = {}

		# compiled function building blocks

		# DATA TO UPLOAD
		self.meshData : state.MeshBuffers = None
		self.simStaticData : state.SimStaticData = None
		self.frameData : state.FrameBoundData = None
		self.substepData : state.SubstepBoundData = None
		self.dynamicData : state.DynamicData = None


	def syncMeshes(self, meshes:list[UserMesh]):
		"""update mesh definitions -
		for each, check if positions changed, or topo changed
		any meshes not found in given meshes will be removed
		"""
		topoChanged = False
		nameSet = {mesh.name for mesh in meshes}
		for i, mesh in enumerate(meshes):
			if mesh.name not in self.meshes:
				topoChanged = True
				self.meshes[mesh.name] = mesh
				continue







	def syncBodies(self, bodies:list[Body]):
		"""update body definitions"""
		self.bodies = bodies

	def include(self, other:SimBuilder):
		"""merge other sim into this one
		"""


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






