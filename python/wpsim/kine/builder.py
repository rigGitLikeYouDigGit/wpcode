from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
import sys
from importlib import import_module, reload
from collections import defaultdict

from dataclasses import dataclass, field, asdict

import jax
from jax import numpy as jnp, jit
import jax_dataclasses as jdc

from wplib.codegen import strtemplate
from wplib import trace as wptrace

from wpsim.kine import state, constraint, sim

"""author-facing side of the rig - store rich representation, then compile.
we still need to generate linearised structures as fast as possible
TODO: unify these data classes with the
plugin-side data classes 
"""

jnp.ndarray.__hash__ = lambda self: hash(self.tobytes())


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
class BuilderSimParam:
	"""simulation parameter definition -
	sim params are global to the sim, and can be
	used in constraints etc
	"""
	name : str
	length : int # 1, 4, or 16
	defaultValue : jnp.ndarray

@dataclass
class BuilderRamp:
	"""ramp definition - used to define parameter ramps over time.
	we sample each ramp at 32 points equally spaced over 0-1 time
	"""
	name : str
	points : jnp.ndarray # (32, ) float32

@dataclass
class BuilderMultiRamps:
	"""ramp definition - used to define parameter ramps over time.
	we sample each ramp at 32 points equally spaced over 0-1 time
	"""
	ramps : list[BuilderRamp]

@dataclass
class BuilderForceField:
	"""force field definition - global to sim,
	applied to all bodies
	TODO: add falloff, noise, turbulence etc"""
	name : str
	shapeType : int # e.g. 'sphere', 'box', 'infinite'
	radius : float
	height : float
	halfExtents : jnp.ndarray # (3, )


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
	com: jnp.ndarray
	inertia: jnp.ndarray
	meshMap : dict[str, BuilderMesh] = field(default_factory=dict)
	curveMap : dict[str, BuilderNurbsCurve] = field(default_factory=dict)
	transformMap : dict[str, BuilderTransform] = field(default_factory=dict)
	mass : float = 1.0,
	active : int = 1 # 0 disabled, 1 active, 2 static
	damping : T.Sequence[float] = (
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.0) # linear and angular damping

@dataclass
class BuilderConstraint:
	"""constraint definition -
	we assume each constraint has a single 'representative' function that
	measures the constraint value, which is used for constraint solving and
	also exposed to the user for plotting, debugging, etc.

	If changes are found per frame, send in form of (name, type, {paramName : paramValue})
	"""
	name : str
	type : str # e.g. 'pointOnCurve', 'mutualPointOnCurve', 'orientation', etc
	params : dict[str, jnp.ndarray | float | None | str] # parameters for
		# constraint function, e.g. stiffness, target value, etc
	# if value is a string, expect it to be a parametre name OR an exp string
	# that returns a new val based on vars





@dataclass
class BuilderVar:
	"""named variable - combined with params during build?
	"""
	name : str
	constant : float | jnp.ndarray | None = None # if constant given,
	# use this as default value
	exp : str | None = None # None if constant, otherwise an expression to compute this variable from other variables and params
	size : int = 4

NVars = int

@jdc.pytree_dataclass
class SimVars:
	"""arbitrary variables measured from the sim to use in constraints
	for simplicity we do everything as at least float4s"""
	float4Vars : jnp.ndarray[NVars, 4]
	floatVarNameIndexMap : dict[str, int] # map from variable name to index in float4Vars




class SimBuilder:
	"""user-facing side of rig - store rich representation, then compile

	simName will usually just be the name of the character -
	2 named sims can never interact, but can run in parallel

	should be able to update only single span in buffer when input changes -

	refer to meshes owned by body by "bodyName.meshName", and constraints by "constraintName.paramName"

	"""

	constraintNameTypeMap = {
		"pointConstraint" : constraint.PointConstraintBucket
	}

	def __init__(self, name:str):
		self.name = name
		# self.simParams : dict[UserSimParamType, dict[str, jnp.ndarray]] = {}
		# self.measuredFns : dict[UserMeasureFnType, dict[str, jnp.ndarray]] = {}
		# self.constraints : dict[UserConstraintFnType, dict[str, jnp.ndarray]] = {}
		# self.meshes : dict[str, BuilderMesh] = {}
		# self.weightedMeshes : dict[str, BuilderMesh] = {}
		self.builderBodyMap = {} # type: dict[str, BuilderBody]
		self.bodyNameToIndexMap = {}
		self.builderConstraintMap : dict[str, BuilderConstraint] = {}

		# compiled function building blocks

		"""self.varToBucketConstraintParamMap = {
			varIndex : (constraint type, constraint name, param name)
		}"""
		self.varToConstraintParamMap : dict[
			int, tuple[str, str, str]] = {}

		self.constraintNameMap = dict[str, tuple[str, int]]

		self.constraintTypeMap : dict[str, constraint.ConstraintBucket] = {}

		# list of python modules to check for user constraints and var functions
		self.pythonPathsToLoad = []
		self.loadedModules = []

		# variable build maps
		# { var size : [var values for all vars of this size] }
		self.varSizeArrMap : dict[int, jnp.ndarray] = {}
		# map of {varName : (var size, var index within size arr) }
		self.varNameIndexMap : dict[str, tuple[int, int]] = {}
		# { var name : exp string }
		self.varNameExpMap : dict[str, str] = {}

		# DATA TO UPLOAD
		self.meshData : state.MeshBuffers = None
		self.simStaticData : state.SimStaticData = None
		self.frameData : state.FrameBoundData = None
		self.substepData : state.BodyState = None
		self.dynamicData : state.DynamicData = None

	def reloadModules(self):
		"""reload all user modules to check for new constraint and var functions
		this is a bit hacky but allows us to not have to worry about reloads during development"""
		self.constraintNameTypeMap.clear()
		self.loadedModules = []
		for path in self.pythonPathsToLoad:
			if path in sys.modules:
				module = reload(sys.modules[path])
			else:
				module = import_module(path)
			self.loadedModules.append(module)
			# scan for any attributes on this module that are constraint buckets, and register them
			for attrName in dir(module):
				attr = getattr(module, attrName)
				if (isinstance(attr, type) and
						constraint.ConstraintBucket.__name__ in [base.__name__ for base in attr.__bases__]):
					self.registerConstraintType(attr)


	def registerConstraintType(self, constraintCls:type):
		"""register a constraint type for use in the sim - this is needed to
		map from user-facing constraint definitions to the compiled constraint
		buckets that run in the sim loop"""
		self.constraintNameTypeMap[constraintCls.__name__] = constraintCls

	def constraintTypeForName(self, name:str):
		"""get the constraint type for the given name, to find the right bucket
		type to compile to"""
		return self.constraintNameTypeMap[name]

	def buildMeshes(self, body:BuilderBody, meshes:list[BuilderMesh]):
		pass

	def buildBodies(self, builderBodies:list[BuilderBody])->tuple[state.BodyState, state.BodyMetadata, state.GeometryBuffers,]:
		"""from individual body dataclasses, compile full body states and
		geometry buffers for the sim
		"""
		self.bodyNameToIndexMap = {}
		position = []  # (N, 3)
		orientation = [] # (N, 4)  unit quaternions
		# Velocities
		linearVelocity = [] # (N, 3)
		angularVelocity = []  # (N, 3)
		# Mass properties (world-constant or body-constant)
		invMass = [] # (N,)
		invInertiaBody = [] # (N, 3)  diagonal inertia in body frame
		# forces
		force = [] # (N, 3)
		torque = [] # (N, 3)
		damping = []

		# transform buffers
		tfMatrices = []
		tfOffsets = [0]

		# mesh buffers
		points = []
		triIndices = []
		pointOffsets = [0]
		triOffsets = [0]

		for i, body in enumerate(builderBodies):
			self.bodyNameToIndexMap[body.name] = i
			position.append(body.restPos)
			orientation.append(body.restQuat)
			linearVelocity.append(jnp.zeros(3))
			angularVelocity.append(jnp.zeros(3))
			invMass.append(1.0 / body.mass if body.mass > 0 else 0.00001)
			invInertiaBody.append(1.0 / body.inertia)
			force.append(jnp.zeros(3))
			torque.append(jnp.zeros(3))
			damping.append(body.damping)

			for tfName, tf in body.transformMap.items():
				tfMatrices.append(tf.mat)
			tfOffsets.append(len(tfMatrices))

			for meshName, mesh in body.meshMap.items():
				points.extend(mesh.points)
				triIndices.extend(mesh.indices)
				pointOffsets.append(len(pointOffsets))
				triOffsets.append(len(triOffsets))
				#self.buildMeshes(body, mesh)

		bs = state.BodyState(
			position=jnp.array(position),
			orientation=jnp.array(orientation),
			linearVelocity=jnp.array(linearVelocity),
			angularVelocity=jnp.array(angularVelocity),
			invMass=jnp.array(invMass),
			invInertiaBody=jnp.array(invInertiaBody),
			force=jnp.array(force),
			torque=jnp.array(torque),
			damping=jnp.array(damping),
		)
		bodyMetadata = state.BodyMetadata(
			restPosition=jnp.array(position),
			restOrientation=jnp.array(orientation),
		)
		geometryBuffers = state.GeometryBuffers(
			tfs=state.TransformBuffers(
				matrices=tfMatrices,
				offsets=jnp.array(tfOffsets),
			),
			meshes=state.MeshBuffers(
				points=jnp.array(points),
				triIndices=jnp.array(triIndices),
				pointOffsets=jnp.array(pointOffsets),
				triOffsets=jnp.array(triOffsets),
			)
		)
		return bs, bodyMetadata, geometryBuffers


	def addVar(self, varName, size, defaultValue=None, exp=None)->tuple[int, int]:
		"""add a variable to the sim - this can be used in constraint definitions
		to create constraints that depend on arbitrary variables measured from
		the sim, or computed from other variables and params

		return the index of the var"""
		if defaultValue is None:
			if size == 1:
				defaultValue = 0.0
			else:
				defaultValue = jnp.zeros(size)
		self.varSizeArrMap.setdefault(size, []).append(defaultValue)
		self.varNameIndexMap[varName] = (
			size, len(self.varSizeArrMap[size]) - 1 )
		if exp is not None:
			self.varNameExpMap[varName] = exp
		return (size, len(self.varSizeArrMap[size]) - 1)

	def build(self,
	          builderBodies: list[BuilderBody],
	          builderVars: list[BuilderVar],
	          builderConstraints: list[BuilderConstraint],
	          ):
		bodyState, bodyMetadata, geometryBuffers = self.buildBodies(
			builderBodies)

		# { name : (constraint type, constraint index ) )
		self.constraintNameMap : dict[str, tuple[str, int]] = {}

		self.varSizeArrMap.clear()
		self.varNameIndexMap.clear()
		self.varNameExpMap.clear()

		for i, var in enumerate(builderVars):
			self.addVar(var.name, var.size, var.constant, var.exp)

		"""below is a bit verbose in code, but we're building 
		arrays potentially piecemeal from different sources"""
		constraintTypeParamArrMap : dict[str, dict[str, list]] = {}

		for i, cn in enumerate(builderConstraints):
			assert cn.type in self.constraintNameTypeMap, f"Constraint type '{cn.type}' not registered"
			constraintType = self.constraintTypeForName(cn.type)
			self.constraintNameMap[cn.name] = (cn.type, i)
			constraintTypeParamArrMap[cn.type].setdefault(
				{f : [] for f in constraintType.fieldNames()}
			)
			for k, v in cn.params.items():
				if k in constraintType.bodyParams():
					try:
						bodyIndex = self.bodyNameToIndexMap[v]
					except KeyError as e:
						print("ERROR in constraint", cn.name, " - body name not found:", v)
						raise e
					constraintTypeParamArrMap[cn.type][k].append(bodyIndex)
					continue

				if isinstance(v, str):
					# if it's literally the name of a param, sub its index
					if v in self.varNameIndexMap:
						size, varIndex = self.varNameIndexMap[v]
					else:
						# otherwise it's an expression to eval at runtime
						varName = f"{cn.name}_{k}"
						size, varIndex = self.addVar(varName,
						            constraintType.fieldSizeMap()[k],
						            exp=v)
					constraintTypeParamArrMap[cn.type][k].append(varIndex)

		for cnTypeStr, paramArrMap in constraintTypeParamArrMap.items():
			cnType = self.constraintTypeForName(cnTypeStr)
			self.constraintTypeMap[cnTypeStr] = cnType(
				**{k : jnp.array(v) for k, v in paramArrMap.items()}
			)

	VarDoubleBufferT = tuple[
		dict[int, jnp.ndarray], ...
	]

	def demoVarExpString(self):
		"""demo of how we can use exp strings to compute variables from other vars and params
		double buffer used to access prev values"""

		def runVars(varSizeArrMapDoubleBuffer:SimBuilder.VarDoubleBufferT,
		            writeIndex:int)->SimBuilder.VarDoubleBufferT:
			varSizeArrMap = varSizeArrMapDoubleBuffer[not writeIndex]
			varA = ...
			varB = varA * 2
			varC = (varA, varB, 0)

			size_1_vals = (varA, varB)
			size_1_indices = jnp.arange(len(self.varSizeArrMap[1]))
			self.varSizeArrMap[1].at[size_1_indices].set(size_1_vals)

			size_3_vals = (varC, )
			size_3_indices = jnp.arange(len(self.varSizeArrMap[3]))
			self.varSizeArrMap[3].at[size_3_indices].set(size_3_vals)

			# you absolute moron you don't need this, you have the index
			# if writeIndex:
			# 	return varSizeArrMapDoubleBuffer[0], varSizeArrMap
			# else:
			# 	return varSizeArrMap, varSizeArrMapDoubleBuffer[1]
			return varSizeArrMap, varSizeArrMapDoubleBuffer[not writeIndex]

	def genVarExpString(self):
		"""complex var syntax is gonna be clunky for now
		var exps have access to:
		VARIDX[] = name index map
		BODYIDX[] = body name index map


		"""
		# generate initial assignment blocks
		lines = []
		sizeAssignIndexMap = defaultdict(list)
		sizeAssignVars = defaultdict(list)
		lines.append(strtemplate.Comment("Generating arg processing lines"))
		lines.append(strtemplate.Comment("last frame's new vals are this "
		                                 "frame's previous"))
		lines.append(strtemplate.Assign(
			"PREVVAR", strtemplate.Literal("_varSizeArrMapDoubleBuffer[1]")
		))
		lines.append(strtemplate.Assign(
			"varSizeArrMap", strtemplate.Literal(
				"_varSizeArrMapDoubleBuffer[0]")
		))

		lines.append(strtemplate.Comment("Generating variables"))
		for k, (size, index) in self.varNameIndexMap.items():
			if k in self.varNameExpMap:
				lines.append(strtemplate.Assign(
					k, self.varNameExpMap[k]
				))
			else:
				lines.append(strtemplate.Assign(
					k,
					strtemplate.Literal(f"varSizeArrMap[{size}][{index}]")
				))
				sizeAssignVars[size].append(k)
				sizeAssignIndexMap[size].append(index)

		lines.append(strtemplate.Comment("Final assignments"))
		# generate final gather lines
		for size in sizeAssignIndexMap:
			valVarName = f"size{size}Vals"
			indexVarName = f"size{size}Indices"
			lines.append(strtemplate.Assign(valVarName, strtemplate.Literal(
				f"({', '.join(sizeAssignVars[size])})")))
			lines.append(strtemplate.Assign(indexVarName, strtemplate.Literal(
				str(tuple(sizeAssignIndexMap[size])))))
			lines.append(f"varSizeArrMap[{size}] = varSizeArrMap[{size}].at["
			             f"{indexVarName}].set({valVarName})")

		# make return statement
		# lines.append(strtemplate.IfBlock(
		# 	[(strtemplate.Literal("_writeIndex"),
		# 	strtemplate.Literal(
		# 		"return varSizeArrMap, _varSizeArrMapDoubleBuffer[0]"))
		# 	 ],
		# 	(strtemplate.Literal(
		# 		"return _varSizeArrMapDoubleBuffer[1], varSizeArrMap"
		# 	), )
		# ))

		lines.append(strtemplate.Literal(
			"return PREVVAR, varSizeArrMap"
		))

		result = "\n".join(str(l) for l in lines)
		print("gen var exp string:\n", result)


		fnDef = strtemplate.FunctionTemplate(
			"runVars",
			[
				(
					"_varSizeArrMapDoubleBuffer",
					"SimBuilder.VarDoubleBufferT"
				),
				# (
				# 	"_writeIndex",
				# 	"int"
				# ),
				("BODYIDX", "dict[str, int]"),
				("VARIDX", "dict[str, tuple[int, int]]"),
				("SIM", "sim.KineSim" )
			],
			fnKwargs={},
			fnBody=strtemplate.TextBlock(
				result
			),
			returnType="SimBuilder.VarDoubleBufferT"
		)
		return str(fnDef)

	def compileVarExpString(self, expStr:str, fnGlobals:dict):
		fn = wptrace.compileCodeToFunction(expStr, fnGlobals,
		                                   "runVars")
		compFn = jax.jit(
			fn,
		    static_argnames=["BODYIDX", "VARIDX",],

		)
		return compFn

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






