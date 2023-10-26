

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, Sdf

from wpm import core
from wpm.constant import *
from wpm import om, cmds


"""libs for converting between maya and usd types"""

def usdAPath(tokens:list[str])->str:
	"""return an absolute usd prim path from the given tokens"""
	return "/" + "/".join(tokens)

def currentMayaScenePath()->Path:
	"""return the current maya scene path"""
	return Path(cmds.file(q=True, sn=True))


def applyUsdXformOpToTransform(op:UsdGeom.XformOp, mfn:om.MFnTransform):
	"""apply the given usd xform op to the given maya transform"""
	if op.GetOpType() == UsdGeom.XformOp.TypeScale:
		mfn.setScale(op.Get()) # accepts list, not MVector :)))
	elif op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
		mfn.setTranslation(om.MVector(op.Get()), om.MSpace.kTransform)
	elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
		mfn.setRotation(om.MEulerRotation(op.Get()), om.MSpace.kTransform)

class UsdConverter:
	""" base class for converting between USD prim and maya MFn"""

	mfnType : type[om.MFn] = None
	usdPrimName : str = None

	@classmethod
	def toUsdPrim(cls, mfn:mfnType, stage:Usd.Stage, pathTokens:list[str])->Usd.Prim:
		prim = stage.DefinePrim(usdAPath(pathTokens), cls.usdPrimName)
		raise NotImplementedError


	@classmethod
	def toMFn(cls, prim:Usd.Prim, stage:Usd.Stage,
	          parentMfn:om.MFnTransform=None)->mfnType:
		"""return a new MFn and create maya node matching given prim"""
		mfn = cls.mfnType()
		obj = mfn.create()
		mfn.setName(prim.GetName())
		raise NotImplementedError



class UsdXformConverter(UsdConverter):
	"""convert transform nodes to USD Xform"""
	mfnType = om.MFnTransform
	usdPrimName = "Xform"

	@classmethod
	def toUsdPrim(cls, mfn:mfnType, stage:Usd.Stage, pathTokens:list[str])->Usd.Prim:
		# define xform, extract xform ops from maya attributes
		prim : UsdGeom.Xform = UsdGeom.Xform.Define(stage, usdAPath(pathTokens))
		ops : list[UsdGeom.XformOp] = []
		# scale
		scale = mfn.scale()
		if om.MVector(scale) != ONE_VECTOR:
			op : UsdGeom.XformOp = prim.AddScaleOp()
			op.Set(tuple(scale))
		# translate
		translate = mfn.translation(om.MSpace.kTransform)
		if translate != ZERO_VECTOR:
			op = prim.AddTranslateOp()
			op.Set(tuple(translate))

		# rotate
		rotate = mfn.rotation(om.MSpace.kTransform)
		if rotate != om.MEulerRotation(ZERO_VECTOR):
			op = prim.AddRotateXYZOp()
			op.Set(tuple(rotate))

		return prim


	@classmethod
	def toMFn(cls, prim:Usd.Prim, stage:Usd.Stage,
	          parentMfn:om.MFnTransform=None)->mfnType:
		"""return a new MFnTransform from the given prim"""
		mfn = cls.mfnType()
		obj = mfn.create()
		mfn.setName(prim.GetName())
		if parentMfn:
			parentMfn.addChild(obj)

		# apply xform ops
		prim : UsdGeom.Xform = UsdGeom.Xform(prim)
		for op in prim.GetOrderedXformOps():
			applyUsdXformOpToTransform(op, mfn)

		return mfn

class UsdMeshConverter(UsdConverter):
	"""convert mesh nodes to USD Mesh"""
	mfnType = om.MFnMesh
	usdPrimName = "Mesh"

	@classmethod
	def toUsdPrim(cls, mfn:mfnType, stage:Usd.Stage, pathTokens:list[str])->Usd.Prim:
		# define mesh, get mesh data from MFn
		prim : UsdGeom.Mesh = UsdGeom.Mesh.Define(stage, usdAPath(pathTokens))

		# topology
		faceCounts, faceIndices = mfn.getVertices()
		prim.CreateFaceVertexCountsAttr(
			faceCounts,
		)
		prim.CreateFaceVertexIndicesAttr(
			faceIndices,
		)

		# point coords
		prim.CreatePointsAttr(
			np.array(mfn.getPoints(), dtype=np.float32)[:, :3],
		)

		# normals
		normals = mfn.getNormals()
		prim.CreateNormalsAttr(
			np.array(normals, dtype=np.float32)[:, :3],
		)

		# uv
		# uvCounts, uvIds = mfn.getAssignedUVs()

		return prim


	@classmethod
	def toMFn(cls, prim:Usd.Prim, stage:Usd.Stage,
	          parentMfn:om.MFnTransform=None)->mfnType:
		"""return a new MFnMesh from the given prim"""
		mfn = cls.mfnType()

		# get mesh data from prim
		prim : UsdGeom.Mesh = UsdGeom.Mesh(prim)
		pointCoords = tuple(map(om.MPoint, prim.GetPointsAttr().Get()))
		faceCounts = prim.GetFaceVertexCountsAttr().Get()
		faceIndices = prim.GetFaceVertexIndicesAttr().Get()


		obj = mfn.create(
			pointCoords,
			faceCounts,
			faceIndices,
			parent=parentMfn
		)
		mfn.setName(prim.GetName())

		return mfn

class UsdNurbsCurveConverter(UsdConverter):
	"""convert curve shape nodes to USD NurbsCurves.
	For now, we only support one curve per prim - grooms
	might be their own thing later."""
	mfnType = om.MFnNurbsCurve
	usdPrimName = "NurbsCurves"

	@classmethod
	def toUsdPrim(cls, mfn:mfnType, stage:Usd.Stage, pathTokens:list[str]) ->Usd.Prim:
		prim : Usd.Prim = stage.DefinePrim(usdAPath(pathTokens), cls.usdPrimName)
		# coords
		prim.GetAttribute("points").Set(
			np.array(mfn.cvPositions(), dtype=np.float32)[:, :3],
		)
		# knots
		prim.GetAttribute("knots").Set(
			np.array(mfn.knots(), dtype=np.float32),
		)
		# orders
		prim.GetAttribute("order").Set(
			np.array(mfn.degree, dtype=np.int32),
		)
		# form
		prim.GetAttribute("curveVertexCounts").Set(
			np.array(mfn.numCVs, dtype=np.int32),
		)
		return prim

	@classmethod
	def toMFn(cls, prim:Usd.Prim, stage:Usd.Stage,
	          parentMfn:om.MFnTransform=None)->mfnType:
		"""return a new MFnNurbsCurve from the given prim"""
		mfn = cls.mfnType()
		#prim : UsdGeom.NurbsCurves = UsdGeom.NurbsCurves(prim)
		obj = mfn.create(
			prim.GetAttribute("points").Get(),
			prim.GetAttribute("knots").Get(),
			prim.GetAttribute("order").Get(),
			prim.GetAttribute("curveVertexCounts").Get(),
			parent=parentMfn,
		)
		mfn.setName(prim.GetName())
		return mfn

class UsdMeshConverter(UsdConverter):
	"""convert mesh nodes to USD Mesh"""
	mfnType = om.MFnMesh
	usdPrimName = "Mesh"

	@classmethod
	def uvPrimNameFromSetName(cls, setName:str):
		return "uv_" + setName
	@classmethod
	def uvSetNameFromPrimName(cls, primName:str):
		return primName[3:]

	@classmethod
	def toUsdPrim(cls, mfn:mfnType, stage:Usd.Stage, pathTokens:list[str])->Usd.Prim:
		"""need a proper API to define arbitrary per-point, per-vertex data,
		 groups, UV sets, etc"""
		# define mesh, get mesh data from MFn
		prim : Usd.Prim = stage.DefinePrim(usdAPath(pathTokens), cls.usdPrimName)

		# topology
		faceCounts, faceIndices = mfn.getVertices()
		prim.GetAttribute("faceVertexCounts").Set(faceCounts)
		prim.GetAttribute("faceVertexIndices").Set(faceIndices)

		# point coords
		prim.GetAttribute("points").Set(
			np.array(mfn.getPoints(), dtype=np.float32)[:, :3],
		)

		# normals
		prim.GetAttribute("normals").Set(
			np.array(mfn.getNormals(), dtype=np.float32)[:, :3],
		)

		# uvs - custom support for uv sets
		uvSetNamesAttr = prim.CreateAttribute("uvSetNames", Sdf.ValueTypeNames.TokenArray)
		uvSetNamesAttr.Set(mfn.getUVSetNames())

		for name in mfn.getUVSetNames():
			uvCounts, uvIds = mfn.getAssignedUVs(name)
			prim.GetAttribute(
				cls.uvPrimNameFromSetName(name),
				Sdf.ValueTypeNames.TexCoord2fArray,
			).Set(
				np.array(mfn.getUVs(name), dtype=np.float32)[:, :2],
			)

		return prim

	@classmethod
	def toMFn(cls, prim:Usd.Prim, stage:Usd.Stage,
	          parentMfn:om.MFnTransform=None)->mfnType:
		"""return a new MFnMesh from the given prim"""
		mfn = cls.mfnType()

		# get mesh data from prim
		prim : UsdGeom.Mesh = UsdGeom.Mesh(prim)
		pointCoords = tuple(map(om.MPoint, prim.GetPointsAttr().Get()))
		faceCounts = prim.GetFaceVertexCountsAttr().Get()
		faceIndices = prim.GetFaceVertexIndicesAttr().Get()
		normals = tuple(map(om.MVector, prim.GetNormalsAttr().Get()))

		# create mesh
		obj = mfn.create(
			pointCoords,
			faceCounts,
			faceIndices,

			parent=parentMfn,
		)
		mfn.setName(prim.GetName())

		# set normals
		mfn.setFaceVertexNormals(
			normals,
			faceIndices,
			faceCounts,
		)
		return mfn


toUsdMap = {i.mfnType : i for i in UsdConverter.__subclasses__()}
toMFnMap = {i.usdPrimName : i for i in UsdConverter.__subclasses__()}


def _nodeToUsdPrim(mfn:om.MFnDependencyNode, stage:Usd.Stage, pathTokens:list[str])->Usd.Prim:
	"""return a new prim and create maya node matching given prim,
	including any children"""
	pathTokens.append(mfn.name())
	converterCls = toUsdMap.get(mfn.__class__, None)
	if converterCls is None:
		print("no USD converter for", mfn.__class__, "skipping")
		return None
	prim = converterCls.toUsdPrim(mfn, stage, pathTokens)
	if isinstance(mfn, om.MFnTransform):
		for childId in range(mfn.childCount()):
			_nodeToUsdPrim(core.getMFn(mfn.child(childId)), stage, list(pathTokens))

def _usdPrimToMFn(prim:Usd.Prim, stage:Usd.Stage, parentMfn=None)->om.MFn:
	"""return a new MFn and create maya node matching given prim,
	recreating any children"""
	mfn = toMFnMap[prim.GetTypeName()].getMFn(prim, stage, parentMfn)
	for child in prim.GetChildren():
		_usdPrimToMFn(child, stage, mfn)
	return mfn


def saveMayaNodeToUsd(mfn:om.MFnDependencyNode, stage:Usd.Stage=None, filePath:Path=None)->Usd.Stage:
	"""save a maya node to usd.
	 If stage is given, update it
	 If filePath is specified, save stage to that file,
	    otherwise return the stage
	"""

	mfn = core.getMFn(mfn)
	stage : Usd.Stage = stage or Usd.Stage.CreateInMemory()
	primPathTokens = []
	_nodeToUsdPrim(mfn, stage, primPathTokens)
	if filePath:
		#stage.Save()
		stage.Export(str(filePath))
	return stage

def loadFromUsd(stage:Usd.Stage=None, filePath:Path=None)->om.MFn:
	"""load a usd file into maya"""
	if not stage:
		if not filePath:
			raise ValueError("must specify either stage or filePath")
		filePath = Path(filePath)
		if not filePath.exists():
			raise ValueError("file does not exist:", filePath)
	stage = stage or Usd.Stage.Open(str(filePath))
	return _usdPrimToMFn(stage.GetPseudoRoot().GetChildren()[0], stage)



