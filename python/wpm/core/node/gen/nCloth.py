

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	NBase = Catalogue.NBase
else:
	from .. import retriever
	NBase = retriever.getNodeCls("NBase")
	assert NBase

# add node doc



# region plug type defs
class AddCrossLinksPlug(Plug):
	node : NCloth = None
	pass
class AirTightnessPlug(Plug):
	node : NCloth = None
	pass
class BendAngleDropoffPlug(Plug):
	node : NCloth = None
	pass
class BendAngleDropoffMapPlug(Plug):
	node : NCloth = None
	pass
class BendAngleDropoffMapTypePlug(Plug):
	node : NCloth = None
	pass
class BendAngleDropoffPerVertexPlug(Plug):
	node : NCloth = None
	pass
class BendAngleScalePlug(Plug):
	node : NCloth = None
	pass
class BendMapPlug(Plug):
	node : NCloth = None
	pass
class BendMapTypePlug(Plug):
	node : NCloth = None
	pass
class BendPerVertexPlug(Plug):
	node : NCloth = None
	pass
class BendResistancePlug(Plug):
	node : NCloth = None
	pass
class BendSolverPlug(Plug):
	node : NCloth = None
	pass
class CacheUsagePlug(Plug):
	node : NCloth = None
	pass
class CacheableAttributesPlug(Plug):
	node : NCloth = None
	pass
class CollideLastThresholdPlug(Plug):
	node : NCloth = None
	pass
class CompressionMapPlug(Plug):
	node : NCloth = None
	pass
class CompressionMapTypePlug(Plug):
	node : NCloth = None
	pass
class CompressionPerVertexPlug(Plug):
	node : NCloth = None
	pass
class CompressionResistancePlug(Plug):
	node : NCloth = None
	pass
class DeformMapPlug(Plug):
	node : NCloth = None
	pass
class DeformMapTypePlug(Plug):
	node : NCloth = None
	pass
class DeformPerVertexPlug(Plug):
	node : NCloth = None
	pass
class DeformResistancePlug(Plug):
	node : NCloth = None
	pass
class DragPlug(Plug):
	node : NCloth = None
	pass
class DragMapPlug(Plug):
	node : NCloth = None
	pass
class DragMapTypePlug(Plug):
	node : NCloth = None
	pass
class DragPerVertexPlug(Plug):
	node : NCloth = None
	pass
class EvaluationOrderPlug(Plug):
	node : NCloth = None
	pass
class IgnoreSolverGravityPlug(Plug):
	node : NCloth = None
	pass
class IgnoreSolverWindPlug(Plug):
	node : NCloth = None
	pass
class IncompressibilityPlug(Plug):
	node : NCloth = None
	pass
class InputAttractDampPlug(Plug):
	node : NCloth = None
	pass
class InputAttractMapPlug(Plug):
	node : NCloth = None
	pass
class InputAttractMapTypePlug(Plug):
	node : NCloth = None
	pass
class InputAttractMethodPlug(Plug):
	node : NCloth = None
	pass
class InputAttractPerVertexPlug(Plug):
	node : NCloth = None
	pass
class InputMeshAttractPlug(Plug):
	node : NCloth = None
	pass
class InputMotionDragPlug(Plug):
	node : NCloth = None
	pass
class LiftPlug(Plug):
	node : NCloth = None
	pass
class LiftMapPlug(Plug):
	node : NCloth = None
	pass
class LiftMapTypePlug(Plug):
	node : NCloth = None
	pass
class LiftPerVertexPlug(Plug):
	node : NCloth = None
	pass
class MinimalBendPlug(Plug):
	node : NCloth = None
	pass
class MinimalShearPlug(Plug):
	node : NCloth = None
	pass
class MinimalStretchPlug(Plug):
	node : NCloth = None
	pass
class NumSubdivisionsPlug(Plug):
	node : NCloth = None
	pass
class OutputMeshPlug(Plug):
	node : NCloth = None
	pass
class OutputStartMeshPlug(Plug):
	node : NCloth = None
	pass
class PressurePlug(Plug):
	node : NCloth = None
	pass
class PressureDampingPlug(Plug):
	node : NCloth = None
	pass
class PressureMethodPlug(Plug):
	node : NCloth = None
	pass
class PumpRatePlug(Plug):
	node : NCloth = None
	pass
class RestLengthScaleMapPlug(Plug):
	node : NCloth = None
	pass
class RestLengthScaleMapTypePlug(Plug):
	node : NCloth = None
	pass
class RestLengthScalePerVertexPlug(Plug):
	node : NCloth = None
	pass
class RestShapeMeshPlug(Plug):
	node : NCloth = None
	pass
class RestitutionAnglePlug(Plug):
	node : NCloth = None
	pass
class RestitutionAngleMapPlug(Plug):
	node : NCloth = None
	pass
class RestitutionAngleMapTypePlug(Plug):
	node : NCloth = None
	pass
class RestitutionAnglePerVertexPlug(Plug):
	node : NCloth = None
	pass
class RestitutionTensionPlug(Plug):
	node : NCloth = None
	pass
class RigidityPlug(Plug):
	node : NCloth = None
	pass
class RigidityMapPlug(Plug):
	node : NCloth = None
	pass
class RigidityMapTypePlug(Plug):
	node : NCloth = None
	pass
class RigidityPerVertexPlug(Plug):
	node : NCloth = None
	pass
class ScalingRelationPlug(Plug):
	node : NCloth = None
	pass
class SealHolesPlug(Plug):
	node : NCloth = None
	pass
class SelfCollideWidthScalePlug(Plug):
	node : NCloth = None
	pass
class SelfCollisionSoftnessPlug(Plug):
	node : NCloth = None
	pass
class SelfCrossoverPushPlug(Plug):
	node : NCloth = None
	pass
class SelfTrappedCheckPlug(Plug):
	node : NCloth = None
	pass
class ShearResistancePlug(Plug):
	node : NCloth = None
	pass
class SolverDisplayPlug(Plug):
	node : NCloth = None
	pass
class SortLinksPlug(Plug):
	node : NCloth = None
	pass
class StartPressurePlug(Plug):
	node : NCloth = None
	pass
class StretchDampPlug(Plug):
	node : NCloth = None
	pass
class StretchHierarchyLevelsPlug(Plug):
	node : NCloth = None
	pass
class StretchHierarchyPercentPlug(Plug):
	node : NCloth = None
	pass
class StretchMapPlug(Plug):
	node : NCloth = None
	pass
class StretchMapTypePlug(Plug):
	node : NCloth = None
	pass
class StretchPerVertexPlug(Plug):
	node : NCloth = None
	pass
class StretchResistancePlug(Plug):
	node : NCloth = None
	pass
class TangentialDragPlug(Plug):
	node : NCloth = None
	pass
class TangentialDragMapPlug(Plug):
	node : NCloth = None
	pass
class TangentialDragMapTypePlug(Plug):
	node : NCloth = None
	pass
class TangentialDragPerVertexPlug(Plug):
	node : NCloth = None
	pass
class UsePolygonShellsPlug(Plug):
	node : NCloth = None
	pass
class WindSelfShadowPlug(Plug):
	node : NCloth = None
	pass
class WrinkleMapPlug(Plug):
	node : NCloth = None
	pass
class WrinkleMapScalePlug(Plug):
	node : NCloth = None
	pass
class WrinkleMapTypePlug(Plug):
	node : NCloth = None
	pass
class WrinklePerVertexPlug(Plug):
	node : NCloth = None
	pass
# endregion


# define node class
class NCloth(NBase):
	addCrossLinks_ : AddCrossLinksPlug = PlugDescriptor("addCrossLinks")
	airTightness_ : AirTightnessPlug = PlugDescriptor("airTightness")
	bendAngleDropoff_ : BendAngleDropoffPlug = PlugDescriptor("bendAngleDropoff")
	bendAngleDropoffMap_ : BendAngleDropoffMapPlug = PlugDescriptor("bendAngleDropoffMap")
	bendAngleDropoffMapType_ : BendAngleDropoffMapTypePlug = PlugDescriptor("bendAngleDropoffMapType")
	bendAngleDropoffPerVertex_ : BendAngleDropoffPerVertexPlug = PlugDescriptor("bendAngleDropoffPerVertex")
	bendAngleScale_ : BendAngleScalePlug = PlugDescriptor("bendAngleScale")
	bendMap_ : BendMapPlug = PlugDescriptor("bendMap")
	bendMapType_ : BendMapTypePlug = PlugDescriptor("bendMapType")
	bendPerVertex_ : BendPerVertexPlug = PlugDescriptor("bendPerVertex")
	bendResistance_ : BendResistancePlug = PlugDescriptor("bendResistance")
	bendSolver_ : BendSolverPlug = PlugDescriptor("bendSolver")
	cacheUsage_ : CacheUsagePlug = PlugDescriptor("cacheUsage")
	cacheableAttributes_ : CacheableAttributesPlug = PlugDescriptor("cacheableAttributes")
	collideLastThreshold_ : CollideLastThresholdPlug = PlugDescriptor("collideLastThreshold")
	compressionMap_ : CompressionMapPlug = PlugDescriptor("compressionMap")
	compressionMapType_ : CompressionMapTypePlug = PlugDescriptor("compressionMapType")
	compressionPerVertex_ : CompressionPerVertexPlug = PlugDescriptor("compressionPerVertex")
	compressionResistance_ : CompressionResistancePlug = PlugDescriptor("compressionResistance")
	deformMap_ : DeformMapPlug = PlugDescriptor("deformMap")
	deformMapType_ : DeformMapTypePlug = PlugDescriptor("deformMapType")
	deformPerVertex_ : DeformPerVertexPlug = PlugDescriptor("deformPerVertex")
	deformResistance_ : DeformResistancePlug = PlugDescriptor("deformResistance")
	drag_ : DragPlug = PlugDescriptor("drag")
	dragMap_ : DragMapPlug = PlugDescriptor("dragMap")
	dragMapType_ : DragMapTypePlug = PlugDescriptor("dragMapType")
	dragPerVertex_ : DragPerVertexPlug = PlugDescriptor("dragPerVertex")
	evaluationOrder_ : EvaluationOrderPlug = PlugDescriptor("evaluationOrder")
	ignoreSolverGravity_ : IgnoreSolverGravityPlug = PlugDescriptor("ignoreSolverGravity")
	ignoreSolverWind_ : IgnoreSolverWindPlug = PlugDescriptor("ignoreSolverWind")
	incompressibility_ : IncompressibilityPlug = PlugDescriptor("incompressibility")
	inputAttractDamp_ : InputAttractDampPlug = PlugDescriptor("inputAttractDamp")
	inputAttractMap_ : InputAttractMapPlug = PlugDescriptor("inputAttractMap")
	inputAttractMapType_ : InputAttractMapTypePlug = PlugDescriptor("inputAttractMapType")
	inputAttractMethod_ : InputAttractMethodPlug = PlugDescriptor("inputAttractMethod")
	inputAttractPerVertex_ : InputAttractPerVertexPlug = PlugDescriptor("inputAttractPerVertex")
	inputMeshAttract_ : InputMeshAttractPlug = PlugDescriptor("inputMeshAttract")
	inputMotionDrag_ : InputMotionDragPlug = PlugDescriptor("inputMotionDrag")
	lift_ : LiftPlug = PlugDescriptor("lift")
	liftMap_ : LiftMapPlug = PlugDescriptor("liftMap")
	liftMapType_ : LiftMapTypePlug = PlugDescriptor("liftMapType")
	liftPerVertex_ : LiftPerVertexPlug = PlugDescriptor("liftPerVertex")
	minimalBend_ : MinimalBendPlug = PlugDescriptor("minimalBend")
	minimalShear_ : MinimalShearPlug = PlugDescriptor("minimalShear")
	minimalStretch_ : MinimalStretchPlug = PlugDescriptor("minimalStretch")
	numSubdivisions_ : NumSubdivisionsPlug = PlugDescriptor("numSubdivisions")
	outputMesh_ : OutputMeshPlug = PlugDescriptor("outputMesh")
	outputStartMesh_ : OutputStartMeshPlug = PlugDescriptor("outputStartMesh")
	pressure_ : PressurePlug = PlugDescriptor("pressure")
	pressureDamping_ : PressureDampingPlug = PlugDescriptor("pressureDamping")
	pressureMethod_ : PressureMethodPlug = PlugDescriptor("pressureMethod")
	pumpRate_ : PumpRatePlug = PlugDescriptor("pumpRate")
	restLengthScaleMap_ : RestLengthScaleMapPlug = PlugDescriptor("restLengthScaleMap")
	restLengthScaleMapType_ : RestLengthScaleMapTypePlug = PlugDescriptor("restLengthScaleMapType")
	restLengthScalePerVertex_ : RestLengthScalePerVertexPlug = PlugDescriptor("restLengthScalePerVertex")
	restShapeMesh_ : RestShapeMeshPlug = PlugDescriptor("restShapeMesh")
	restitutionAngle_ : RestitutionAnglePlug = PlugDescriptor("restitutionAngle")
	restitutionAngleMap_ : RestitutionAngleMapPlug = PlugDescriptor("restitutionAngleMap")
	restitutionAngleMapType_ : RestitutionAngleMapTypePlug = PlugDescriptor("restitutionAngleMapType")
	restitutionAnglePerVertex_ : RestitutionAnglePerVertexPlug = PlugDescriptor("restitutionAnglePerVertex")
	restitutionTension_ : RestitutionTensionPlug = PlugDescriptor("restitutionTension")
	rigidity_ : RigidityPlug = PlugDescriptor("rigidity")
	rigidityMap_ : RigidityMapPlug = PlugDescriptor("rigidityMap")
	rigidityMapType_ : RigidityMapTypePlug = PlugDescriptor("rigidityMapType")
	rigidityPerVertex_ : RigidityPerVertexPlug = PlugDescriptor("rigidityPerVertex")
	scalingRelation_ : ScalingRelationPlug = PlugDescriptor("scalingRelation")
	sealHoles_ : SealHolesPlug = PlugDescriptor("sealHoles")
	selfCollideWidthScale_ : SelfCollideWidthScalePlug = PlugDescriptor("selfCollideWidthScale")
	selfCollisionSoftness_ : SelfCollisionSoftnessPlug = PlugDescriptor("selfCollisionSoftness")
	selfCrossoverPush_ : SelfCrossoverPushPlug = PlugDescriptor("selfCrossoverPush")
	selfTrappedCheck_ : SelfTrappedCheckPlug = PlugDescriptor("selfTrappedCheck")
	shearResistance_ : ShearResistancePlug = PlugDescriptor("shearResistance")
	solverDisplay_ : SolverDisplayPlug = PlugDescriptor("solverDisplay")
	sortLinks_ : SortLinksPlug = PlugDescriptor("sortLinks")
	startPressure_ : StartPressurePlug = PlugDescriptor("startPressure")
	stretchDamp_ : StretchDampPlug = PlugDescriptor("stretchDamp")
	stretchHierarchyLevels_ : StretchHierarchyLevelsPlug = PlugDescriptor("stretchHierarchyLevels")
	stretchHierarchyPercent_ : StretchHierarchyPercentPlug = PlugDescriptor("stretchHierarchyPercent")
	stretchMap_ : StretchMapPlug = PlugDescriptor("stretchMap")
	stretchMapType_ : StretchMapTypePlug = PlugDescriptor("stretchMapType")
	stretchPerVertex_ : StretchPerVertexPlug = PlugDescriptor("stretchPerVertex")
	stretchResistance_ : StretchResistancePlug = PlugDescriptor("stretchResistance")
	tangentialDrag_ : TangentialDragPlug = PlugDescriptor("tangentialDrag")
	tangentialDragMap_ : TangentialDragMapPlug = PlugDescriptor("tangentialDragMap")
	tangentialDragMapType_ : TangentialDragMapTypePlug = PlugDescriptor("tangentialDragMapType")
	tangentialDragPerVertex_ : TangentialDragPerVertexPlug = PlugDescriptor("tangentialDragPerVertex")
	usePolygonShells_ : UsePolygonShellsPlug = PlugDescriptor("usePolygonShells")
	windSelfShadow_ : WindSelfShadowPlug = PlugDescriptor("windSelfShadow")
	wrinkleMap_ : WrinkleMapPlug = PlugDescriptor("wrinkleMap")
	wrinkleMapScale_ : WrinkleMapScalePlug = PlugDescriptor("wrinkleMapScale")
	wrinkleMapType_ : WrinkleMapTypePlug = PlugDescriptor("wrinkleMapType")
	wrinklePerVertex_ : WrinklePerVertexPlug = PlugDescriptor("wrinklePerVertex")

	# node attributes

	typeName = "nCloth"
	apiTypeInt = 1007
	apiTypeStr = "kNCloth"
	typeIdInt = 1313033295
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["addCrossLinks", "airTightness", "bendAngleDropoff", "bendAngleDropoffMap", "bendAngleDropoffMapType", "bendAngleDropoffPerVertex", "bendAngleScale", "bendMap", "bendMapType", "bendPerVertex", "bendResistance", "bendSolver", "cacheUsage", "cacheableAttributes", "collideLastThreshold", "compressionMap", "compressionMapType", "compressionPerVertex", "compressionResistance", "deformMap", "deformMapType", "deformPerVertex", "deformResistance", "drag", "dragMap", "dragMapType", "dragPerVertex", "evaluationOrder", "ignoreSolverGravity", "ignoreSolverWind", "incompressibility", "inputAttractDamp", "inputAttractMap", "inputAttractMapType", "inputAttractMethod", "inputAttractPerVertex", "inputMeshAttract", "inputMotionDrag", "lift", "liftMap", "liftMapType", "liftPerVertex", "minimalBend", "minimalShear", "minimalStretch", "numSubdivisions", "outputMesh", "outputStartMesh", "pressure", "pressureDamping", "pressureMethod", "pumpRate", "restLengthScaleMap", "restLengthScaleMapType", "restLengthScalePerVertex", "restShapeMesh", "restitutionAngle", "restitutionAngleMap", "restitutionAngleMapType", "restitutionAnglePerVertex", "restitutionTension", "rigidity", "rigidityMap", "rigidityMapType", "rigidityPerVertex", "scalingRelation", "sealHoles", "selfCollideWidthScale", "selfCollisionSoftness", "selfCrossoverPush", "selfTrappedCheck", "shearResistance", "solverDisplay", "sortLinks", "startPressure", "stretchDamp", "stretchHierarchyLevels", "stretchHierarchyPercent", "stretchMap", "stretchMapType", "stretchPerVertex", "stretchResistance", "tangentialDrag", "tangentialDragMap", "tangentialDragMapType", "tangentialDragPerVertex", "usePolygonShells", "windSelfShadow", "wrinkleMap", "wrinkleMapScale", "wrinkleMapType", "wrinklePerVertex"]
	nodeLeafPlugs = ["addCrossLinks", "airTightness", "bendAngleDropoff", "bendAngleDropoffMap", "bendAngleDropoffMapType", "bendAngleDropoffPerVertex", "bendAngleScale", "bendMap", "bendMapType", "bendPerVertex", "bendResistance", "bendSolver", "cacheUsage", "cacheableAttributes", "collideLastThreshold", "compressionMap", "compressionMapType", "compressionPerVertex", "compressionResistance", "deformMap", "deformMapType", "deformPerVertex", "deformResistance", "drag", "dragMap", "dragMapType", "dragPerVertex", "evaluationOrder", "ignoreSolverGravity", "ignoreSolverWind", "incompressibility", "inputAttractDamp", "inputAttractMap", "inputAttractMapType", "inputAttractMethod", "inputAttractPerVertex", "inputMeshAttract", "inputMotionDrag", "lift", "liftMap", "liftMapType", "liftPerVertex", "minimalBend", "minimalShear", "minimalStretch", "numSubdivisions", "outputMesh", "outputStartMesh", "pressure", "pressureDamping", "pressureMethod", "pumpRate", "restLengthScaleMap", "restLengthScaleMapType", "restLengthScalePerVertex", "restShapeMesh", "restitutionAngle", "restitutionAngleMap", "restitutionAngleMapType", "restitutionAnglePerVertex", "restitutionTension", "rigidity", "rigidityMap", "rigidityMapType", "rigidityPerVertex", "scalingRelation", "sealHoles", "selfCollideWidthScale", "selfCollisionSoftness", "selfCrossoverPush", "selfTrappedCheck", "shearResistance", "solverDisplay", "sortLinks", "startPressure", "stretchDamp", "stretchHierarchyLevels", "stretchHierarchyPercent", "stretchMap", "stretchMapType", "stretchPerVertex", "stretchResistance", "tangentialDrag", "tangentialDragMap", "tangentialDragMapType", "tangentialDragPerVertex", "usePolygonShells", "windSelfShadow", "wrinkleMap", "wrinkleMapScale", "wrinkleMapType", "wrinklePerVertex"]
	pass

