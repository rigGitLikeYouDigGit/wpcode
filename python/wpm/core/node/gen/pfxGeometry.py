

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Shape = retriever.getNodeCls("Shape")
assert Shape
if T.TYPE_CHECKING:
	from .. import Shape

# add node doc



# region plug type defs
class BrushPlug(Plug):
	node : PfxGeometry = None
	pass
class CameraPointXPlug(Plug):
	parent : CameraPointPlug = PlugDescriptor("cameraPoint")
	node : PfxGeometry = None
	pass
class CameraPointYPlug(Plug):
	parent : CameraPointPlug = PlugDescriptor("cameraPoint")
	node : PfxGeometry = None
	pass
class CameraPointZPlug(Plug):
	parent : CameraPointPlug = PlugDescriptor("cameraPoint")
	node : PfxGeometry = None
	pass
class CameraPointPlug(Plug):
	cameraPointX_ : CameraPointXPlug = PlugDescriptor("cameraPointX")
	cpx_ : CameraPointXPlug = PlugDescriptor("cameraPointX")
	cameraPointY_ : CameraPointYPlug = PlugDescriptor("cameraPointY")
	cpy_ : CameraPointYPlug = PlugDescriptor("cameraPointY")
	cameraPointZ_ : CameraPointZPlug = PlugDescriptor("cameraPointZ")
	cpz_ : CameraPointZPlug = PlugDescriptor("cameraPointZ")
	node : PfxGeometry = None
	pass
class ControlCurvePlug(Plug):
	node : PfxGeometry = None
	pass
class CurveAlignPlug(Plug):
	node : PfxGeometry = None
	pass
class CurveModePlug(Plug):
	node : PfxGeometry = None
	pass
class DegreePlug(Plug):
	node : PfxGeometry = None
	pass
class DisplayPercentPlug(Plug):
	node : PfxGeometry = None
	pass
class DrawAsMeshPlug(Plug):
	node : PfxGeometry = None
	pass
class DrawOrderPlug(Plug):
	node : PfxGeometry = None
	pass
class FlowerCurveModePlug(Plug):
	node : PfxGeometry = None
	pass
class FlowerVertBufSizePlug(Plug):
	node : PfxGeometry = None
	pass
class LeafCurveModePlug(Plug):
	node : PfxGeometry = None
	pass
class LeafVertBufSizePlug(Plug):
	node : PfxGeometry = None
	pass
class LineModifierPlug(Plug):
	node : PfxGeometry = None
	pass
class MainVertBufSizePlug(Plug):
	node : PfxGeometry = None
	pass
class MaxDrawSegmentsPlug(Plug):
	node : PfxGeometry = None
	pass
class MeshHardEdgesPlug(Plug):
	node : PfxGeometry = None
	pass
class MeshPolyLimitPlug(Plug):
	node : PfxGeometry = None
	pass
class MeshQuadOutputPlug(Plug):
	node : PfxGeometry = None
	pass
class MeshVertexColorModePlug(Plug):
	node : PfxGeometry = None
	pass
class MotionBlurredPlug(Plug):
	node : PfxGeometry = None
	pass
class OutFlowerCurveCountPlug(Plug):
	node : PfxGeometry = None
	pass
class OutFlowerCurvesPlug(Plug):
	node : PfxGeometry = None
	pass
class OutFlowerMeshPlug(Plug):
	node : PfxGeometry = None
	pass
class OutLeafCurveCountPlug(Plug):
	node : PfxGeometry = None
	pass
class OutLeafCurvesPlug(Plug):
	node : PfxGeometry = None
	pass
class OutLeafMeshPlug(Plug):
	node : PfxGeometry = None
	pass
class OutMainCurveCountPlug(Plug):
	node : PfxGeometry = None
	pass
class OutMainCurvesPlug(Plug):
	node : PfxGeometry = None
	pass
class OutMainMeshPlug(Plug):
	node : PfxGeometry = None
	pass
class PrimaryVisibilityPlug(Plug):
	node : PfxGeometry = None
	pass
class SeedPlug(Plug):
	node : PfxGeometry = None
	pass
class SurfaceOffsetPlug(Plug):
	node : PfxGeometry = None
	pass
class WorldFlowerMeshPlug(Plug):
	node : PfxGeometry = None
	pass
class WorldLeafMeshPlug(Plug):
	node : PfxGeometry = None
	pass
class WorldMainMeshPlug(Plug):
	node : PfxGeometry = None
	pass
# endregion


# define node class
class PfxGeometry(Shape):
	brush_ : BrushPlug = PlugDescriptor("brush")
	cameraPointX_ : CameraPointXPlug = PlugDescriptor("cameraPointX")
	cameraPointY_ : CameraPointYPlug = PlugDescriptor("cameraPointY")
	cameraPointZ_ : CameraPointZPlug = PlugDescriptor("cameraPointZ")
	cameraPoint_ : CameraPointPlug = PlugDescriptor("cameraPoint")
	controlCurve_ : ControlCurvePlug = PlugDescriptor("controlCurve")
	curveAlign_ : CurveAlignPlug = PlugDescriptor("curveAlign")
	curveMode_ : CurveModePlug = PlugDescriptor("curveMode")
	degree_ : DegreePlug = PlugDescriptor("degree")
	displayPercent_ : DisplayPercentPlug = PlugDescriptor("displayPercent")
	drawAsMesh_ : DrawAsMeshPlug = PlugDescriptor("drawAsMesh")
	drawOrder_ : DrawOrderPlug = PlugDescriptor("drawOrder")
	flowerCurveMode_ : FlowerCurveModePlug = PlugDescriptor("flowerCurveMode")
	flowerVertBufSize_ : FlowerVertBufSizePlug = PlugDescriptor("flowerVertBufSize")
	leafCurveMode_ : LeafCurveModePlug = PlugDescriptor("leafCurveMode")
	leafVertBufSize_ : LeafVertBufSizePlug = PlugDescriptor("leafVertBufSize")
	lineModifier_ : LineModifierPlug = PlugDescriptor("lineModifier")
	mainVertBufSize_ : MainVertBufSizePlug = PlugDescriptor("mainVertBufSize")
	maxDrawSegments_ : MaxDrawSegmentsPlug = PlugDescriptor("maxDrawSegments")
	meshHardEdges_ : MeshHardEdgesPlug = PlugDescriptor("meshHardEdges")
	meshPolyLimit_ : MeshPolyLimitPlug = PlugDescriptor("meshPolyLimit")
	meshQuadOutput_ : MeshQuadOutputPlug = PlugDescriptor("meshQuadOutput")
	meshVertexColorMode_ : MeshVertexColorModePlug = PlugDescriptor("meshVertexColorMode")
	motionBlurred_ : MotionBlurredPlug = PlugDescriptor("motionBlurred")
	outFlowerCurveCount_ : OutFlowerCurveCountPlug = PlugDescriptor("outFlowerCurveCount")
	outFlowerCurves_ : OutFlowerCurvesPlug = PlugDescriptor("outFlowerCurves")
	outFlowerMesh_ : OutFlowerMeshPlug = PlugDescriptor("outFlowerMesh")
	outLeafCurveCount_ : OutLeafCurveCountPlug = PlugDescriptor("outLeafCurveCount")
	outLeafCurves_ : OutLeafCurvesPlug = PlugDescriptor("outLeafCurves")
	outLeafMesh_ : OutLeafMeshPlug = PlugDescriptor("outLeafMesh")
	outMainCurveCount_ : OutMainCurveCountPlug = PlugDescriptor("outMainCurveCount")
	outMainCurves_ : OutMainCurvesPlug = PlugDescriptor("outMainCurves")
	outMainMesh_ : OutMainMeshPlug = PlugDescriptor("outMainMesh")
	primaryVisibility_ : PrimaryVisibilityPlug = PlugDescriptor("primaryVisibility")
	seed_ : SeedPlug = PlugDescriptor("seed")
	surfaceOffset_ : SurfaceOffsetPlug = PlugDescriptor("surfaceOffset")
	worldFlowerMesh_ : WorldFlowerMeshPlug = PlugDescriptor("worldFlowerMesh")
	worldLeafMesh_ : WorldLeafMeshPlug = PlugDescriptor("worldLeafMesh")
	worldMainMesh_ : WorldMainMeshPlug = PlugDescriptor("worldMainMesh")

	# node attributes

	typeName = "pfxGeometry"
	apiTypeInt = 945
	apiTypeStr = "kPfxGeometry"
	typeIdInt = 1346783045
	MFnCls = om.MFnDagNode
	pass

