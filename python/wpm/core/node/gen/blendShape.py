

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
GeometryFilter = retriever.getNodeCls("GeometryFilter")
assert GeometryFilter
if T.TYPE_CHECKING:
	from .. import GeometryFilter

# add node doc



# region plug type defs
class BaseOriginXPlug(Plug):
	parent : BaseOriginPlug = PlugDescriptor("baseOrigin")
	node : BlendShape = None
	pass
class BaseOriginYPlug(Plug):
	parent : BaseOriginPlug = PlugDescriptor("baseOrigin")
	node : BlendShape = None
	pass
class BaseOriginZPlug(Plug):
	parent : BaseOriginPlug = PlugDescriptor("baseOrigin")
	node : BlendShape = None
	pass
class BaseOriginPlug(Plug):
	baseOriginX_ : BaseOriginXPlug = PlugDescriptor("baseOriginX")
	bx_ : BaseOriginXPlug = PlugDescriptor("baseOriginX")
	baseOriginY_ : BaseOriginYPlug = PlugDescriptor("baseOriginY")
	by_ : BaseOriginYPlug = PlugDescriptor("baseOriginY")
	baseOriginZ_ : BaseOriginZPlug = PlugDescriptor("baseOriginZ")
	bz_ : BaseOriginZPlug = PlugDescriptor("baseOriginZ")
	node : BlendShape = None
	pass
class DeformationOrderPlug(Plug):
	node : BlendShape = None
	pass
class IconPlug(Plug):
	node : BlendShape = None
	pass
class InbetweenTargetNamePlug(Plug):
	parent : InbetweenInfoPlug = PlugDescriptor("inbetweenInfo")
	node : BlendShape = None
	pass
class InbetweenTargetTypePlug(Plug):
	parent : InbetweenInfoPlug = PlugDescriptor("inbetweenInfo")
	node : BlendShape = None
	pass
class InbetweenVisibilityPlug(Plug):
	parent : InbetweenInfoPlug = PlugDescriptor("inbetweenInfo")
	node : BlendShape = None
	pass
class InterpolationPlug(Plug):
	parent : InbetweenInfoPlug = PlugDescriptor("inbetweenInfo")
	node : BlendShape = None
	pass
class InterpolationCurvePlug(Plug):
	parent : InbetweenInfoPlug = PlugDescriptor("inbetweenInfo")
	curvePosition_ : CurvePositionPlug = PlugDescriptor("curvePosition")
	cvp_ : CurvePositionPlug = PlugDescriptor("curvePosition")
	curveValue_ : CurveValuePlug = PlugDescriptor("curveValue")
	cvv_ : CurveValuePlug = PlugDescriptor("curveValue")
	node : BlendShape = None
	pass
class InbetweenInfoPlug(Plug):
	parent : InbetweenInfoGroupPlug = PlugDescriptor("inbetweenInfoGroup")
	inbetweenTargetName_ : InbetweenTargetNamePlug = PlugDescriptor("inbetweenTargetName")
	ibtn_ : InbetweenTargetNamePlug = PlugDescriptor("inbetweenTargetName")
	inbetweenTargetType_ : InbetweenTargetTypePlug = PlugDescriptor("inbetweenTargetType")
	ibtt_ : InbetweenTargetTypePlug = PlugDescriptor("inbetweenTargetType")
	inbetweenVisibility_ : InbetweenVisibilityPlug = PlugDescriptor("inbetweenVisibility")
	ibvs_ : InbetweenVisibilityPlug = PlugDescriptor("inbetweenVisibility")
	interpolation_ : InterpolationPlug = PlugDescriptor("interpolation")
	itp_ : InterpolationPlug = PlugDescriptor("interpolation")
	interpolationCurve_ : InterpolationCurvePlug = PlugDescriptor("interpolationCurve")
	itc_ : InterpolationCurvePlug = PlugDescriptor("interpolationCurve")
	node : BlendShape = None
	pass
class InbetweenInfoGroupPlug(Plug):
	inbetweenInfo_ : InbetweenInfoPlug = PlugDescriptor("inbetweenInfo")
	ibi_ : InbetweenInfoPlug = PlugDescriptor("inbetweenInfo")
	node : BlendShape = None
	pass
class BaseWeightsPlug(Plug):
	parent : InputTargetPlug = PlugDescriptor("inputTarget")
	node : BlendShape = None
	pass
class DeformMatrixPlug(Plug):
	parent : InputTargetPlug = PlugDescriptor("inputTarget")
	node : BlendShape = None
	pass
class DeformMatrixModifiedPlug(Plug):
	parent : InputTargetPlug = PlugDescriptor("inputTarget")
	node : BlendShape = None
	pass
class InputTargetItemPlug(Plug):
	parent : InputTargetGroupPlug = PlugDescriptor("inputTargetGroup")
	inputComponentsTarget_ : InputComponentsTargetPlug = PlugDescriptor("inputComponentsTarget")
	ict_ : InputComponentsTargetPlug = PlugDescriptor("inputComponentsTarget")
	inputGeomTarget_ : InputGeomTargetPlug = PlugDescriptor("inputGeomTarget")
	igt_ : InputGeomTargetPlug = PlugDescriptor("inputGeomTarget")
	inputPointsTarget_ : InputPointsTargetPlug = PlugDescriptor("inputPointsTarget")
	ipt_ : InputPointsTargetPlug = PlugDescriptor("inputPointsTarget")
	inputRelativeComponentsTarget_ : InputRelativeComponentsTargetPlug = PlugDescriptor("inputRelativeComponentsTarget")
	irc_ : InputRelativeComponentsTargetPlug = PlugDescriptor("inputRelativeComponentsTarget")
	inputRelativePointsTarget_ : InputRelativePointsTargetPlug = PlugDescriptor("inputRelativePointsTarget")
	irp_ : InputRelativePointsTargetPlug = PlugDescriptor("inputRelativePointsTarget")
	node : BlendShape = None
	pass
class NormalizationIdPlug(Plug):
	parent : InputTargetGroupPlug = PlugDescriptor("inputTargetGroup")
	node : BlendShape = None
	pass
class PostDeformersModePlug(Plug):
	parent : InputTargetGroupPlug = PlugDescriptor("inputTargetGroup")
	node : BlendShape = None
	pass
class TargetBindMatrixPlug(Plug):
	parent : InputTargetGroupPlug = PlugDescriptor("inputTargetGroup")
	node : BlendShape = None
	pass
class TargetMatrixPlug(Plug):
	parent : InputTargetGroupPlug = PlugDescriptor("inputTargetGroup")
	node : BlendShape = None
	pass
class TargetWeightsPlug(Plug):
	parent : InputTargetGroupPlug = PlugDescriptor("inputTargetGroup")
	node : BlendShape = None
	pass
class InputTargetGroupPlug(Plug):
	parent : InputTargetPlug = PlugDescriptor("inputTarget")
	inputTargetItem_ : InputTargetItemPlug = PlugDescriptor("inputTargetItem")
	iti_ : InputTargetItemPlug = PlugDescriptor("inputTargetItem")
	normalizationId_ : NormalizationIdPlug = PlugDescriptor("normalizationId")
	nid_ : NormalizationIdPlug = PlugDescriptor("normalizationId")
	postDeformersMode_ : PostDeformersModePlug = PlugDescriptor("postDeformersMode")
	pdm_ : PostDeformersModePlug = PlugDescriptor("postDeformersMode")
	targetBindMatrix_ : TargetBindMatrixPlug = PlugDescriptor("targetBindMatrix")
	bmx_ : TargetBindMatrixPlug = PlugDescriptor("targetBindMatrix")
	targetMatrix_ : TargetMatrixPlug = PlugDescriptor("targetMatrix")
	tmx_ : TargetMatrixPlug = PlugDescriptor("targetMatrix")
	targetWeights_ : TargetWeightsPlug = PlugDescriptor("targetWeights")
	tw_ : TargetWeightsPlug = PlugDescriptor("targetWeights")
	node : BlendShape = None
	pass
class NormalizationUseWeightsPlug(Plug):
	parent : NormalizationGroupPlug = PlugDescriptor("normalizationGroup")
	node : BlendShape = None
	pass
class NormalizationWeightsPlug(Plug):
	parent : NormalizationGroupPlug = PlugDescriptor("normalizationGroup")
	node : BlendShape = None
	pass
class NormalizationGroupPlug(Plug):
	parent : InputTargetPlug = PlugDescriptor("inputTarget")
	normalizationUseWeights_ : NormalizationUseWeightsPlug = PlugDescriptor("normalizationUseWeights")
	nuw_ : NormalizationUseWeightsPlug = PlugDescriptor("normalizationUseWeights")
	normalizationWeights_ : NormalizationWeightsPlug = PlugDescriptor("normalizationWeights")
	nw_ : NormalizationWeightsPlug = PlugDescriptor("normalizationWeights")
	node : BlendShape = None
	pass
class PaintTargetIndexPlug(Plug):
	parent : InputTargetPlug = PlugDescriptor("inputTarget")
	node : BlendShape = None
	pass
class PaintTargetWeightsPlug(Plug):
	parent : InputTargetPlug = PlugDescriptor("inputTarget")
	node : BlendShape = None
	pass
class SculptInbetweenWeightPlug(Plug):
	parent : InputTargetPlug = PlugDescriptor("inputTarget")
	node : BlendShape = None
	pass
class SculptTargetIndexPlug(Plug):
	parent : InputTargetPlug = PlugDescriptor("inputTarget")
	node : BlendShape = None
	pass
class XVertexPlug(Plug):
	parent : VertexPlug = PlugDescriptor("vertex")
	node : BlendShape = None
	pass
class YVertexPlug(Plug):
	parent : VertexPlug = PlugDescriptor("vertex")
	node : BlendShape = None
	pass
class ZVertexPlug(Plug):
	parent : VertexPlug = PlugDescriptor("vertex")
	node : BlendShape = None
	pass
class VertexPlug(Plug):
	parent : SculptTargetTweaksPlug = PlugDescriptor("sculptTargetTweaks")
	xVertex_ : XVertexPlug = PlugDescriptor("xVertex")
	vx_ : XVertexPlug = PlugDescriptor("xVertex")
	yVertex_ : YVertexPlug = PlugDescriptor("yVertex")
	vy_ : YVertexPlug = PlugDescriptor("yVertex")
	zVertex_ : ZVertexPlug = PlugDescriptor("zVertex")
	vz_ : ZVertexPlug = PlugDescriptor("zVertex")
	node : BlendShape = None
	pass
class SculptTargetTweaksPlug(Plug):
	parent : InputTargetPlug = PlugDescriptor("inputTarget")
	controlPoints_ : ControlPointsPlug = PlugDescriptor("controlPoints")
	cp_ : ControlPointsPlug = PlugDescriptor("controlPoints")
	vertex_ : VertexPlug = PlugDescriptor("vertex")
	vt_ : VertexPlug = PlugDescriptor("vertex")
	node : BlendShape = None
	pass
class InputTargetPlug(Plug):
	baseWeights_ : BaseWeightsPlug = PlugDescriptor("baseWeights")
	bw_ : BaseWeightsPlug = PlugDescriptor("baseWeights")
	deformMatrix_ : DeformMatrixPlug = PlugDescriptor("deformMatrix")
	dmx_ : DeformMatrixPlug = PlugDescriptor("deformMatrix")
	deformMatrixModified_ : DeformMatrixModifiedPlug = PlugDescriptor("deformMatrixModified")
	dmxm_ : DeformMatrixModifiedPlug = PlugDescriptor("deformMatrixModified")
	inputTargetGroup_ : InputTargetGroupPlug = PlugDescriptor("inputTargetGroup")
	itg_ : InputTargetGroupPlug = PlugDescriptor("inputTargetGroup")
	normalizationGroup_ : NormalizationGroupPlug = PlugDescriptor("normalizationGroup")
	ng_ : NormalizationGroupPlug = PlugDescriptor("normalizationGroup")
	paintTargetIndex_ : PaintTargetIndexPlug = PlugDescriptor("paintTargetIndex")
	pti_ : PaintTargetIndexPlug = PlugDescriptor("paintTargetIndex")
	paintTargetWeights_ : PaintTargetWeightsPlug = PlugDescriptor("paintTargetWeights")
	pwt_ : PaintTargetWeightsPlug = PlugDescriptor("paintTargetWeights")
	sculptInbetweenWeight_ : SculptInbetweenWeightPlug = PlugDescriptor("sculptInbetweenWeight")
	siw_ : SculptInbetweenWeightPlug = PlugDescriptor("sculptInbetweenWeight")
	sculptTargetIndex_ : SculptTargetIndexPlug = PlugDescriptor("sculptTargetIndex")
	sti_ : SculptTargetIndexPlug = PlugDescriptor("sculptTargetIndex")
	sculptTargetTweaks_ : SculptTargetTweaksPlug = PlugDescriptor("sculptTargetTweaks")
	stt_ : SculptTargetTweaksPlug = PlugDescriptor("sculptTargetTweaks")
	node : BlendShape = None
	pass
class InputComponentsTargetPlug(Plug):
	parent : InputTargetItemPlug = PlugDescriptor("inputTargetItem")
	node : BlendShape = None
	pass
class InputGeomTargetPlug(Plug):
	parent : InputTargetItemPlug = PlugDescriptor("inputTargetItem")
	node : BlendShape = None
	pass
class InputPointsTargetPlug(Plug):
	parent : InputTargetItemPlug = PlugDescriptor("inputTargetItem")
	node : BlendShape = None
	pass
class InputRelativeComponentsTargetPlug(Plug):
	parent : InputTargetItemPlug = PlugDescriptor("inputTargetItem")
	node : BlendShape = None
	pass
class InputRelativePointsTargetPlug(Plug):
	parent : InputTargetItemPlug = PlugDescriptor("inputTargetItem")
	node : BlendShape = None
	pass
class CurvePositionPlug(Plug):
	parent : InterpolationCurvePlug = PlugDescriptor("interpolationCurve")
	node : BlendShape = None
	pass
class CurveValuePlug(Plug):
	parent : InterpolationCurvePlug = PlugDescriptor("interpolationCurve")
	node : BlendShape = None
	pass
class LocalVertexFramePlug(Plug):
	node : BlendShape = None
	pass
class MidLayerIdPlug(Plug):
	node : BlendShape = None
	pass
class MidLayerParentPlug(Plug):
	node : BlendShape = None
	pass
class NextNodePlug(Plug):
	node : BlendShape = None
	pass
class NextTargetPlug(Plug):
	node : BlendShape = None
	pass
class OffsetXPlug(Plug):
	parent : OffsetDeformerPlug = PlugDescriptor("offsetDeformer")
	node : BlendShape = None
	pass
class OffsetYPlug(Plug):
	parent : OffsetDeformerPlug = PlugDescriptor("offsetDeformer")
	node : BlendShape = None
	pass
class OffsetZPlug(Plug):
	parent : OffsetDeformerPlug = PlugDescriptor("offsetDeformer")
	node : BlendShape = None
	pass
class OffsetDeformerPlug(Plug):
	offsetX_ : OffsetXPlug = PlugDescriptor("offsetX")
	ofx_ : OffsetXPlug = PlugDescriptor("offsetX")
	offsetY_ : OffsetYPlug = PlugDescriptor("offsetY")
	ofy_ : OffsetYPlug = PlugDescriptor("offsetY")
	offsetZ_ : OffsetZPlug = PlugDescriptor("offsetZ")
	ofz_ : OffsetZPlug = PlugDescriptor("offsetZ")
	node : BlendShape = None
	pass
class OriginPlug(Plug):
	node : BlendShape = None
	pass
class PaintWeightsPlug(Plug):
	node : BlendShape = None
	pass
class ParallelBlenderPlug(Plug):
	node : BlendShape = None
	pass
class ParentDirectoryPlug(Plug):
	node : BlendShape = None
	pass
class XValuePlug(Plug):
	parent : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : BlendShape = None
	pass
class YValuePlug(Plug):
	parent : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : BlendShape = None
	pass
class ZValuePlug(Plug):
	parent : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : BlendShape = None
	pass
class ControlPointsPlug(Plug):
	parent : SculptTargetTweaksPlug = PlugDescriptor("sculptTargetTweaks")
	xValue_ : XValuePlug = PlugDescriptor("xValue")
	xv_ : XValuePlug = PlugDescriptor("xValue")
	yValue_ : YValuePlug = PlugDescriptor("yValue")
	yv_ : YValuePlug = PlugDescriptor("yValue")
	zValue_ : ZValuePlug = PlugDescriptor("zValue")
	zv_ : ZValuePlug = PlugDescriptor("zValue")
	node : BlendShape = None
	pass
class SupportNegativeWeightsPlug(Plug):
	node : BlendShape = None
	pass
class SymmetryEdgePlug(Plug):
	node : BlendShape = None
	pass
class ChildIndicesPlug(Plug):
	parent : TargetDirectoryPlug = PlugDescriptor("targetDirectory")
	node : BlendShape = None
	pass
class DirectoryNamePlug(Plug):
	parent : TargetDirectoryPlug = PlugDescriptor("targetDirectory")
	node : BlendShape = None
	pass
class DirectoryParentVisibilityPlug(Plug):
	parent : TargetDirectoryPlug = PlugDescriptor("targetDirectory")
	node : BlendShape = None
	pass
class DirectoryVisibilityPlug(Plug):
	parent : TargetDirectoryPlug = PlugDescriptor("targetDirectory")
	node : BlendShape = None
	pass
class DirectoryWeightPlug(Plug):
	parent : TargetDirectoryPlug = PlugDescriptor("targetDirectory")
	node : BlendShape = None
	pass
class ParentIndexPlug(Plug):
	parent : TargetDirectoryPlug = PlugDescriptor("targetDirectory")
	node : BlendShape = None
	pass
class TargetDirectoryPlug(Plug):
	childIndices_ : ChildIndicesPlug = PlugDescriptor("childIndices")
	cid_ : ChildIndicesPlug = PlugDescriptor("childIndices")
	directoryName_ : DirectoryNamePlug = PlugDescriptor("directoryName")
	dtn_ : DirectoryNamePlug = PlugDescriptor("directoryName")
	directoryParentVisibility_ : DirectoryParentVisibilityPlug = PlugDescriptor("directoryParentVisibility")
	dpvs_ : DirectoryParentVisibilityPlug = PlugDescriptor("directoryParentVisibility")
	directoryVisibility_ : DirectoryVisibilityPlug = PlugDescriptor("directoryVisibility")
	dvs_ : DirectoryVisibilityPlug = PlugDescriptor("directoryVisibility")
	directoryWeight_ : DirectoryWeightPlug = PlugDescriptor("directoryWeight")
	dwgh_ : DirectoryWeightPlug = PlugDescriptor("directoryWeight")
	parentIndex_ : ParentIndexPlug = PlugDescriptor("parentIndex")
	pnid_ : ParentIndexPlug = PlugDescriptor("parentIndex")
	node : BlendShape = None
	pass
class TargetOriginXPlug(Plug):
	parent : TargetOriginPlug = PlugDescriptor("targetOrigin")
	node : BlendShape = None
	pass
class TargetOriginYPlug(Plug):
	parent : TargetOriginPlug = PlugDescriptor("targetOrigin")
	node : BlendShape = None
	pass
class TargetOriginZPlug(Plug):
	parent : TargetOriginPlug = PlugDescriptor("targetOrigin")
	node : BlendShape = None
	pass
class TargetOriginPlug(Plug):
	targetOriginX_ : TargetOriginXPlug = PlugDescriptor("targetOriginX")
	tx_ : TargetOriginXPlug = PlugDescriptor("targetOriginX")
	targetOriginY_ : TargetOriginYPlug = PlugDescriptor("targetOriginY")
	ty_ : TargetOriginYPlug = PlugDescriptor("targetOriginY")
	targetOriginZ_ : TargetOriginZPlug = PlugDescriptor("targetOriginZ")
	tz_ : TargetOriginZPlug = PlugDescriptor("targetOriginZ")
	node : BlendShape = None
	pass
class TargetParentVisibilityPlug(Plug):
	node : BlendShape = None
	pass
class TargetVisibilityPlug(Plug):
	node : BlendShape = None
	pass
class TopologyCheckPlug(Plug):
	node : BlendShape = None
	pass
class UseTargetCompWeightsPlug(Plug):
	node : BlendShape = None
	pass
class WeightPlug(Plug):
	node : BlendShape = None
	pass
# endregion


# define node class
class BlendShape(GeometryFilter):
	baseOriginX_ : BaseOriginXPlug = PlugDescriptor("baseOriginX")
	baseOriginY_ : BaseOriginYPlug = PlugDescriptor("baseOriginY")
	baseOriginZ_ : BaseOriginZPlug = PlugDescriptor("baseOriginZ")
	baseOrigin_ : BaseOriginPlug = PlugDescriptor("baseOrigin")
	deformationOrder_ : DeformationOrderPlug = PlugDescriptor("deformationOrder")
	icon_ : IconPlug = PlugDescriptor("icon")
	inbetweenTargetName_ : InbetweenTargetNamePlug = PlugDescriptor("inbetweenTargetName")
	inbetweenTargetType_ : InbetweenTargetTypePlug = PlugDescriptor("inbetweenTargetType")
	inbetweenVisibility_ : InbetweenVisibilityPlug = PlugDescriptor("inbetweenVisibility")
	interpolation_ : InterpolationPlug = PlugDescriptor("interpolation")
	interpolationCurve_ : InterpolationCurvePlug = PlugDescriptor("interpolationCurve")
	inbetweenInfo_ : InbetweenInfoPlug = PlugDescriptor("inbetweenInfo")
	inbetweenInfoGroup_ : InbetweenInfoGroupPlug = PlugDescriptor("inbetweenInfoGroup")
	baseWeights_ : BaseWeightsPlug = PlugDescriptor("baseWeights")
	deformMatrix_ : DeformMatrixPlug = PlugDescriptor("deformMatrix")
	deformMatrixModified_ : DeformMatrixModifiedPlug = PlugDescriptor("deformMatrixModified")
	inputTargetItem_ : InputTargetItemPlug = PlugDescriptor("inputTargetItem")
	normalizationId_ : NormalizationIdPlug = PlugDescriptor("normalizationId")
	postDeformersMode_ : PostDeformersModePlug = PlugDescriptor("postDeformersMode")
	targetBindMatrix_ : TargetBindMatrixPlug = PlugDescriptor("targetBindMatrix")
	targetMatrix_ : TargetMatrixPlug = PlugDescriptor("targetMatrix")
	targetWeights_ : TargetWeightsPlug = PlugDescriptor("targetWeights")
	inputTargetGroup_ : InputTargetGroupPlug = PlugDescriptor("inputTargetGroup")
	normalizationUseWeights_ : NormalizationUseWeightsPlug = PlugDescriptor("normalizationUseWeights")
	normalizationWeights_ : NormalizationWeightsPlug = PlugDescriptor("normalizationWeights")
	normalizationGroup_ : NormalizationGroupPlug = PlugDescriptor("normalizationGroup")
	paintTargetIndex_ : PaintTargetIndexPlug = PlugDescriptor("paintTargetIndex")
	paintTargetWeights_ : PaintTargetWeightsPlug = PlugDescriptor("paintTargetWeights")
	sculptInbetweenWeight_ : SculptInbetweenWeightPlug = PlugDescriptor("sculptInbetweenWeight")
	sculptTargetIndex_ : SculptTargetIndexPlug = PlugDescriptor("sculptTargetIndex")
	xVertex_ : XVertexPlug = PlugDescriptor("xVertex")
	yVertex_ : YVertexPlug = PlugDescriptor("yVertex")
	zVertex_ : ZVertexPlug = PlugDescriptor("zVertex")
	vertex_ : VertexPlug = PlugDescriptor("vertex")
	sculptTargetTweaks_ : SculptTargetTweaksPlug = PlugDescriptor("sculptTargetTweaks")
	inputTarget_ : InputTargetPlug = PlugDescriptor("inputTarget")
	inputComponentsTarget_ : InputComponentsTargetPlug = PlugDescriptor("inputComponentsTarget")
	inputGeomTarget_ : InputGeomTargetPlug = PlugDescriptor("inputGeomTarget")
	inputPointsTarget_ : InputPointsTargetPlug = PlugDescriptor("inputPointsTarget")
	inputRelativeComponentsTarget_ : InputRelativeComponentsTargetPlug = PlugDescriptor("inputRelativeComponentsTarget")
	inputRelativePointsTarget_ : InputRelativePointsTargetPlug = PlugDescriptor("inputRelativePointsTarget")
	curvePosition_ : CurvePositionPlug = PlugDescriptor("curvePosition")
	curveValue_ : CurveValuePlug = PlugDescriptor("curveValue")
	localVertexFrame_ : LocalVertexFramePlug = PlugDescriptor("localVertexFrame")
	midLayerId_ : MidLayerIdPlug = PlugDescriptor("midLayerId")
	midLayerParent_ : MidLayerParentPlug = PlugDescriptor("midLayerParent")
	nextNode_ : NextNodePlug = PlugDescriptor("nextNode")
	nextTarget_ : NextTargetPlug = PlugDescriptor("nextTarget")
	offsetX_ : OffsetXPlug = PlugDescriptor("offsetX")
	offsetY_ : OffsetYPlug = PlugDescriptor("offsetY")
	offsetZ_ : OffsetZPlug = PlugDescriptor("offsetZ")
	offsetDeformer_ : OffsetDeformerPlug = PlugDescriptor("offsetDeformer")
	origin_ : OriginPlug = PlugDescriptor("origin")
	paintWeights_ : PaintWeightsPlug = PlugDescriptor("paintWeights")
	parallelBlender_ : ParallelBlenderPlug = PlugDescriptor("parallelBlender")
	parentDirectory_ : ParentDirectoryPlug = PlugDescriptor("parentDirectory")
	xValue_ : XValuePlug = PlugDescriptor("xValue")
	yValue_ : YValuePlug = PlugDescriptor("yValue")
	zValue_ : ZValuePlug = PlugDescriptor("zValue")
	controlPoints_ : ControlPointsPlug = PlugDescriptor("controlPoints")
	supportNegativeWeights_ : SupportNegativeWeightsPlug = PlugDescriptor("supportNegativeWeights")
	symmetryEdge_ : SymmetryEdgePlug = PlugDescriptor("symmetryEdge")
	childIndices_ : ChildIndicesPlug = PlugDescriptor("childIndices")
	directoryName_ : DirectoryNamePlug = PlugDescriptor("directoryName")
	directoryParentVisibility_ : DirectoryParentVisibilityPlug = PlugDescriptor("directoryParentVisibility")
	directoryVisibility_ : DirectoryVisibilityPlug = PlugDescriptor("directoryVisibility")
	directoryWeight_ : DirectoryWeightPlug = PlugDescriptor("directoryWeight")
	parentIndex_ : ParentIndexPlug = PlugDescriptor("parentIndex")
	targetDirectory_ : TargetDirectoryPlug = PlugDescriptor("targetDirectory")
	targetOriginX_ : TargetOriginXPlug = PlugDescriptor("targetOriginX")
	targetOriginY_ : TargetOriginYPlug = PlugDescriptor("targetOriginY")
	targetOriginZ_ : TargetOriginZPlug = PlugDescriptor("targetOriginZ")
	targetOrigin_ : TargetOriginPlug = PlugDescriptor("targetOrigin")
	targetParentVisibility_ : TargetParentVisibilityPlug = PlugDescriptor("targetParentVisibility")
	targetVisibility_ : TargetVisibilityPlug = PlugDescriptor("targetVisibility")
	topologyCheck_ : TopologyCheckPlug = PlugDescriptor("topologyCheck")
	useTargetCompWeights_ : UseTargetCompWeightsPlug = PlugDescriptor("useTargetCompWeights")
	weight_ : WeightPlug = PlugDescriptor("weight")

	# node attributes

	typeName = "blendShape"
	apiTypeInt = 336
	apiTypeStr = "kBlendShape"
	typeIdInt = 1178750035
	MFnCls = om.MFnGeometryFilter
	nodeLeafClassAttrs = ["baseOriginX", "baseOriginY", "baseOriginZ", "baseOrigin", "deformationOrder", "icon", "inbetweenTargetName", "inbetweenTargetType", "inbetweenVisibility", "interpolation", "interpolationCurve", "inbetweenInfo", "inbetweenInfoGroup", "baseWeights", "deformMatrix", "deformMatrixModified", "inputTargetItem", "normalizationId", "postDeformersMode", "targetBindMatrix", "targetMatrix", "targetWeights", "inputTargetGroup", "normalizationUseWeights", "normalizationWeights", "normalizationGroup", "paintTargetIndex", "paintTargetWeights", "sculptInbetweenWeight", "sculptTargetIndex", "xVertex", "yVertex", "zVertex", "vertex", "sculptTargetTweaks", "inputTarget", "inputComponentsTarget", "inputGeomTarget", "inputPointsTarget", "inputRelativeComponentsTarget", "inputRelativePointsTarget", "curvePosition", "curveValue", "localVertexFrame", "midLayerId", "midLayerParent", "nextNode", "nextTarget", "offsetX", "offsetY", "offsetZ", "offsetDeformer", "origin", "paintWeights", "parallelBlender", "parentDirectory", "xValue", "yValue", "zValue", "controlPoints", "supportNegativeWeights", "symmetryEdge", "childIndices", "directoryName", "directoryParentVisibility", "directoryVisibility", "directoryWeight", "parentIndex", "targetDirectory", "targetOriginX", "targetOriginY", "targetOriginZ", "targetOrigin", "targetParentVisibility", "targetVisibility", "topologyCheck", "useTargetCompWeights", "weight"]
	nodeLeafPlugs = ["baseOrigin", "deformationOrder", "icon", "inbetweenInfoGroup", "inputTarget", "localVertexFrame", "midLayerId", "midLayerParent", "nextNode", "nextTarget", "offsetDeformer", "origin", "paintWeights", "parallelBlender", "parentDirectory", "supportNegativeWeights", "symmetryEdge", "targetDirectory", "targetOrigin", "targetParentVisibility", "targetVisibility", "topologyCheck", "useTargetCompWeights", "weight"]
	pass

