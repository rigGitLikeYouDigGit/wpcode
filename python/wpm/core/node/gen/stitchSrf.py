

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AbstractBaseCreate = Catalogue.AbstractBaseCreate
else:
	from .. import retriever
	AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
	assert AbstractBaseCreate

# add node doc



# region plug type defs
class BiasPlug(Plug):
	node : StitchSrf = None
	pass
class CvIthIndexPlug(Plug):
	node : StitchSrf = None
	pass
class CvJthIndexPlug(Plug):
	node : StitchSrf = None
	pass
class CvpositionXPlug(Plug):
	parent : CvPositionPlug = PlugDescriptor("cvPosition")
	node : StitchSrf = None
	pass
class CvpositionYPlug(Plug):
	parent : CvPositionPlug = PlugDescriptor("cvPosition")
	node : StitchSrf = None
	pass
class CvpositionZPlug(Plug):
	parent : CvPositionPlug = PlugDescriptor("cvPosition")
	node : StitchSrf = None
	pass
class CvPositionPlug(Plug):
	cvpositionX_ : CvpositionXPlug = PlugDescriptor("cvpositionX")
	cvx_ : CvpositionXPlug = PlugDescriptor("cvpositionX")
	cvpositionY_ : CvpositionYPlug = PlugDescriptor("cvpositionY")
	cvy_ : CvpositionYPlug = PlugDescriptor("cvpositionY")
	cvpositionZ_ : CvpositionZPlug = PlugDescriptor("cvpositionZ")
	cvz_ : CvpositionZPlug = PlugDescriptor("cvpositionZ")
	node : StitchSrf = None
	pass
class FixBoundaryPlug(Plug):
	node : StitchSrf = None
	pass
class InputCurvePlug(Plug):
	node : StitchSrf = None
	pass
class InputMatchCurvePlug(Plug):
	node : StitchSrf = None
	pass
class InputReferenceCOSPlug(Plug):
	node : StitchSrf = None
	pass
class InputSurfacePlug(Plug):
	node : StitchSrf = None
	pass
class NormalXPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : StitchSrf = None
	pass
class NormalYPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : StitchSrf = None
	pass
class NormalZPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : StitchSrf = None
	pass
class NormalPlug(Plug):
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	nx_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	ny_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	nz_ : NormalZPlug = PlugDescriptor("normalZ")
	node : StitchSrf = None
	pass
class OutputSurfacePlug(Plug):
	node : StitchSrf = None
	pass
class ParameterUPlug(Plug):
	node : StitchSrf = None
	pass
class ParameterVPlug(Plug):
	node : StitchSrf = None
	pass
class PositionXPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : StitchSrf = None
	pass
class PositionYPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : StitchSrf = None
	pass
class PositionZPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : StitchSrf = None
	pass
class PositionPlug(Plug):
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	px_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	py_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	pz_ : PositionZPlug = PlugDescriptor("positionZ")
	node : StitchSrf = None
	pass
class PositionalContinuityPlug(Plug):
	node : StitchSrf = None
	pass
class ShouldBeLastPlug(Plug):
	node : StitchSrf = None
	pass
class StepCountPlug(Plug):
	node : StitchSrf = None
	pass
class TangentialContinuityPlug(Plug):
	node : StitchSrf = None
	pass
class TogglePointNormalsPlug(Plug):
	node : StitchSrf = None
	pass
class TogglePointPositionPlug(Plug):
	node : StitchSrf = None
	pass
class ToggleTolerancePlug(Plug):
	node : StitchSrf = None
	pass
class TolerancePlug(Plug):
	node : StitchSrf = None
	pass
# endregion


# define node class
class StitchSrf(AbstractBaseCreate):
	bias_ : BiasPlug = PlugDescriptor("bias")
	cvIthIndex_ : CvIthIndexPlug = PlugDescriptor("cvIthIndex")
	cvJthIndex_ : CvJthIndexPlug = PlugDescriptor("cvJthIndex")
	cvpositionX_ : CvpositionXPlug = PlugDescriptor("cvpositionX")
	cvpositionY_ : CvpositionYPlug = PlugDescriptor("cvpositionY")
	cvpositionZ_ : CvpositionZPlug = PlugDescriptor("cvpositionZ")
	cvPosition_ : CvPositionPlug = PlugDescriptor("cvPosition")
	fixBoundary_ : FixBoundaryPlug = PlugDescriptor("fixBoundary")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	inputMatchCurve_ : InputMatchCurvePlug = PlugDescriptor("inputMatchCurve")
	inputReferenceCOS_ : InputReferenceCOSPlug = PlugDescriptor("inputReferenceCOS")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	normal_ : NormalPlug = PlugDescriptor("normal")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	parameterU_ : ParameterUPlug = PlugDescriptor("parameterU")
	parameterV_ : ParameterVPlug = PlugDescriptor("parameterV")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	position_ : PositionPlug = PlugDescriptor("position")
	positionalContinuity_ : PositionalContinuityPlug = PlugDescriptor("positionalContinuity")
	shouldBeLast_ : ShouldBeLastPlug = PlugDescriptor("shouldBeLast")
	stepCount_ : StepCountPlug = PlugDescriptor("stepCount")
	tangentialContinuity_ : TangentialContinuityPlug = PlugDescriptor("tangentialContinuity")
	togglePointNormals_ : TogglePointNormalsPlug = PlugDescriptor("togglePointNormals")
	togglePointPosition_ : TogglePointPositionPlug = PlugDescriptor("togglePointPosition")
	toggleTolerance_ : ToggleTolerancePlug = PlugDescriptor("toggleTolerance")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")

	# node attributes

	typeName = "stitchSrf"
	apiTypeInt = 101
	apiTypeStr = "kStitchSrf"
	typeIdInt = 1314083923
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["bias", "cvIthIndex", "cvJthIndex", "cvpositionX", "cvpositionY", "cvpositionZ", "cvPosition", "fixBoundary", "inputCurve", "inputMatchCurve", "inputReferenceCOS", "inputSurface", "normalX", "normalY", "normalZ", "normal", "outputSurface", "parameterU", "parameterV", "positionX", "positionY", "positionZ", "position", "positionalContinuity", "shouldBeLast", "stepCount", "tangentialContinuity", "togglePointNormals", "togglePointPosition", "toggleTolerance", "tolerance"]
	nodeLeafPlugs = ["bias", "cvIthIndex", "cvJthIndex", "cvPosition", "fixBoundary", "inputCurve", "inputMatchCurve", "inputReferenceCOS", "inputSurface", "normal", "outputSurface", "parameterU", "parameterV", "position", "positionalContinuity", "shouldBeLast", "stepCount", "tangentialContinuity", "togglePointNormals", "togglePointPosition", "toggleTolerance", "tolerance"]
	pass

