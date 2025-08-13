

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PfxGeometry = retriever.getNodeCls("PfxGeometry")
assert PfxGeometry
if T.TYPE_CHECKING:
	from .. import PfxGeometry

# add node doc



# region plug type defs
class CollisionObjectPlug(Plug):
	node : Stroke = None
	pass
class MaxClipPlug(Plug):
	node : Stroke = None
	pass
class MinClipPlug(Plug):
	node : Stroke = None
	pass
class MinimalTwistPlug(Plug):
	node : Stroke = None
	pass
class NormalXPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : Stroke = None
	pass
class NormalYPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : Stroke = None
	pass
class NormalZPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : Stroke = None
	pass
class NormalPlug(Plug):
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	nmx_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	nmy_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	nmz_ : NormalZPlug = PlugDescriptor("normalZ")
	node : Stroke = None
	pass
class OutNormalXPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Stroke = None
	pass
class OutNormalYPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Stroke = None
	pass
class OutNormalZPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Stroke = None
	pass
class OutNormalPlug(Plug):
	outNormalX_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	onx_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	outNormalY_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	ony_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	outNormalZ_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	onz_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	node : Stroke = None
	pass
class OutPointXPlug(Plug):
	parent : OutPointPlug = PlugDescriptor("outPoint")
	node : Stroke = None
	pass
class OutPointYPlug(Plug):
	parent : OutPointPlug = PlugDescriptor("outPoint")
	node : Stroke = None
	pass
class OutPointZPlug(Plug):
	parent : OutPointPlug = PlugDescriptor("outPoint")
	node : Stroke = None
	pass
class OutPointPlug(Plug):
	outPointX_ : OutPointXPlug = PlugDescriptor("outPointX")
	ox_ : OutPointXPlug = PlugDescriptor("outPointX")
	outPointY_ : OutPointYPlug = PlugDescriptor("outPointY")
	oy_ : OutPointYPlug = PlugDescriptor("outPointY")
	outPointZ_ : OutPointZPlug = PlugDescriptor("outPointZ")
	oz_ : OutPointZPlug = PlugDescriptor("outPointZ")
	node : Stroke = None
	pass
class CurvePlug(Plug):
	parent : PathCurvePlug = PlugDescriptor("pathCurve")
	node : Stroke = None
	pass
class OppositePlug(Plug):
	parent : PathCurvePlug = PlugDescriptor("pathCurve")
	node : Stroke = None
	pass
class SamplesPlug(Plug):
	parent : PathCurvePlug = PlugDescriptor("pathCurve")
	node : Stroke = None
	pass
class PathCurvePlug(Plug):
	curve_ : CurvePlug = PlugDescriptor("curve")
	crv_ : CurvePlug = PlugDescriptor("curve")
	opposite_ : OppositePlug = PlugDescriptor("opposite")
	opp_ : OppositePlug = PlugDescriptor("opposite")
	samples_ : SamplesPlug = PlugDescriptor("samples")
	smp_ : SamplesPlug = PlugDescriptor("samples")
	node : Stroke = None
	pass
class PerspectivePlug(Plug):
	node : Stroke = None
	pass
class PressurePlug(Plug):
	node : Stroke = None
	pass
class PressureMap1Plug(Plug):
	node : Stroke = None
	pass
class PressureMap2Plug(Plug):
	node : Stroke = None
	pass
class PressureMap3Plug(Plug):
	node : Stroke = None
	pass
class PressureMax1Plug(Plug):
	node : Stroke = None
	pass
class PressureMax2Plug(Plug):
	node : Stroke = None
	pass
class PressureMax3Plug(Plug):
	node : Stroke = None
	pass
class PressureMin1Plug(Plug):
	node : Stroke = None
	pass
class PressureMin2Plug(Plug):
	node : Stroke = None
	pass
class PressureMin3Plug(Plug):
	node : Stroke = None
	pass
class PressureScale_FloatValuePlug(Plug):
	parent : PressureScalePlug = PlugDescriptor("pressureScale")
	node : Stroke = None
	pass
class PressureScale_InterpPlug(Plug):
	parent : PressureScalePlug = PlugDescriptor("pressureScale")
	node : Stroke = None
	pass
class PressureScale_PositionPlug(Plug):
	parent : PressureScalePlug = PlugDescriptor("pressureScale")
	node : Stroke = None
	pass
class PressureScalePlug(Plug):
	pressureScale_FloatValue_ : PressureScale_FloatValuePlug = PlugDescriptor("pressureScale_FloatValue")
	pscfv_ : PressureScale_FloatValuePlug = PlugDescriptor("pressureScale_FloatValue")
	pressureScale_Interp_ : PressureScale_InterpPlug = PlugDescriptor("pressureScale_Interp")
	psci_ : PressureScale_InterpPlug = PlugDescriptor("pressureScale_Interp")
	pressureScale_Position_ : PressureScale_PositionPlug = PlugDescriptor("pressureScale_Position")
	pscp_ : PressureScale_PositionPlug = PlugDescriptor("pressureScale_Position")
	node : Stroke = None
	pass
class SampleDensityPlug(Plug):
	node : Stroke = None
	pass
class SmoothingPlug(Plug):
	node : Stroke = None
	pass
class UseNormalPlug(Plug):
	node : Stroke = None
	pass
class UvSetNamePlug(Plug):
	node : Stroke = None
	pass
# endregion


# define node class
class Stroke(PfxGeometry):
	collisionObject_ : CollisionObjectPlug = PlugDescriptor("collisionObject")
	maxClip_ : MaxClipPlug = PlugDescriptor("maxClip")
	minClip_ : MinClipPlug = PlugDescriptor("minClip")
	minimalTwist_ : MinimalTwistPlug = PlugDescriptor("minimalTwist")
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	normal_ : NormalPlug = PlugDescriptor("normal")
	outNormalX_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	outNormalY_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	outNormalZ_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	outNormal_ : OutNormalPlug = PlugDescriptor("outNormal")
	outPointX_ : OutPointXPlug = PlugDescriptor("outPointX")
	outPointY_ : OutPointYPlug = PlugDescriptor("outPointY")
	outPointZ_ : OutPointZPlug = PlugDescriptor("outPointZ")
	outPoint_ : OutPointPlug = PlugDescriptor("outPoint")
	curve_ : CurvePlug = PlugDescriptor("curve")
	opposite_ : OppositePlug = PlugDescriptor("opposite")
	samples_ : SamplesPlug = PlugDescriptor("samples")
	pathCurve_ : PathCurvePlug = PlugDescriptor("pathCurve")
	perspective_ : PerspectivePlug = PlugDescriptor("perspective")
	pressure_ : PressurePlug = PlugDescriptor("pressure")
	pressureMap1_ : PressureMap1Plug = PlugDescriptor("pressureMap1")
	pressureMap2_ : PressureMap2Plug = PlugDescriptor("pressureMap2")
	pressureMap3_ : PressureMap3Plug = PlugDescriptor("pressureMap3")
	pressureMax1_ : PressureMax1Plug = PlugDescriptor("pressureMax1")
	pressureMax2_ : PressureMax2Plug = PlugDescriptor("pressureMax2")
	pressureMax3_ : PressureMax3Plug = PlugDescriptor("pressureMax3")
	pressureMin1_ : PressureMin1Plug = PlugDescriptor("pressureMin1")
	pressureMin2_ : PressureMin2Plug = PlugDescriptor("pressureMin2")
	pressureMin3_ : PressureMin3Plug = PlugDescriptor("pressureMin3")
	pressureScale_FloatValue_ : PressureScale_FloatValuePlug = PlugDescriptor("pressureScale_FloatValue")
	pressureScale_Interp_ : PressureScale_InterpPlug = PlugDescriptor("pressureScale_Interp")
	pressureScale_Position_ : PressureScale_PositionPlug = PlugDescriptor("pressureScale_Position")
	pressureScale_ : PressureScalePlug = PlugDescriptor("pressureScale")
	sampleDensity_ : SampleDensityPlug = PlugDescriptor("sampleDensity")
	smoothing_ : SmoothingPlug = PlugDescriptor("smoothing")
	useNormal_ : UseNormalPlug = PlugDescriptor("useNormal")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")

	# node attributes

	typeName = "stroke"
	apiTypeInt = 764
	apiTypeStr = "kStroke"
	typeIdInt = 1398035019
	MFnCls = om.MFnDagNode
	pass

