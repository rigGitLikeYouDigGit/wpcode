

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Transform = Catalogue.Transform
else:
	from .. import retriever
	Transform = retriever.getNodeCls("Transform")
	assert Transform

# add node doc



# region plug type defs
class AlphaPlug(Plug):
	node : HikEffector = None
	pass
class AuxEffectorPlug(Plug):
	node : HikEffector = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : HikEffector = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : HikEffector = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : HikEffector = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	clb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	clg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	clr_ : ColorRPlug = PlugDescriptor("colorR")
	node : HikEffector = None
	pass
class EffectorIDPlug(Plug):
	node : HikEffector = None
	pass
class FkjointPlug(Plug):
	node : HikEffector = None
	pass
class HandlePlug(Plug):
	node : HikEffector = None
	pass
class JointPlug(Plug):
	node : HikEffector = None
	pass
class MarkerLookPlug(Plug):
	node : HikEffector = None
	pass
class PinningPlug(Plug):
	node : HikEffector = None
	pass
class PivotOffsetXPlug(Plug):
	parent : PivotOffsetPlug = PlugDescriptor("pivotOffset")
	node : HikEffector = None
	pass
class PivotOffsetYPlug(Plug):
	parent : PivotOffsetPlug = PlugDescriptor("pivotOffset")
	node : HikEffector = None
	pass
class PivotOffsetZPlug(Plug):
	parent : PivotOffsetPlug = PlugDescriptor("pivotOffset")
	node : HikEffector = None
	pass
class PivotOffsetPlug(Plug):
	pivotOffsetX_ : PivotOffsetXPlug = PlugDescriptor("pivotOffsetX")
	px_ : PivotOffsetXPlug = PlugDescriptor("pivotOffsetX")
	pivotOffsetY_ : PivotOffsetYPlug = PlugDescriptor("pivotOffsetY")
	py_ : PivotOffsetYPlug = PlugDescriptor("pivotOffsetY")
	pivotOffsetZ_ : PivotOffsetZPlug = PlugDescriptor("pivotOffsetZ")
	pz_ : PivotOffsetZPlug = PlugDescriptor("pivotOffsetZ")
	node : HikEffector = None
	pass
class PivotsPlug(Plug):
	node : HikEffector = None
	pass
class PreRotationXPlug(Plug):
	parent : PreRotationPlug = PlugDescriptor("preRotation")
	node : HikEffector = None
	pass
class PreRotationYPlug(Plug):
	parent : PreRotationPlug = PlugDescriptor("preRotation")
	node : HikEffector = None
	pass
class PreRotationZPlug(Plug):
	parent : PreRotationPlug = PlugDescriptor("preRotation")
	node : HikEffector = None
	pass
class PreRotationPlug(Plug):
	preRotationX_ : PreRotationXPlug = PlugDescriptor("preRotationX")
	prx_ : PreRotationXPlug = PlugDescriptor("preRotationX")
	preRotationY_ : PreRotationYPlug = PlugDescriptor("preRotationY")
	pry_ : PreRotationYPlug = PlugDescriptor("preRotationY")
	preRotationZ_ : PreRotationZPlug = PlugDescriptor("preRotationZ")
	prz_ : PreRotationZPlug = PlugDescriptor("preRotationZ")
	node : HikEffector = None
	pass
class RadiusPlug(Plug):
	node : HikEffector = None
	pass
class ReachRotationPlug(Plug):
	node : HikEffector = None
	pass
class ReachTranslationPlug(Plug):
	node : HikEffector = None
	pass
# endregion


# define node class
class HikEffector(Transform):
	alpha_ : AlphaPlug = PlugDescriptor("alpha")
	auxEffector_ : AuxEffectorPlug = PlugDescriptor("auxEffector")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	effectorID_ : EffectorIDPlug = PlugDescriptor("effectorID")
	fkjoint_ : FkjointPlug = PlugDescriptor("fkjoint")
	handle_ : HandlePlug = PlugDescriptor("handle")
	joint_ : JointPlug = PlugDescriptor("joint")
	markerLook_ : MarkerLookPlug = PlugDescriptor("markerLook")
	pinning_ : PinningPlug = PlugDescriptor("pinning")
	pivotOffsetX_ : PivotOffsetXPlug = PlugDescriptor("pivotOffsetX")
	pivotOffsetY_ : PivotOffsetYPlug = PlugDescriptor("pivotOffsetY")
	pivotOffsetZ_ : PivotOffsetZPlug = PlugDescriptor("pivotOffsetZ")
	pivotOffset_ : PivotOffsetPlug = PlugDescriptor("pivotOffset")
	pivots_ : PivotsPlug = PlugDescriptor("pivots")
	preRotationX_ : PreRotationXPlug = PlugDescriptor("preRotationX")
	preRotationY_ : PreRotationYPlug = PlugDescriptor("preRotationY")
	preRotationZ_ : PreRotationZPlug = PlugDescriptor("preRotationZ")
	preRotation_ : PreRotationPlug = PlugDescriptor("preRotation")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	reachRotation_ : ReachRotationPlug = PlugDescriptor("reachRotation")
	reachTranslation_ : ReachTranslationPlug = PlugDescriptor("reachTranslation")

	# node attributes

	typeName = "hikEffector"
	apiTypeInt = 961
	apiTypeStr = "kHikEffector"
	typeIdInt = 1145456971
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["alpha", "auxEffector", "colorB", "colorG", "colorR", "color", "effectorID", "fkjoint", "handle", "joint", "markerLook", "pinning", "pivotOffsetX", "pivotOffsetY", "pivotOffsetZ", "pivotOffset", "pivots", "preRotationX", "preRotationY", "preRotationZ", "preRotation", "radius", "reachRotation", "reachTranslation"]
	nodeLeafPlugs = ["alpha", "auxEffector", "color", "effectorID", "fkjoint", "handle", "joint", "markerLook", "pinning", "pivotOffset", "pivots", "preRotation", "radius", "reachRotation", "reachTranslation"]
	pass

