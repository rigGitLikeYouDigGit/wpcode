

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
	node : HikIKEffector = None
	pass
class AltConstraintTargetGXPlug(Plug):
	node : HikIKEffector = None
	pass
class AlternateGXPlug(Plug):
	node : HikIKEffector = None
	pass
class AuxEffectorPlug(Plug):
	node : HikIKEffector = None
	pass
class AuxiliariesPlug(Plug):
	node : HikIKEffector = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : HikIKEffector = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : HikIKEffector = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : HikIKEffector = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	clb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	clg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	clr_ : ColorRPlug = PlugDescriptor("colorR")
	node : HikIKEffector = None
	pass
class EffectorIDPlug(Plug):
	node : HikIKEffector = None
	pass
class JointOrientXPlug(Plug):
	parent : JointOrientPlug = PlugDescriptor("jointOrient")
	node : HikIKEffector = None
	pass
class JointOrientYPlug(Plug):
	parent : JointOrientPlug = PlugDescriptor("jointOrient")
	node : HikIKEffector = None
	pass
class JointOrientZPlug(Plug):
	parent : JointOrientPlug = PlugDescriptor("jointOrient")
	node : HikIKEffector = None
	pass
class JointOrientPlug(Plug):
	jointOrientX_ : JointOrientXPlug = PlugDescriptor("jointOrientX")
	jox_ : JointOrientXPlug = PlugDescriptor("jointOrientX")
	jointOrientY_ : JointOrientYPlug = PlugDescriptor("jointOrientY")
	joy_ : JointOrientYPlug = PlugDescriptor("jointOrientY")
	jointOrientZ_ : JointOrientZPlug = PlugDescriptor("jointOrientZ")
	joz_ : JointOrientZPlug = PlugDescriptor("jointOrientZ")
	node : HikIKEffector = None
	pass
class LookPlug(Plug):
	node : HikIKEffector = None
	pass
class MarkerLookPlug(Plug):
	node : HikIKEffector = None
	pass
class PinRPlug(Plug):
	node : HikIKEffector = None
	pass
class PinTPlug(Plug):
	node : HikIKEffector = None
	pass
class PinningPlug(Plug):
	node : HikIKEffector = None
	pass
class PivotOffsetXPlug(Plug):
	parent : PivotOffsetPlug = PlugDescriptor("pivotOffset")
	node : HikIKEffector = None
	pass
class PivotOffsetYPlug(Plug):
	parent : PivotOffsetPlug = PlugDescriptor("pivotOffset")
	node : HikIKEffector = None
	pass
class PivotOffsetZPlug(Plug):
	parent : PivotOffsetPlug = PlugDescriptor("pivotOffset")
	node : HikIKEffector = None
	pass
class PivotOffsetPlug(Plug):
	pivotOffsetX_ : PivotOffsetXPlug = PlugDescriptor("pivotOffsetX")
	px_ : PivotOffsetXPlug = PlugDescriptor("pivotOffsetX")
	pivotOffsetY_ : PivotOffsetYPlug = PlugDescriptor("pivotOffsetY")
	py_ : PivotOffsetYPlug = PlugDescriptor("pivotOffsetY")
	pivotOffsetZ_ : PivotOffsetZPlug = PlugDescriptor("pivotOffsetZ")
	pz_ : PivotOffsetZPlug = PlugDescriptor("pivotOffsetZ")
	node : HikIKEffector = None
	pass
class RadiusPlug(Plug):
	node : HikIKEffector = None
	pass
class ReachRotationPlug(Plug):
	node : HikIKEffector = None
	pass
class ReachTranslationPlug(Plug):
	node : HikIKEffector = None
	pass
class RotateOffsetXPlug(Plug):
	parent : RotateOffsetPlug = PlugDescriptor("rotateOffset")
	node : HikIKEffector = None
	pass
class RotateOffsetYPlug(Plug):
	parent : RotateOffsetPlug = PlugDescriptor("rotateOffset")
	node : HikIKEffector = None
	pass
class RotateOffsetZPlug(Plug):
	parent : RotateOffsetPlug = PlugDescriptor("rotateOffset")
	node : HikIKEffector = None
	pass
class RotateOffsetPlug(Plug):
	rotateOffsetX_ : RotateOffsetXPlug = PlugDescriptor("rotateOffsetX")
	rox_ : RotateOffsetXPlug = PlugDescriptor("rotateOffsetX")
	rotateOffsetY_ : RotateOffsetYPlug = PlugDescriptor("rotateOffsetY")
	roy_ : RotateOffsetYPlug = PlugDescriptor("rotateOffsetY")
	rotateOffsetZ_ : RotateOffsetZPlug = PlugDescriptor("rotateOffsetZ")
	roz_ : RotateOffsetZPlug = PlugDescriptor("rotateOffsetZ")
	node : HikIKEffector = None
	pass
class ScaleOffsetXPlug(Plug):
	parent : ScaleOffsetPlug = PlugDescriptor("scaleOffset")
	node : HikIKEffector = None
	pass
class ScaleOffsetYPlug(Plug):
	parent : ScaleOffsetPlug = PlugDescriptor("scaleOffset")
	node : HikIKEffector = None
	pass
class ScaleOffsetZPlug(Plug):
	parent : ScaleOffsetPlug = PlugDescriptor("scaleOffset")
	node : HikIKEffector = None
	pass
class ScaleOffsetPlug(Plug):
	scaleOffsetX_ : ScaleOffsetXPlug = PlugDescriptor("scaleOffsetX")
	sox_ : ScaleOffsetXPlug = PlugDescriptor("scaleOffsetX")
	scaleOffsetY_ : ScaleOffsetYPlug = PlugDescriptor("scaleOffsetY")
	soy_ : ScaleOffsetYPlug = PlugDescriptor("scaleOffsetY")
	scaleOffsetZ_ : ScaleOffsetZPlug = PlugDescriptor("scaleOffsetZ")
	soz_ : ScaleOffsetZPlug = PlugDescriptor("scaleOffsetZ")
	node : HikIKEffector = None
	pass
class TranslateOffsetXPlug(Plug):
	parent : TranslateOffsetPlug = PlugDescriptor("translateOffset")
	node : HikIKEffector = None
	pass
class TranslateOffsetYPlug(Plug):
	parent : TranslateOffsetPlug = PlugDescriptor("translateOffset")
	node : HikIKEffector = None
	pass
class TranslateOffsetZPlug(Plug):
	parent : TranslateOffsetPlug = PlugDescriptor("translateOffset")
	node : HikIKEffector = None
	pass
class TranslateOffsetPlug(Plug):
	translateOffsetX_ : TranslateOffsetXPlug = PlugDescriptor("translateOffsetX")
	tox_ : TranslateOffsetXPlug = PlugDescriptor("translateOffsetX")
	translateOffsetY_ : TranslateOffsetYPlug = PlugDescriptor("translateOffsetY")
	toy_ : TranslateOffsetYPlug = PlugDescriptor("translateOffsetY")
	translateOffsetZ_ : TranslateOffsetZPlug = PlugDescriptor("translateOffsetZ")
	toz_ : TranslateOffsetZPlug = PlugDescriptor("translateOffsetZ")
	node : HikIKEffector = None
	pass
class UseAlternateGXPlug(Plug):
	node : HikIKEffector = None
	pass
# endregion


# define node class
class HikIKEffector(Transform):
	alpha_ : AlphaPlug = PlugDescriptor("alpha")
	altConstraintTargetGX_ : AltConstraintTargetGXPlug = PlugDescriptor("altConstraintTargetGX")
	alternateGX_ : AlternateGXPlug = PlugDescriptor("alternateGX")
	auxEffector_ : AuxEffectorPlug = PlugDescriptor("auxEffector")
	auxiliaries_ : AuxiliariesPlug = PlugDescriptor("auxiliaries")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	effectorID_ : EffectorIDPlug = PlugDescriptor("effectorID")
	jointOrientX_ : JointOrientXPlug = PlugDescriptor("jointOrientX")
	jointOrientY_ : JointOrientYPlug = PlugDescriptor("jointOrientY")
	jointOrientZ_ : JointOrientZPlug = PlugDescriptor("jointOrientZ")
	jointOrient_ : JointOrientPlug = PlugDescriptor("jointOrient")
	look_ : LookPlug = PlugDescriptor("look")
	markerLook_ : MarkerLookPlug = PlugDescriptor("markerLook")
	pinR_ : PinRPlug = PlugDescriptor("pinR")
	pinT_ : PinTPlug = PlugDescriptor("pinT")
	pinning_ : PinningPlug = PlugDescriptor("pinning")
	pivotOffsetX_ : PivotOffsetXPlug = PlugDescriptor("pivotOffsetX")
	pivotOffsetY_ : PivotOffsetYPlug = PlugDescriptor("pivotOffsetY")
	pivotOffsetZ_ : PivotOffsetZPlug = PlugDescriptor("pivotOffsetZ")
	pivotOffset_ : PivotOffsetPlug = PlugDescriptor("pivotOffset")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	reachRotation_ : ReachRotationPlug = PlugDescriptor("reachRotation")
	reachTranslation_ : ReachTranslationPlug = PlugDescriptor("reachTranslation")
	rotateOffsetX_ : RotateOffsetXPlug = PlugDescriptor("rotateOffsetX")
	rotateOffsetY_ : RotateOffsetYPlug = PlugDescriptor("rotateOffsetY")
	rotateOffsetZ_ : RotateOffsetZPlug = PlugDescriptor("rotateOffsetZ")
	rotateOffset_ : RotateOffsetPlug = PlugDescriptor("rotateOffset")
	scaleOffsetX_ : ScaleOffsetXPlug = PlugDescriptor("scaleOffsetX")
	scaleOffsetY_ : ScaleOffsetYPlug = PlugDescriptor("scaleOffsetY")
	scaleOffsetZ_ : ScaleOffsetZPlug = PlugDescriptor("scaleOffsetZ")
	scaleOffset_ : ScaleOffsetPlug = PlugDescriptor("scaleOffset")
	translateOffsetX_ : TranslateOffsetXPlug = PlugDescriptor("translateOffsetX")
	translateOffsetY_ : TranslateOffsetYPlug = PlugDescriptor("translateOffsetY")
	translateOffsetZ_ : TranslateOffsetZPlug = PlugDescriptor("translateOffsetZ")
	translateOffset_ : TranslateOffsetPlug = PlugDescriptor("translateOffset")
	useAlternateGX_ : UseAlternateGXPlug = PlugDescriptor("useAlternateGX")

	# node attributes

	typeName = "hikIKEffector"
	apiTypeInt = 962
	apiTypeStr = "kHikIKEffector"
	typeIdInt = 1229669702
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["alpha", "altConstraintTargetGX", "alternateGX", "auxEffector", "auxiliaries", "colorB", "colorG", "colorR", "color", "effectorID", "jointOrientX", "jointOrientY", "jointOrientZ", "jointOrient", "look", "markerLook", "pinR", "pinT", "pinning", "pivotOffsetX", "pivotOffsetY", "pivotOffsetZ", "pivotOffset", "radius", "reachRotation", "reachTranslation", "rotateOffsetX", "rotateOffsetY", "rotateOffsetZ", "rotateOffset", "scaleOffsetX", "scaleOffsetY", "scaleOffsetZ", "scaleOffset", "translateOffsetX", "translateOffsetY", "translateOffsetZ", "translateOffset", "useAlternateGX"]
	nodeLeafPlugs = ["alpha", "altConstraintTargetGX", "alternateGX", "auxEffector", "auxiliaries", "color", "effectorID", "jointOrient", "look", "markerLook", "pinR", "pinT", "pinning", "pivotOffset", "radius", "reachRotation", "reachTranslation", "rotateOffset", "scaleOffset", "translateOffset", "useAlternateGX"]
	pass

