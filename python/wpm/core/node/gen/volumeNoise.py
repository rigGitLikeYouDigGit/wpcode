

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Texture3d = retriever.getNodeCls("Texture3d")
assert Texture3d
if T.TYPE_CHECKING:
	from .. import Texture3d

# add node doc



# region plug type defs
class AmplitudePlug(Plug):
	node : VolumeNoise = None
	pass
class DensityPlug(Plug):
	node : VolumeNoise = None
	pass
class DepthMaxPlug(Plug):
	node : VolumeNoise = None
	pass
class FalloffPlug(Plug):
	node : VolumeNoise = None
	pass
class FrequencyPlug(Plug):
	node : VolumeNoise = None
	pass
class FrequencyRatioPlug(Plug):
	node : VolumeNoise = None
	pass
class ImplodePlug(Plug):
	node : VolumeNoise = None
	pass
class ImplodeCenterXPlug(Plug):
	parent : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	node : VolumeNoise = None
	pass
class ImplodeCenterYPlug(Plug):
	parent : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	node : VolumeNoise = None
	pass
class ImplodeCenterZPlug(Plug):
	parent : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	node : VolumeNoise = None
	pass
class ImplodeCenterPlug(Plug):
	implodeCenterX_ : ImplodeCenterXPlug = PlugDescriptor("implodeCenterX")
	imx_ : ImplodeCenterXPlug = PlugDescriptor("implodeCenterX")
	implodeCenterY_ : ImplodeCenterYPlug = PlugDescriptor("implodeCenterY")
	imy_ : ImplodeCenterYPlug = PlugDescriptor("implodeCenterY")
	implodeCenterZ_ : ImplodeCenterZPlug = PlugDescriptor("implodeCenterZ")
	imz_ : ImplodeCenterZPlug = PlugDescriptor("implodeCenterZ")
	node : VolumeNoise = None
	pass
class InflectionPlug(Plug):
	node : VolumeNoise = None
	pass
class NoiseTypePlug(Plug):
	node : VolumeNoise = None
	pass
class NumWavesPlug(Plug):
	node : VolumeNoise = None
	pass
class OriginXPlug(Plug):
	parent : OriginPlug = PlugDescriptor("origin")
	node : VolumeNoise = None
	pass
class OriginYPlug(Plug):
	parent : OriginPlug = PlugDescriptor("origin")
	node : VolumeNoise = None
	pass
class OriginZPlug(Plug):
	parent : OriginPlug = PlugDescriptor("origin")
	node : VolumeNoise = None
	pass
class OriginPlug(Plug):
	originX_ : OriginXPlug = PlugDescriptor("originX")
	orx_ : OriginXPlug = PlugDescriptor("originX")
	originY_ : OriginYPlug = PlugDescriptor("originY")
	ory_ : OriginYPlug = PlugDescriptor("originY")
	originZ_ : OriginZPlug = PlugDescriptor("originZ")
	orz_ : OriginZPlug = PlugDescriptor("originZ")
	node : VolumeNoise = None
	pass
class RandomnessPlug(Plug):
	node : VolumeNoise = None
	pass
class RatioPlug(Plug):
	node : VolumeNoise = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : VolumeNoise = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : VolumeNoise = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : VolumeNoise = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : VolumeNoise = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : VolumeNoise = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : VolumeNoise = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : VolumeNoise = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	roy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	roz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : VolumeNoise = None
	pass
class ScaleXPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : VolumeNoise = None
	pass
class ScaleYPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : VolumeNoise = None
	pass
class ScaleZPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : VolumeNoise = None
	pass
class ScalePlug(Plug):
	scaleX_ : ScaleXPlug = PlugDescriptor("scaleX")
	sx_ : ScaleXPlug = PlugDescriptor("scaleX")
	scaleY_ : ScaleYPlug = PlugDescriptor("scaleY")
	sy_ : ScaleYPlug = PlugDescriptor("scaleY")
	scaleZ_ : ScaleZPlug = PlugDescriptor("scaleZ")
	sz_ : ScaleZPlug = PlugDescriptor("scaleZ")
	node : VolumeNoise = None
	pass
class SizeRandPlug(Plug):
	node : VolumeNoise = None
	pass
class SpottynessPlug(Plug):
	node : VolumeNoise = None
	pass
class ThresholdPlug(Plug):
	node : VolumeNoise = None
	pass
class TimePlug(Plug):
	node : VolumeNoise = None
	pass
class XPixelAnglePlug(Plug):
	node : VolumeNoise = None
	pass
# endregion


# define node class
class VolumeNoise(Texture3d):
	amplitude_ : AmplitudePlug = PlugDescriptor("amplitude")
	density_ : DensityPlug = PlugDescriptor("density")
	depthMax_ : DepthMaxPlug = PlugDescriptor("depthMax")
	falloff_ : FalloffPlug = PlugDescriptor("falloff")
	frequency_ : FrequencyPlug = PlugDescriptor("frequency")
	frequencyRatio_ : FrequencyRatioPlug = PlugDescriptor("frequencyRatio")
	implode_ : ImplodePlug = PlugDescriptor("implode")
	implodeCenterX_ : ImplodeCenterXPlug = PlugDescriptor("implodeCenterX")
	implodeCenterY_ : ImplodeCenterYPlug = PlugDescriptor("implodeCenterY")
	implodeCenterZ_ : ImplodeCenterZPlug = PlugDescriptor("implodeCenterZ")
	implodeCenter_ : ImplodeCenterPlug = PlugDescriptor("implodeCenter")
	inflection_ : InflectionPlug = PlugDescriptor("inflection")
	noiseType_ : NoiseTypePlug = PlugDescriptor("noiseType")
	numWaves_ : NumWavesPlug = PlugDescriptor("numWaves")
	originX_ : OriginXPlug = PlugDescriptor("originX")
	originY_ : OriginYPlug = PlugDescriptor("originY")
	originZ_ : OriginZPlug = PlugDescriptor("originZ")
	origin_ : OriginPlug = PlugDescriptor("origin")
	randomness_ : RandomnessPlug = PlugDescriptor("randomness")
	ratio_ : RatioPlug = PlugDescriptor("ratio")
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	refPointCamera_ : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	refPointObj_ : RefPointObjPlug = PlugDescriptor("refPointObj")
	scaleX_ : ScaleXPlug = PlugDescriptor("scaleX")
	scaleY_ : ScaleYPlug = PlugDescriptor("scaleY")
	scaleZ_ : ScaleZPlug = PlugDescriptor("scaleZ")
	scale_ : ScalePlug = PlugDescriptor("scale")
	sizeRand_ : SizeRandPlug = PlugDescriptor("sizeRand")
	spottyness_ : SpottynessPlug = PlugDescriptor("spottyness")
	threshold_ : ThresholdPlug = PlugDescriptor("threshold")
	time_ : TimePlug = PlugDescriptor("time")
	xPixelAngle_ : XPixelAnglePlug = PlugDescriptor("xPixelAngle")

	# node attributes

	typeName = "volumeNoise"
	apiTypeInt = 876
	apiTypeStr = "kVolumeNoise"
	typeIdInt = 1381258803
	MFnCls = om.MFnDependencyNode
	pass

