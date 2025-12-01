

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
TextureEnv = retriever.getNodeCls("TextureEnv")
assert TextureEnv
if T.TYPE_CHECKING:
	from .. import TextureEnv

# add node doc



# region plug type defs
class AirDensityPlug(Plug):
	node : EnvSky = None
	pass
class AltitudePlug(Plug):
	node : EnvSky = None
	pass
class AzimuthPlug(Plug):
	node : EnvSky = None
	pass
class BlurPlug(Plug):
	node : EnvSky = None
	pass
class CloudBrightnessBPlug(Plug):
	parent : CloudBrightnessPlug = PlugDescriptor("cloudBrightness")
	node : EnvSky = None
	pass
class CloudBrightnessGPlug(Plug):
	parent : CloudBrightnessPlug = PlugDescriptor("cloudBrightness")
	node : EnvSky = None
	pass
class CloudBrightnessRPlug(Plug):
	parent : CloudBrightnessPlug = PlugDescriptor("cloudBrightness")
	node : EnvSky = None
	pass
class CloudBrightnessPlug(Plug):
	cloudBrightnessB_ : CloudBrightnessBPlug = PlugDescriptor("cloudBrightnessB")
	cbb_ : CloudBrightnessBPlug = PlugDescriptor("cloudBrightnessB")
	cloudBrightnessG_ : CloudBrightnessGPlug = PlugDescriptor("cloudBrightnessG")
	cbg_ : CloudBrightnessGPlug = PlugDescriptor("cloudBrightnessG")
	cloudBrightnessR_ : CloudBrightnessRPlug = PlugDescriptor("cloudBrightnessR")
	cbr_ : CloudBrightnessRPlug = PlugDescriptor("cloudBrightnessR")
	node : EnvSky = None
	pass
class CloudSamplesPlug(Plug):
	node : EnvSky = None
	pass
class CloudTexturePlug(Plug):
	node : EnvSky = None
	pass
class DensityPlug(Plug):
	node : EnvSky = None
	pass
class DustDensityPlug(Plug):
	node : EnvSky = None
	pass
class ElevationPlug(Plug):
	node : EnvSky = None
	pass
class FloorAltitudePlug(Plug):
	node : EnvSky = None
	pass
class FloorColorBPlug(Plug):
	parent : FloorColorPlug = PlugDescriptor("floorColor")
	node : EnvSky = None
	pass
class FloorColorGPlug(Plug):
	parent : FloorColorPlug = PlugDescriptor("floorColor")
	node : EnvSky = None
	pass
class FloorColorRPlug(Plug):
	parent : FloorColorPlug = PlugDescriptor("floorColor")
	node : EnvSky = None
	pass
class FloorColorPlug(Plug):
	floorColorB_ : FloorColorBPlug = PlugDescriptor("floorColorB")
	fcb_ : FloorColorBPlug = PlugDescriptor("floorColorB")
	floorColorG_ : FloorColorGPlug = PlugDescriptor("floorColorG")
	fcg_ : FloorColorGPlug = PlugDescriptor("floorColorG")
	floorColorR_ : FloorColorRPlug = PlugDescriptor("floorColorR")
	fcr_ : FloorColorRPlug = PlugDescriptor("floorColorR")
	node : EnvSky = None
	pass
class FloorSamplesPlug(Plug):
	node : EnvSky = None
	pass
class HaloBrightnessBPlug(Plug):
	parent : HaloBrightnessPlug = PlugDescriptor("haloBrightness")
	node : EnvSky = None
	pass
class HaloBrightnessGPlug(Plug):
	parent : HaloBrightnessPlug = PlugDescriptor("haloBrightness")
	node : EnvSky = None
	pass
class HaloBrightnessRPlug(Plug):
	parent : HaloBrightnessPlug = PlugDescriptor("haloBrightness")
	node : EnvSky = None
	pass
class HaloBrightnessPlug(Plug):
	haloBrightnessB_ : HaloBrightnessBPlug = PlugDescriptor("haloBrightnessB")
	hbb_ : HaloBrightnessBPlug = PlugDescriptor("haloBrightnessB")
	haloBrightnessG_ : HaloBrightnessGPlug = PlugDescriptor("haloBrightnessG")
	hbg_ : HaloBrightnessGPlug = PlugDescriptor("haloBrightnessG")
	haloBrightnessR_ : HaloBrightnessRPlug = PlugDescriptor("haloBrightnessR")
	hbr_ : HaloBrightnessRPlug = PlugDescriptor("haloBrightnessR")
	node : EnvSky = None
	pass
class HaloSizePlug(Plug):
	node : EnvSky = None
	pass
class HasFloorPlug(Plug):
	node : EnvSky = None
	pass
class PointCameraXPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : EnvSky = None
	pass
class PointCameraYPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : EnvSky = None
	pass
class PointCameraZPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : EnvSky = None
	pass
class PointCameraPlug(Plug):
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	px_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	py_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pz_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	node : EnvSky = None
	pass
class PowerPlug(Plug):
	node : EnvSky = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : EnvSky = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : EnvSky = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : EnvSky = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : EnvSky = None
	pass
class SizePlug(Plug):
	node : EnvSky = None
	pass
class SkyBrightnessBPlug(Plug):
	parent : SkyBrightnessPlug = PlugDescriptor("skyBrightness")
	node : EnvSky = None
	pass
class SkyBrightnessGPlug(Plug):
	parent : SkyBrightnessPlug = PlugDescriptor("skyBrightness")
	node : EnvSky = None
	pass
class SkyBrightnessRPlug(Plug):
	parent : SkyBrightnessPlug = PlugDescriptor("skyBrightness")
	node : EnvSky = None
	pass
class SkyBrightnessPlug(Plug):
	skyBrightnessB_ : SkyBrightnessBPlug = PlugDescriptor("skyBrightnessB")
	skb_ : SkyBrightnessBPlug = PlugDescriptor("skyBrightnessB")
	skyBrightnessG_ : SkyBrightnessGPlug = PlugDescriptor("skyBrightnessG")
	skg_ : SkyBrightnessGPlug = PlugDescriptor("skyBrightnessG")
	skyBrightnessR_ : SkyBrightnessRPlug = PlugDescriptor("skyBrightnessR")
	skr_ : SkyBrightnessRPlug = PlugDescriptor("skyBrightnessR")
	node : EnvSky = None
	pass
class SkyRadiusPlug(Plug):
	node : EnvSky = None
	pass
class SkySamplesPlug(Plug):
	node : EnvSky = None
	pass
class SkyThicknessPlug(Plug):
	node : EnvSky = None
	pass
class SunBrightnessBPlug(Plug):
	parent : SunBrightnessPlug = PlugDescriptor("sunBrightness")
	node : EnvSky = None
	pass
class SunBrightnessGPlug(Plug):
	parent : SunBrightnessPlug = PlugDescriptor("sunBrightness")
	node : EnvSky = None
	pass
class SunBrightnessRPlug(Plug):
	parent : SunBrightnessPlug = PlugDescriptor("sunBrightness")
	node : EnvSky = None
	pass
class SunBrightnessPlug(Plug):
	sunBrightnessB_ : SunBrightnessBPlug = PlugDescriptor("sunBrightnessB")
	sub_ : SunBrightnessBPlug = PlugDescriptor("sunBrightnessB")
	sunBrightnessG_ : SunBrightnessGPlug = PlugDescriptor("sunBrightnessG")
	sug_ : SunBrightnessGPlug = PlugDescriptor("sunBrightnessG")
	sunBrightnessR_ : SunBrightnessRPlug = PlugDescriptor("sunBrightnessR")
	sur_ : SunBrightnessRPlug = PlugDescriptor("sunBrightnessR")
	node : EnvSky = None
	pass
class SunsetBrightnessBPlug(Plug):
	parent : SunsetBrightnessPlug = PlugDescriptor("sunsetBrightness")
	node : EnvSky = None
	pass
class SunsetBrightnessGPlug(Plug):
	parent : SunsetBrightnessPlug = PlugDescriptor("sunsetBrightness")
	node : EnvSky = None
	pass
class SunsetBrightnessRPlug(Plug):
	parent : SunsetBrightnessPlug = PlugDescriptor("sunsetBrightness")
	node : EnvSky = None
	pass
class SunsetBrightnessPlug(Plug):
	sunsetBrightnessB_ : SunsetBrightnessBPlug = PlugDescriptor("sunsetBrightnessB")
	ssb_ : SunsetBrightnessBPlug = PlugDescriptor("sunsetBrightnessB")
	sunsetBrightnessG_ : SunsetBrightnessGPlug = PlugDescriptor("sunsetBrightnessG")
	ssg_ : SunsetBrightnessGPlug = PlugDescriptor("sunsetBrightnessG")
	sunsetBrightnessR_ : SunsetBrightnessRPlug = PlugDescriptor("sunsetBrightnessR")
	ssr_ : SunsetBrightnessRPlug = PlugDescriptor("sunsetBrightnessR")
	node : EnvSky = None
	pass
class ThresholdPlug(Plug):
	node : EnvSky = None
	pass
class TotalBrightnessPlug(Plug):
	node : EnvSky = None
	pass
class UseTexturePlug(Plug):
	node : EnvSky = None
	pass
# endregion


# define node class
class EnvSky(TextureEnv):
	airDensity_ : AirDensityPlug = PlugDescriptor("airDensity")
	altitude_ : AltitudePlug = PlugDescriptor("altitude")
	azimuth_ : AzimuthPlug = PlugDescriptor("azimuth")
	blur_ : BlurPlug = PlugDescriptor("blur")
	cloudBrightnessB_ : CloudBrightnessBPlug = PlugDescriptor("cloudBrightnessB")
	cloudBrightnessG_ : CloudBrightnessGPlug = PlugDescriptor("cloudBrightnessG")
	cloudBrightnessR_ : CloudBrightnessRPlug = PlugDescriptor("cloudBrightnessR")
	cloudBrightness_ : CloudBrightnessPlug = PlugDescriptor("cloudBrightness")
	cloudSamples_ : CloudSamplesPlug = PlugDescriptor("cloudSamples")
	cloudTexture_ : CloudTexturePlug = PlugDescriptor("cloudTexture")
	density_ : DensityPlug = PlugDescriptor("density")
	dustDensity_ : DustDensityPlug = PlugDescriptor("dustDensity")
	elevation_ : ElevationPlug = PlugDescriptor("elevation")
	floorAltitude_ : FloorAltitudePlug = PlugDescriptor("floorAltitude")
	floorColorB_ : FloorColorBPlug = PlugDescriptor("floorColorB")
	floorColorG_ : FloorColorGPlug = PlugDescriptor("floorColorG")
	floorColorR_ : FloorColorRPlug = PlugDescriptor("floorColorR")
	floorColor_ : FloorColorPlug = PlugDescriptor("floorColor")
	floorSamples_ : FloorSamplesPlug = PlugDescriptor("floorSamples")
	haloBrightnessB_ : HaloBrightnessBPlug = PlugDescriptor("haloBrightnessB")
	haloBrightnessG_ : HaloBrightnessGPlug = PlugDescriptor("haloBrightnessG")
	haloBrightnessR_ : HaloBrightnessRPlug = PlugDescriptor("haloBrightnessR")
	haloBrightness_ : HaloBrightnessPlug = PlugDescriptor("haloBrightness")
	haloSize_ : HaloSizePlug = PlugDescriptor("haloSize")
	hasFloor_ : HasFloorPlug = PlugDescriptor("hasFloor")
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pointCamera_ : PointCameraPlug = PlugDescriptor("pointCamera")
	power_ : PowerPlug = PlugDescriptor("power")
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	refPointCamera_ : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	size_ : SizePlug = PlugDescriptor("size")
	skyBrightnessB_ : SkyBrightnessBPlug = PlugDescriptor("skyBrightnessB")
	skyBrightnessG_ : SkyBrightnessGPlug = PlugDescriptor("skyBrightnessG")
	skyBrightnessR_ : SkyBrightnessRPlug = PlugDescriptor("skyBrightnessR")
	skyBrightness_ : SkyBrightnessPlug = PlugDescriptor("skyBrightness")
	skyRadius_ : SkyRadiusPlug = PlugDescriptor("skyRadius")
	skySamples_ : SkySamplesPlug = PlugDescriptor("skySamples")
	skyThickness_ : SkyThicknessPlug = PlugDescriptor("skyThickness")
	sunBrightnessB_ : SunBrightnessBPlug = PlugDescriptor("sunBrightnessB")
	sunBrightnessG_ : SunBrightnessGPlug = PlugDescriptor("sunBrightnessG")
	sunBrightnessR_ : SunBrightnessRPlug = PlugDescriptor("sunBrightnessR")
	sunBrightness_ : SunBrightnessPlug = PlugDescriptor("sunBrightness")
	sunsetBrightnessB_ : SunsetBrightnessBPlug = PlugDescriptor("sunsetBrightnessB")
	sunsetBrightnessG_ : SunsetBrightnessGPlug = PlugDescriptor("sunsetBrightnessG")
	sunsetBrightnessR_ : SunsetBrightnessRPlug = PlugDescriptor("sunsetBrightnessR")
	sunsetBrightness_ : SunsetBrightnessPlug = PlugDescriptor("sunsetBrightness")
	threshold_ : ThresholdPlug = PlugDescriptor("threshold")
	totalBrightness_ : TotalBrightnessPlug = PlugDescriptor("totalBrightness")
	useTexture_ : UseTexturePlug = PlugDescriptor("useTexture")

	# node attributes

	typeName = "envSky"
	apiTypeInt = 494
	apiTypeStr = "kEnvSky"
	typeIdInt = 1380275019
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["airDensity", "altitude", "azimuth", "blur", "cloudBrightnessB", "cloudBrightnessG", "cloudBrightnessR", "cloudBrightness", "cloudSamples", "cloudTexture", "density", "dustDensity", "elevation", "floorAltitude", "floorColorB", "floorColorG", "floorColorR", "floorColor", "floorSamples", "haloBrightnessB", "haloBrightnessG", "haloBrightnessR", "haloBrightness", "haloSize", "hasFloor", "pointCameraX", "pointCameraY", "pointCameraZ", "pointCamera", "power", "refPointCameraX", "refPointCameraY", "refPointCameraZ", "refPointCamera", "size", "skyBrightnessB", "skyBrightnessG", "skyBrightnessR", "skyBrightness", "skyRadius", "skySamples", "skyThickness", "sunBrightnessB", "sunBrightnessG", "sunBrightnessR", "sunBrightness", "sunsetBrightnessB", "sunsetBrightnessG", "sunsetBrightnessR", "sunsetBrightness", "threshold", "totalBrightness", "useTexture"]
	nodeLeafPlugs = ["airDensity", "altitude", "azimuth", "blur", "cloudBrightness", "cloudSamples", "cloudTexture", "density", "dustDensity", "elevation", "floorAltitude", "floorColor", "floorSamples", "haloBrightness", "haloSize", "hasFloor", "pointCamera", "power", "refPointCamera", "size", "skyBrightness", "skyRadius", "skySamples", "skyThickness", "sunBrightness", "sunsetBrightness", "threshold", "totalBrightness", "useTexture"]
	pass

