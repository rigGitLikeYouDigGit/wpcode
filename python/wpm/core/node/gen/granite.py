

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
class CellSizePlug(Plug):
	node : Granite = None
	pass
class Color1BPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : Granite = None
	pass
class Color1GPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : Granite = None
	pass
class Color1RPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : Granite = None
	pass
class Color1Plug(Plug):
	color1B_ : Color1BPlug = PlugDescriptor("color1B")
	c1b_ : Color1BPlug = PlugDescriptor("color1B")
	color1G_ : Color1GPlug = PlugDescriptor("color1G")
	c1g_ : Color1GPlug = PlugDescriptor("color1G")
	color1R_ : Color1RPlug = PlugDescriptor("color1R")
	c1r_ : Color1RPlug = PlugDescriptor("color1R")
	node : Granite = None
	pass
class Color2BPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : Granite = None
	pass
class Color2GPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : Granite = None
	pass
class Color2RPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : Granite = None
	pass
class Color2Plug(Plug):
	color2B_ : Color2BPlug = PlugDescriptor("color2B")
	c2b_ : Color2BPlug = PlugDescriptor("color2B")
	color2G_ : Color2GPlug = PlugDescriptor("color2G")
	c2g_ : Color2GPlug = PlugDescriptor("color2G")
	color2R_ : Color2RPlug = PlugDescriptor("color2R")
	c2r_ : Color2RPlug = PlugDescriptor("color2R")
	node : Granite = None
	pass
class Color3BPlug(Plug):
	parent : Color3Plug = PlugDescriptor("color3")
	node : Granite = None
	pass
class Color3GPlug(Plug):
	parent : Color3Plug = PlugDescriptor("color3")
	node : Granite = None
	pass
class Color3RPlug(Plug):
	parent : Color3Plug = PlugDescriptor("color3")
	node : Granite = None
	pass
class Color3Plug(Plug):
	color3B_ : Color3BPlug = PlugDescriptor("color3B")
	c3b_ : Color3BPlug = PlugDescriptor("color3B")
	color3G_ : Color3GPlug = PlugDescriptor("color3G")
	c3g_ : Color3GPlug = PlugDescriptor("color3G")
	color3R_ : Color3RPlug = PlugDescriptor("color3R")
	c3r_ : Color3RPlug = PlugDescriptor("color3R")
	node : Granite = None
	pass
class CreasesPlug(Plug):
	node : Granite = None
	pass
class DensityPlug(Plug):
	node : Granite = None
	pass
class FillerColorBPlug(Plug):
	parent : FillerColorPlug = PlugDescriptor("fillerColor")
	node : Granite = None
	pass
class FillerColorGPlug(Plug):
	parent : FillerColorPlug = PlugDescriptor("fillerColor")
	node : Granite = None
	pass
class FillerColorRPlug(Plug):
	parent : FillerColorPlug = PlugDescriptor("fillerColor")
	node : Granite = None
	pass
class FillerColorPlug(Plug):
	fillerColorB_ : FillerColorBPlug = PlugDescriptor("fillerColorB")
	fcb_ : FillerColorBPlug = PlugDescriptor("fillerColorB")
	fillerColorG_ : FillerColorGPlug = PlugDescriptor("fillerColorG")
	fcg_ : FillerColorGPlug = PlugDescriptor("fillerColorG")
	fillerColorR_ : FillerColorRPlug = PlugDescriptor("fillerColorR")
	fcr_ : FillerColorRPlug = PlugDescriptor("fillerColorR")
	node : Granite = None
	pass
class MixRatioPlug(Plug):
	node : Granite = None
	pass
class RandomnessPlug(Plug):
	node : Granite = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Granite = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Granite = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Granite = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : Granite = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Granite = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Granite = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Granite = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	roy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	roz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : Granite = None
	pass
class SpottynessPlug(Plug):
	node : Granite = None
	pass
class ThresholdPlug(Plug):
	node : Granite = None
	pass
# endregion


# define node class
class Granite(Texture3d):
	cellSize_ : CellSizePlug = PlugDescriptor("cellSize")
	color1B_ : Color1BPlug = PlugDescriptor("color1B")
	color1G_ : Color1GPlug = PlugDescriptor("color1G")
	color1R_ : Color1RPlug = PlugDescriptor("color1R")
	color1_ : Color1Plug = PlugDescriptor("color1")
	color2B_ : Color2BPlug = PlugDescriptor("color2B")
	color2G_ : Color2GPlug = PlugDescriptor("color2G")
	color2R_ : Color2RPlug = PlugDescriptor("color2R")
	color2_ : Color2Plug = PlugDescriptor("color2")
	color3B_ : Color3BPlug = PlugDescriptor("color3B")
	color3G_ : Color3GPlug = PlugDescriptor("color3G")
	color3R_ : Color3RPlug = PlugDescriptor("color3R")
	color3_ : Color3Plug = PlugDescriptor("color3")
	creases_ : CreasesPlug = PlugDescriptor("creases")
	density_ : DensityPlug = PlugDescriptor("density")
	fillerColorB_ : FillerColorBPlug = PlugDescriptor("fillerColorB")
	fillerColorG_ : FillerColorGPlug = PlugDescriptor("fillerColorG")
	fillerColorR_ : FillerColorRPlug = PlugDescriptor("fillerColorR")
	fillerColor_ : FillerColorPlug = PlugDescriptor("fillerColor")
	mixRatio_ : MixRatioPlug = PlugDescriptor("mixRatio")
	randomness_ : RandomnessPlug = PlugDescriptor("randomness")
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	refPointCamera_ : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	refPointObj_ : RefPointObjPlug = PlugDescriptor("refPointObj")
	spottyness_ : SpottynessPlug = PlugDescriptor("spottyness")
	threshold_ : ThresholdPlug = PlugDescriptor("threshold")

	# node attributes

	typeName = "granite"
	apiTypeInt = 511
	apiTypeStr = "kGranite"
	typeIdInt = 1381254994
	MFnCls = om.MFnDependencyNode
	pass

