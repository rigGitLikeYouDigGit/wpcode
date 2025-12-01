

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
class Color1BPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : Rock = None
	pass
class Color1GPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : Rock = None
	pass
class Color1RPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : Rock = None
	pass
class Color1Plug(Plug):
	color1B_ : Color1BPlug = PlugDescriptor("color1B")
	c1b_ : Color1BPlug = PlugDescriptor("color1B")
	color1G_ : Color1GPlug = PlugDescriptor("color1G")
	c1g_ : Color1GPlug = PlugDescriptor("color1G")
	color1R_ : Color1RPlug = PlugDescriptor("color1R")
	c1r_ : Color1RPlug = PlugDescriptor("color1R")
	node : Rock = None
	pass
class Color2BPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : Rock = None
	pass
class Color2GPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : Rock = None
	pass
class Color2RPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : Rock = None
	pass
class Color2Plug(Plug):
	color2B_ : Color2BPlug = PlugDescriptor("color2B")
	c2b_ : Color2BPlug = PlugDescriptor("color2B")
	color2G_ : Color2GPlug = PlugDescriptor("color2G")
	c2g_ : Color2GPlug = PlugDescriptor("color2G")
	color2R_ : Color2RPlug = PlugDescriptor("color2R")
	c2r_ : Color2RPlug = PlugDescriptor("color2R")
	node : Rock = None
	pass
class DiffusionPlug(Plug):
	node : Rock = None
	pass
class GrainSizePlug(Plug):
	node : Rock = None
	pass
class MixRatioPlug(Plug):
	node : Rock = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Rock = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Rock = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Rock = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : Rock = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Rock = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Rock = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Rock = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	roy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	roz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : Rock = None
	pass
# endregion


# define node class
class Rock(Texture3d):
	color1B_ : Color1BPlug = PlugDescriptor("color1B")
	color1G_ : Color1GPlug = PlugDescriptor("color1G")
	color1R_ : Color1RPlug = PlugDescriptor("color1R")
	color1_ : Color1Plug = PlugDescriptor("color1")
	color2B_ : Color2BPlug = PlugDescriptor("color2B")
	color2G_ : Color2GPlug = PlugDescriptor("color2G")
	color2R_ : Color2RPlug = PlugDescriptor("color2R")
	color2_ : Color2Plug = PlugDescriptor("color2")
	diffusion_ : DiffusionPlug = PlugDescriptor("diffusion")
	grainSize_ : GrainSizePlug = PlugDescriptor("grainSize")
	mixRatio_ : MixRatioPlug = PlugDescriptor("mixRatio")
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	refPointCamera_ : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	refPointObj_ : RefPointObjPlug = PlugDescriptor("refPointObj")

	# node attributes

	typeName = "rock"
	apiTypeInt = 514
	apiTypeStr = "kRock"
	typeIdInt = 1381257803
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["color1B", "color1G", "color1R", "color1", "color2B", "color2G", "color2R", "color2", "diffusion", "grainSize", "mixRatio", "refPointCameraX", "refPointCameraY", "refPointCameraZ", "refPointCamera", "refPointObjX", "refPointObjY", "refPointObjZ", "refPointObj"]
	nodeLeafPlugs = ["color1", "color2", "diffusion", "grainSize", "mixRatio", "refPointCamera", "refPointObj"]
	pass

