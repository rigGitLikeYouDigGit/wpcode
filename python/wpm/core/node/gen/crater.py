

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
class BalancePlug(Plug):
	node : Crater = None
	pass
class Channel1BPlug(Plug):
	parent : Channel1Plug = PlugDescriptor("channel1")
	node : Crater = None
	pass
class Channel1GPlug(Plug):
	parent : Channel1Plug = PlugDescriptor("channel1")
	node : Crater = None
	pass
class Channel1RPlug(Plug):
	parent : Channel1Plug = PlugDescriptor("channel1")
	node : Crater = None
	pass
class Channel1Plug(Plug):
	channel1B_ : Channel1BPlug = PlugDescriptor("channel1B")
	c1b_ : Channel1BPlug = PlugDescriptor("channel1B")
	channel1G_ : Channel1GPlug = PlugDescriptor("channel1G")
	c1g_ : Channel1GPlug = PlugDescriptor("channel1G")
	channel1R_ : Channel1RPlug = PlugDescriptor("channel1R")
	c1r_ : Channel1RPlug = PlugDescriptor("channel1R")
	node : Crater = None
	pass
class Channel2BPlug(Plug):
	parent : Channel2Plug = PlugDescriptor("channel2")
	node : Crater = None
	pass
class Channel2GPlug(Plug):
	parent : Channel2Plug = PlugDescriptor("channel2")
	node : Crater = None
	pass
class Channel2RPlug(Plug):
	parent : Channel2Plug = PlugDescriptor("channel2")
	node : Crater = None
	pass
class Channel2Plug(Plug):
	channel2B_ : Channel2BPlug = PlugDescriptor("channel2B")
	c2b_ : Channel2BPlug = PlugDescriptor("channel2B")
	channel2G_ : Channel2GPlug = PlugDescriptor("channel2G")
	c2g_ : Channel2GPlug = PlugDescriptor("channel2G")
	channel2R_ : Channel2RPlug = PlugDescriptor("channel2R")
	c2r_ : Channel2RPlug = PlugDescriptor("channel2R")
	node : Crater = None
	pass
class Channel3BPlug(Plug):
	parent : Channel3Plug = PlugDescriptor("channel3")
	node : Crater = None
	pass
class Channel3GPlug(Plug):
	parent : Channel3Plug = PlugDescriptor("channel3")
	node : Crater = None
	pass
class Channel3RPlug(Plug):
	parent : Channel3Plug = PlugDescriptor("channel3")
	node : Crater = None
	pass
class Channel3Plug(Plug):
	channel3B_ : Channel3BPlug = PlugDescriptor("channel3B")
	c3b_ : Channel3BPlug = PlugDescriptor("channel3B")
	channel3G_ : Channel3GPlug = PlugDescriptor("channel3G")
	c3g_ : Channel3GPlug = PlugDescriptor("channel3G")
	channel3R_ : Channel3RPlug = PlugDescriptor("channel3R")
	c3r_ : Channel3RPlug = PlugDescriptor("channel3R")
	node : Crater = None
	pass
class FrequencyPlug(Plug):
	node : Crater = None
	pass
class MeltPlug(Plug):
	node : Crater = None
	pass
class NormBalancePlug(Plug):
	node : Crater = None
	pass
class NormDepthPlug(Plug):
	node : Crater = None
	pass
class NormFrequencyPlug(Plug):
	node : Crater = None
	pass
class NormMeltPlug(Plug):
	node : Crater = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Crater = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Crater = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Crater = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : Crater = None
	pass
class OutNormalXPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Crater = None
	pass
class OutNormalYPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Crater = None
	pass
class OutNormalZPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Crater = None
	pass
class OutNormalPlug(Plug):
	outNormalX_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	ox_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	outNormalY_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	oy_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	outNormalZ_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	oz_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	node : Crater = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Crater = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Crater = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Crater = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : Crater = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Crater = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Crater = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Crater = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	roy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	roz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : Crater = None
	pass
class ShakerPlug(Plug):
	node : Crater = None
	pass
# endregion


# define node class
class Crater(Texture3d):
	balance_ : BalancePlug = PlugDescriptor("balance")
	channel1B_ : Channel1BPlug = PlugDescriptor("channel1B")
	channel1G_ : Channel1GPlug = PlugDescriptor("channel1G")
	channel1R_ : Channel1RPlug = PlugDescriptor("channel1R")
	channel1_ : Channel1Plug = PlugDescriptor("channel1")
	channel2B_ : Channel2BPlug = PlugDescriptor("channel2B")
	channel2G_ : Channel2GPlug = PlugDescriptor("channel2G")
	channel2R_ : Channel2RPlug = PlugDescriptor("channel2R")
	channel2_ : Channel2Plug = PlugDescriptor("channel2")
	channel3B_ : Channel3BPlug = PlugDescriptor("channel3B")
	channel3G_ : Channel3GPlug = PlugDescriptor("channel3G")
	channel3R_ : Channel3RPlug = PlugDescriptor("channel3R")
	channel3_ : Channel3Plug = PlugDescriptor("channel3")
	frequency_ : FrequencyPlug = PlugDescriptor("frequency")
	melt_ : MeltPlug = PlugDescriptor("melt")
	normBalance_ : NormBalancePlug = PlugDescriptor("normBalance")
	normDepth_ : NormDepthPlug = PlugDescriptor("normDepth")
	normFrequency_ : NormFrequencyPlug = PlugDescriptor("normFrequency")
	normMelt_ : NormMeltPlug = PlugDescriptor("normMelt")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	outNormalX_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	outNormalY_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	outNormalZ_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	outNormal_ : OutNormalPlug = PlugDescriptor("outNormal")
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	refPointCamera_ : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	refPointObj_ : RefPointObjPlug = PlugDescriptor("refPointObj")
	shaker_ : ShakerPlug = PlugDescriptor("shaker")

	# node attributes

	typeName = "crater"
	apiTypeInt = 510
	apiTypeStr = "kCrater"
	typeIdInt = 1381184560
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["balance", "channel1B", "channel1G", "channel1R", "channel1", "channel2B", "channel2G", "channel2R", "channel2", "channel3B", "channel3G", "channel3R", "channel3", "frequency", "melt", "normBalance", "normDepth", "normFrequency", "normMelt", "normalCameraX", "normalCameraY", "normalCameraZ", "normalCamera", "outNormalX", "outNormalY", "outNormalZ", "outNormal", "refPointCameraX", "refPointCameraY", "refPointCameraZ", "refPointCamera", "refPointObjX", "refPointObjY", "refPointObjZ", "refPointObj", "shaker"]
	nodeLeafPlugs = ["balance", "channel1", "channel2", "channel3", "frequency", "melt", "normBalance", "normDepth", "normFrequency", "normMelt", "normalCamera", "outNormal", "refPointCamera", "refPointObj", "shaker"]
	pass

