

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
class Channel1BPlug(Plug):
	parent : Channel1Plug = PlugDescriptor("channel1")
	node : Stucco = None
	pass
class Channel1GPlug(Plug):
	parent : Channel1Plug = PlugDescriptor("channel1")
	node : Stucco = None
	pass
class Channel1RPlug(Plug):
	parent : Channel1Plug = PlugDescriptor("channel1")
	node : Stucco = None
	pass
class Channel1Plug(Plug):
	channel1B_ : Channel1BPlug = PlugDescriptor("channel1B")
	c1b_ : Channel1BPlug = PlugDescriptor("channel1B")
	channel1G_ : Channel1GPlug = PlugDescriptor("channel1G")
	c1g_ : Channel1GPlug = PlugDescriptor("channel1G")
	channel1R_ : Channel1RPlug = PlugDescriptor("channel1R")
	c1r_ : Channel1RPlug = PlugDescriptor("channel1R")
	node : Stucco = None
	pass
class Channel2BPlug(Plug):
	parent : Channel2Plug = PlugDescriptor("channel2")
	node : Stucco = None
	pass
class Channel2GPlug(Plug):
	parent : Channel2Plug = PlugDescriptor("channel2")
	node : Stucco = None
	pass
class Channel2RPlug(Plug):
	parent : Channel2Plug = PlugDescriptor("channel2")
	node : Stucco = None
	pass
class Channel2Plug(Plug):
	channel2B_ : Channel2BPlug = PlugDescriptor("channel2B")
	c2b_ : Channel2BPlug = PlugDescriptor("channel2B")
	channel2G_ : Channel2GPlug = PlugDescriptor("channel2G")
	c2g_ : Channel2GPlug = PlugDescriptor("channel2G")
	channel2R_ : Channel2RPlug = PlugDescriptor("channel2R")
	c2r_ : Channel2RPlug = PlugDescriptor("channel2R")
	node : Stucco = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Stucco = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Stucco = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Stucco = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : Stucco = None
	pass
class NormalDepthPlug(Plug):
	node : Stucco = None
	pass
class NormalMeltPlug(Plug):
	node : Stucco = None
	pass
class OutNormalXPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Stucco = None
	pass
class OutNormalYPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Stucco = None
	pass
class OutNormalZPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Stucco = None
	pass
class OutNormalPlug(Plug):
	outNormalX_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	onx_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	outNormalY_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	ony_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	outNormalZ_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	onz_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	node : Stucco = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Stucco = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Stucco = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Stucco = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : Stucco = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Stucco = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Stucco = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Stucco = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	roy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	roz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : Stucco = None
	pass
class ShakerPlug(Plug):
	node : Stucco = None
	pass
# endregion


# define node class
class Stucco(Texture3d):
	channel1B_ : Channel1BPlug = PlugDescriptor("channel1B")
	channel1G_ : Channel1GPlug = PlugDescriptor("channel1G")
	channel1R_ : Channel1RPlug = PlugDescriptor("channel1R")
	channel1_ : Channel1Plug = PlugDescriptor("channel1")
	channel2B_ : Channel2BPlug = PlugDescriptor("channel2B")
	channel2G_ : Channel2GPlug = PlugDescriptor("channel2G")
	channel2R_ : Channel2RPlug = PlugDescriptor("channel2R")
	channel2_ : Channel2Plug = PlugDescriptor("channel2")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	normalDepth_ : NormalDepthPlug = PlugDescriptor("normalDepth")
	normalMelt_ : NormalMeltPlug = PlugDescriptor("normalMelt")
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

	typeName = "stucco"
	apiTypeInt = 517
	apiTypeStr = "kStucco"
	typeIdInt = 1381185072
	MFnCls = om.MFnDependencyNode
	pass

