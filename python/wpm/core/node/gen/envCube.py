

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
class BackBPlug(Plug):
	parent : BackPlug = PlugDescriptor("back")
	node : EnvCube = None
	pass
class BackGPlug(Plug):
	parent : BackPlug = PlugDescriptor("back")
	node : EnvCube = None
	pass
class BackRPlug(Plug):
	parent : BackPlug = PlugDescriptor("back")
	node : EnvCube = None
	pass
class BackPlug(Plug):
	backB_ : BackBPlug = PlugDescriptor("backB")
	bab_ : BackBPlug = PlugDescriptor("backB")
	backG_ : BackGPlug = PlugDescriptor("backG")
	bag_ : BackGPlug = PlugDescriptor("backG")
	backR_ : BackRPlug = PlugDescriptor("backR")
	bar_ : BackRPlug = PlugDescriptor("backR")
	node : EnvCube = None
	pass
class BottomBPlug(Plug):
	parent : BottomPlug = PlugDescriptor("bottom")
	node : EnvCube = None
	pass
class BottomGPlug(Plug):
	parent : BottomPlug = PlugDescriptor("bottom")
	node : EnvCube = None
	pass
class BottomRPlug(Plug):
	parent : BottomPlug = PlugDescriptor("bottom")
	node : EnvCube = None
	pass
class BottomPlug(Plug):
	bottomB_ : BottomBPlug = PlugDescriptor("bottomB")
	bob_ : BottomBPlug = PlugDescriptor("bottomB")
	bottomG_ : BottomGPlug = PlugDescriptor("bottomG")
	bog_ : BottomGPlug = PlugDescriptor("bottomG")
	bottomR_ : BottomRPlug = PlugDescriptor("bottomR")
	bor_ : BottomRPlug = PlugDescriptor("bottomR")
	node : EnvCube = None
	pass
class FrontBPlug(Plug):
	parent : FrontPlug = PlugDescriptor("front")
	node : EnvCube = None
	pass
class FrontGPlug(Plug):
	parent : FrontPlug = PlugDescriptor("front")
	node : EnvCube = None
	pass
class FrontRPlug(Plug):
	parent : FrontPlug = PlugDescriptor("front")
	node : EnvCube = None
	pass
class FrontPlug(Plug):
	frontB_ : FrontBPlug = PlugDescriptor("frontB")
	frb_ : FrontBPlug = PlugDescriptor("frontB")
	frontG_ : FrontGPlug = PlugDescriptor("frontG")
	frg_ : FrontGPlug = PlugDescriptor("frontG")
	frontR_ : FrontRPlug = PlugDescriptor("frontR")
	frr_ : FrontRPlug = PlugDescriptor("frontR")
	node : EnvCube = None
	pass
class InfiniteSizePlug(Plug):
	node : EnvCube = None
	pass
class InfoBitsPlug(Plug):
	node : EnvCube = None
	pass
class LeftBPlug(Plug):
	parent : LeftPlug = PlugDescriptor("left")
	node : EnvCube = None
	pass
class LeftGPlug(Plug):
	parent : LeftPlug = PlugDescriptor("left")
	node : EnvCube = None
	pass
class LeftRPlug(Plug):
	parent : LeftPlug = PlugDescriptor("left")
	node : EnvCube = None
	pass
class LeftPlug(Plug):
	leftB_ : LeftBPlug = PlugDescriptor("leftB")
	leb_ : LeftBPlug = PlugDescriptor("leftB")
	leftG_ : LeftGPlug = PlugDescriptor("leftG")
	leg_ : LeftGPlug = PlugDescriptor("leftG")
	leftR_ : LeftRPlug = PlugDescriptor("leftR")
	ler_ : LeftRPlug = PlugDescriptor("leftR")
	node : EnvCube = None
	pass
class LookupTypePlug(Plug):
	node : EnvCube = None
	pass
class PointWorldXPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : EnvCube = None
	pass
class PointWorldYPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : EnvCube = None
	pass
class PointWorldZPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : EnvCube = None
	pass
class PointWorldPlug(Plug):
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pwx_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pwy_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	pwz_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	node : EnvCube = None
	pass
class RightBPlug(Plug):
	parent : RightPlug = PlugDescriptor("right")
	node : EnvCube = None
	pass
class RightGPlug(Plug):
	parent : RightPlug = PlugDescriptor("right")
	node : EnvCube = None
	pass
class RightRPlug(Plug):
	parent : RightPlug = PlugDescriptor("right")
	node : EnvCube = None
	pass
class RightPlug(Plug):
	rightB_ : RightBPlug = PlugDescriptor("rightB")
	rib_ : RightBPlug = PlugDescriptor("rightB")
	rightG_ : RightGPlug = PlugDescriptor("rightG")
	rig_ : RightGPlug = PlugDescriptor("rightG")
	rightR_ : RightRPlug = PlugDescriptor("rightR")
	rir_ : RightRPlug = PlugDescriptor("rightR")
	node : EnvCube = None
	pass
class TopBPlug(Plug):
	parent : TopPlug = PlugDescriptor("top")
	node : EnvCube = None
	pass
class TopGPlug(Plug):
	parent : TopPlug = PlugDescriptor("top")
	node : EnvCube = None
	pass
class TopRPlug(Plug):
	parent : TopPlug = PlugDescriptor("top")
	node : EnvCube = None
	pass
class TopPlug(Plug):
	topB_ : TopBPlug = PlugDescriptor("topB")
	tob_ : TopBPlug = PlugDescriptor("topB")
	topG_ : TopGPlug = PlugDescriptor("topG")
	tog_ : TopGPlug = PlugDescriptor("topG")
	topR_ : TopRPlug = PlugDescriptor("topR")
	tor_ : TopRPlug = PlugDescriptor("topR")
	node : EnvCube = None
	pass
# endregion


# define node class
class EnvCube(TextureEnv):
	backB_ : BackBPlug = PlugDescriptor("backB")
	backG_ : BackGPlug = PlugDescriptor("backG")
	backR_ : BackRPlug = PlugDescriptor("backR")
	back_ : BackPlug = PlugDescriptor("back")
	bottomB_ : BottomBPlug = PlugDescriptor("bottomB")
	bottomG_ : BottomGPlug = PlugDescriptor("bottomG")
	bottomR_ : BottomRPlug = PlugDescriptor("bottomR")
	bottom_ : BottomPlug = PlugDescriptor("bottom")
	frontB_ : FrontBPlug = PlugDescriptor("frontB")
	frontG_ : FrontGPlug = PlugDescriptor("frontG")
	frontR_ : FrontRPlug = PlugDescriptor("frontR")
	front_ : FrontPlug = PlugDescriptor("front")
	infiniteSize_ : InfiniteSizePlug = PlugDescriptor("infiniteSize")
	infoBits_ : InfoBitsPlug = PlugDescriptor("infoBits")
	leftB_ : LeftBPlug = PlugDescriptor("leftB")
	leftG_ : LeftGPlug = PlugDescriptor("leftG")
	leftR_ : LeftRPlug = PlugDescriptor("leftR")
	left_ : LeftPlug = PlugDescriptor("left")
	lookupType_ : LookupTypePlug = PlugDescriptor("lookupType")
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	pointWorld_ : PointWorldPlug = PlugDescriptor("pointWorld")
	rightB_ : RightBPlug = PlugDescriptor("rightB")
	rightG_ : RightGPlug = PlugDescriptor("rightG")
	rightR_ : RightRPlug = PlugDescriptor("rightR")
	right_ : RightPlug = PlugDescriptor("right")
	topB_ : TopBPlug = PlugDescriptor("topB")
	topG_ : TopGPlug = PlugDescriptor("topG")
	topR_ : TopRPlug = PlugDescriptor("topR")
	top_ : TopPlug = PlugDescriptor("top")

	# node attributes

	typeName = "envCube"
	apiTypeInt = 492
	apiTypeStr = "kEnvCube"
	typeIdInt = 1380270914
	MFnCls = om.MFnDependencyNode
	pass

