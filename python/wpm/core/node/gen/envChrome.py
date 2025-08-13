

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
class FloorAltitudePlug(Plug):
	node : EnvChrome = None
	pass
class FloorColorBPlug(Plug):
	parent : FloorColorPlug = PlugDescriptor("floorColor")
	node : EnvChrome = None
	pass
class FloorColorGPlug(Plug):
	parent : FloorColorPlug = PlugDescriptor("floorColor")
	node : EnvChrome = None
	pass
class FloorColorRPlug(Plug):
	parent : FloorColorPlug = PlugDescriptor("floorColor")
	node : EnvChrome = None
	pass
class FloorColorPlug(Plug):
	floorColorB_ : FloorColorBPlug = PlugDescriptor("floorColorB")
	fcb_ : FloorColorBPlug = PlugDescriptor("floorColorB")
	floorColorG_ : FloorColorGPlug = PlugDescriptor("floorColorG")
	fcg_ : FloorColorGPlug = PlugDescriptor("floorColorG")
	floorColorR_ : FloorColorRPlug = PlugDescriptor("floorColorR")
	fcr_ : FloorColorRPlug = PlugDescriptor("floorColorR")
	node : EnvChrome = None
	pass
class GridColorBPlug(Plug):
	parent : GridColorPlug = PlugDescriptor("gridColor")
	node : EnvChrome = None
	pass
class GridColorGPlug(Plug):
	parent : GridColorPlug = PlugDescriptor("gridColor")
	node : EnvChrome = None
	pass
class GridColorRPlug(Plug):
	parent : GridColorPlug = PlugDescriptor("gridColor")
	node : EnvChrome = None
	pass
class GridColorPlug(Plug):
	gridColorB_ : GridColorBPlug = PlugDescriptor("gridColorB")
	gcb_ : GridColorBPlug = PlugDescriptor("gridColorB")
	gridColorG_ : GridColorGPlug = PlugDescriptor("gridColorG")
	gcg_ : GridColorGPlug = PlugDescriptor("gridColorG")
	gridColorR_ : GridColorRPlug = PlugDescriptor("gridColorR")
	gcr_ : GridColorRPlug = PlugDescriptor("gridColorR")
	node : EnvChrome = None
	pass
class GridDepthPlug(Plug):
	node : EnvChrome = None
	pass
class GridDepthGainPlug(Plug):
	node : EnvChrome = None
	pass
class GridDepthOffsetPlug(Plug):
	node : EnvChrome = None
	pass
class GridWidthPlug(Plug):
	node : EnvChrome = None
	pass
class GridWidthGainPlug(Plug):
	node : EnvChrome = None
	pass
class GridWidthOffsetPlug(Plug):
	node : EnvChrome = None
	pass
class HorizonColorBPlug(Plug):
	parent : HorizonColorPlug = PlugDescriptor("horizonColor")
	node : EnvChrome = None
	pass
class HorizonColorGPlug(Plug):
	parent : HorizonColorPlug = PlugDescriptor("horizonColor")
	node : EnvChrome = None
	pass
class HorizonColorRPlug(Plug):
	parent : HorizonColorPlug = PlugDescriptor("horizonColor")
	node : EnvChrome = None
	pass
class HorizonColorPlug(Plug):
	horizonColorB_ : HorizonColorBPlug = PlugDescriptor("horizonColorB")
	hcb_ : HorizonColorBPlug = PlugDescriptor("horizonColorB")
	horizonColorG_ : HorizonColorGPlug = PlugDescriptor("horizonColorG")
	hcg_ : HorizonColorGPlug = PlugDescriptor("horizonColorG")
	horizonColorR_ : HorizonColorRPlug = PlugDescriptor("horizonColorR")
	hcr_ : HorizonColorRPlug = PlugDescriptor("horizonColorR")
	node : EnvChrome = None
	pass
class LightColorBPlug(Plug):
	parent : LightColorPlug = PlugDescriptor("lightColor")
	node : EnvChrome = None
	pass
class LightColorGPlug(Plug):
	parent : LightColorPlug = PlugDescriptor("lightColor")
	node : EnvChrome = None
	pass
class LightColorRPlug(Plug):
	parent : LightColorPlug = PlugDescriptor("lightColor")
	node : EnvChrome = None
	pass
class LightColorPlug(Plug):
	lightColorB_ : LightColorBPlug = PlugDescriptor("lightColorB")
	lcb_ : LightColorBPlug = PlugDescriptor("lightColorB")
	lightColorG_ : LightColorGPlug = PlugDescriptor("lightColorG")
	lcg_ : LightColorGPlug = PlugDescriptor("lightColorG")
	lightColorR_ : LightColorRPlug = PlugDescriptor("lightColorR")
	lcr_ : LightColorRPlug = PlugDescriptor("lightColorR")
	node : EnvChrome = None
	pass
class LightDepthPlug(Plug):
	node : EnvChrome = None
	pass
class LightDepthGainPlug(Plug):
	node : EnvChrome = None
	pass
class LightDepthOffsetPlug(Plug):
	node : EnvChrome = None
	pass
class LightWidthPlug(Plug):
	node : EnvChrome = None
	pass
class LightWidthGainPlug(Plug):
	node : EnvChrome = None
	pass
class LightWidthOffsetPlug(Plug):
	node : EnvChrome = None
	pass
class PointCameraXPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : EnvChrome = None
	pass
class PointCameraYPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : EnvChrome = None
	pass
class PointCameraZPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : EnvChrome = None
	pass
class PointCameraPlug(Plug):
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	px_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	py_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pz_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	node : EnvChrome = None
	pass
class RealFloorPlug(Plug):
	node : EnvChrome = None
	pass
class SkyColorBPlug(Plug):
	parent : SkyColorPlug = PlugDescriptor("skyColor")
	node : EnvChrome = None
	pass
class SkyColorGPlug(Plug):
	parent : SkyColorPlug = PlugDescriptor("skyColor")
	node : EnvChrome = None
	pass
class SkyColorRPlug(Plug):
	parent : SkyColorPlug = PlugDescriptor("skyColor")
	node : EnvChrome = None
	pass
class SkyColorPlug(Plug):
	skyColorB_ : SkyColorBPlug = PlugDescriptor("skyColorB")
	scb_ : SkyColorBPlug = PlugDescriptor("skyColorB")
	skyColorG_ : SkyColorGPlug = PlugDescriptor("skyColorG")
	scg_ : SkyColorGPlug = PlugDescriptor("skyColorG")
	skyColorR_ : SkyColorRPlug = PlugDescriptor("skyColorR")
	scr_ : SkyColorRPlug = PlugDescriptor("skyColorR")
	node : EnvChrome = None
	pass
class ZenithColorBPlug(Plug):
	parent : ZenithColorPlug = PlugDescriptor("zenithColor")
	node : EnvChrome = None
	pass
class ZenithColorGPlug(Plug):
	parent : ZenithColorPlug = PlugDescriptor("zenithColor")
	node : EnvChrome = None
	pass
class ZenithColorRPlug(Plug):
	parent : ZenithColorPlug = PlugDescriptor("zenithColor")
	node : EnvChrome = None
	pass
class ZenithColorPlug(Plug):
	zenithColorB_ : ZenithColorBPlug = PlugDescriptor("zenithColorB")
	zcb_ : ZenithColorBPlug = PlugDescriptor("zenithColorB")
	zenithColorG_ : ZenithColorGPlug = PlugDescriptor("zenithColorG")
	zcg_ : ZenithColorGPlug = PlugDescriptor("zenithColorG")
	zenithColorR_ : ZenithColorRPlug = PlugDescriptor("zenithColorR")
	zcr_ : ZenithColorRPlug = PlugDescriptor("zenithColorR")
	node : EnvChrome = None
	pass
# endregion


# define node class
class EnvChrome(TextureEnv):
	floorAltitude_ : FloorAltitudePlug = PlugDescriptor("floorAltitude")
	floorColorB_ : FloorColorBPlug = PlugDescriptor("floorColorB")
	floorColorG_ : FloorColorGPlug = PlugDescriptor("floorColorG")
	floorColorR_ : FloorColorRPlug = PlugDescriptor("floorColorR")
	floorColor_ : FloorColorPlug = PlugDescriptor("floorColor")
	gridColorB_ : GridColorBPlug = PlugDescriptor("gridColorB")
	gridColorG_ : GridColorGPlug = PlugDescriptor("gridColorG")
	gridColorR_ : GridColorRPlug = PlugDescriptor("gridColorR")
	gridColor_ : GridColorPlug = PlugDescriptor("gridColor")
	gridDepth_ : GridDepthPlug = PlugDescriptor("gridDepth")
	gridDepthGain_ : GridDepthGainPlug = PlugDescriptor("gridDepthGain")
	gridDepthOffset_ : GridDepthOffsetPlug = PlugDescriptor("gridDepthOffset")
	gridWidth_ : GridWidthPlug = PlugDescriptor("gridWidth")
	gridWidthGain_ : GridWidthGainPlug = PlugDescriptor("gridWidthGain")
	gridWidthOffset_ : GridWidthOffsetPlug = PlugDescriptor("gridWidthOffset")
	horizonColorB_ : HorizonColorBPlug = PlugDescriptor("horizonColorB")
	horizonColorG_ : HorizonColorGPlug = PlugDescriptor("horizonColorG")
	horizonColorR_ : HorizonColorRPlug = PlugDescriptor("horizonColorR")
	horizonColor_ : HorizonColorPlug = PlugDescriptor("horizonColor")
	lightColorB_ : LightColorBPlug = PlugDescriptor("lightColorB")
	lightColorG_ : LightColorGPlug = PlugDescriptor("lightColorG")
	lightColorR_ : LightColorRPlug = PlugDescriptor("lightColorR")
	lightColor_ : LightColorPlug = PlugDescriptor("lightColor")
	lightDepth_ : LightDepthPlug = PlugDescriptor("lightDepth")
	lightDepthGain_ : LightDepthGainPlug = PlugDescriptor("lightDepthGain")
	lightDepthOffset_ : LightDepthOffsetPlug = PlugDescriptor("lightDepthOffset")
	lightWidth_ : LightWidthPlug = PlugDescriptor("lightWidth")
	lightWidthGain_ : LightWidthGainPlug = PlugDescriptor("lightWidthGain")
	lightWidthOffset_ : LightWidthOffsetPlug = PlugDescriptor("lightWidthOffset")
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pointCamera_ : PointCameraPlug = PlugDescriptor("pointCamera")
	realFloor_ : RealFloorPlug = PlugDescriptor("realFloor")
	skyColorB_ : SkyColorBPlug = PlugDescriptor("skyColorB")
	skyColorG_ : SkyColorGPlug = PlugDescriptor("skyColorG")
	skyColorR_ : SkyColorRPlug = PlugDescriptor("skyColorR")
	skyColor_ : SkyColorPlug = PlugDescriptor("skyColor")
	zenithColorB_ : ZenithColorBPlug = PlugDescriptor("zenithColorB")
	zenithColorG_ : ZenithColorGPlug = PlugDescriptor("zenithColorG")
	zenithColorR_ : ZenithColorRPlug = PlugDescriptor("zenithColorR")
	zenithColor_ : ZenithColorPlug = PlugDescriptor("zenithColor")

	# node attributes

	typeName = "envChrome"
	apiTypeInt = 493
	apiTypeStr = "kEnvChrome"
	typeIdInt = 1380270920
	MFnCls = om.MFnDependencyNode
	pass

