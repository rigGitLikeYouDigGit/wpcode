

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Texture2d = retriever.getNodeCls("Texture2d")
assert Texture2d
if T.TYPE_CHECKING:
	from .. import Texture2d

# add node doc



# region plug type defs
class AmplitudePlug(Plug):
	node : Mountain = None
	pass
class BoundaryPlug(Plug):
	node : Mountain = None
	pass
class DepthMaxPlug(Plug):
	node : Mountain = None
	pass
class RockColorBPlug(Plug):
	parent : RockColorPlug = PlugDescriptor("rockColor")
	node : Mountain = None
	pass
class RockColorGPlug(Plug):
	parent : RockColorPlug = PlugDescriptor("rockColor")
	node : Mountain = None
	pass
class RockColorRPlug(Plug):
	parent : RockColorPlug = PlugDescriptor("rockColor")
	node : Mountain = None
	pass
class RockColorPlug(Plug):
	rockColorB_ : RockColorBPlug = PlugDescriptor("rockColorB")
	rcb_ : RockColorBPlug = PlugDescriptor("rockColorB")
	rockColorG_ : RockColorGPlug = PlugDescriptor("rockColorG")
	rcg_ : RockColorGPlug = PlugDescriptor("rockColorG")
	rockColorR_ : RockColorRPlug = PlugDescriptor("rockColorR")
	rcr_ : RockColorRPlug = PlugDescriptor("rockColorR")
	node : Mountain = None
	pass
class RockRoughnessPlug(Plug):
	node : Mountain = None
	pass
class SnowAltitudePlug(Plug):
	node : Mountain = None
	pass
class SnowColorBPlug(Plug):
	parent : SnowColorPlug = PlugDescriptor("snowColor")
	node : Mountain = None
	pass
class SnowColorGPlug(Plug):
	parent : SnowColorPlug = PlugDescriptor("snowColor")
	node : Mountain = None
	pass
class SnowColorRPlug(Plug):
	parent : SnowColorPlug = PlugDescriptor("snowColor")
	node : Mountain = None
	pass
class SnowColorPlug(Plug):
	snowColorB_ : SnowColorBPlug = PlugDescriptor("snowColorB")
	scb_ : SnowColorBPlug = PlugDescriptor("snowColorB")
	snowColorG_ : SnowColorGPlug = PlugDescriptor("snowColorG")
	scg_ : SnowColorGPlug = PlugDescriptor("snowColorG")
	snowColorR_ : SnowColorRPlug = PlugDescriptor("snowColorR")
	scr_ : SnowColorRPlug = PlugDescriptor("snowColorR")
	node : Mountain = None
	pass
class SnowDropoffPlug(Plug):
	node : Mountain = None
	pass
class SnowRoughnessPlug(Plug):
	node : Mountain = None
	pass
class SnowSlopePlug(Plug):
	node : Mountain = None
	pass
# endregion


# define node class
class Mountain(Texture2d):
	amplitude_ : AmplitudePlug = PlugDescriptor("amplitude")
	boundary_ : BoundaryPlug = PlugDescriptor("boundary")
	depthMax_ : DepthMaxPlug = PlugDescriptor("depthMax")
	rockColorB_ : RockColorBPlug = PlugDescriptor("rockColorB")
	rockColorG_ : RockColorGPlug = PlugDescriptor("rockColorG")
	rockColorR_ : RockColorRPlug = PlugDescriptor("rockColorR")
	rockColor_ : RockColorPlug = PlugDescriptor("rockColor")
	rockRoughness_ : RockRoughnessPlug = PlugDescriptor("rockRoughness")
	snowAltitude_ : SnowAltitudePlug = PlugDescriptor("snowAltitude")
	snowColorB_ : SnowColorBPlug = PlugDescriptor("snowColorB")
	snowColorG_ : SnowColorGPlug = PlugDescriptor("snowColorG")
	snowColorR_ : SnowColorRPlug = PlugDescriptor("snowColorR")
	snowColor_ : SnowColorPlug = PlugDescriptor("snowColor")
	snowDropoff_ : SnowDropoffPlug = PlugDescriptor("snowDropoff")
	snowRoughness_ : SnowRoughnessPlug = PlugDescriptor("snowRoughness")
	snowSlope_ : SnowSlopePlug = PlugDescriptor("snowSlope")

	# node attributes

	typeName = "mountain"
	apiTypeInt = 503
	apiTypeStr = "kMountain"
	typeIdInt = 1381256532
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["amplitude", "boundary", "depthMax", "rockColorB", "rockColorG", "rockColorR", "rockColor", "rockRoughness", "snowAltitude", "snowColorB", "snowColorG", "snowColorR", "snowColor", "snowDropoff", "snowRoughness", "snowSlope"]
	nodeLeafPlugs = ["amplitude", "boundary", "depthMax", "rockColor", "rockRoughness", "snowAltitude", "snowColor", "snowDropoff", "snowRoughness", "snowSlope"]
	pass

