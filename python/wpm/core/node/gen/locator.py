

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
GeometryShape = retriever.getNodeCls("GeometryShape")
assert GeometryShape
if T.TYPE_CHECKING:
	from .. import GeometryShape

# add node doc



# region plug type defs
class LocalPositionXPlug(Plug):
	parent : LocalPositionPlug = PlugDescriptor("localPosition")
	node : Locator = None
	pass
class LocalPositionYPlug(Plug):
	parent : LocalPositionPlug = PlugDescriptor("localPosition")
	node : Locator = None
	pass
class LocalPositionZPlug(Plug):
	parent : LocalPositionPlug = PlugDescriptor("localPosition")
	node : Locator = None
	pass
class LocalPositionPlug(Plug):
	localPositionX_ : LocalPositionXPlug = PlugDescriptor("localPositionX")
	lpx_ : LocalPositionXPlug = PlugDescriptor("localPositionX")
	localPositionY_ : LocalPositionYPlug = PlugDescriptor("localPositionY")
	lpy_ : LocalPositionYPlug = PlugDescriptor("localPositionY")
	localPositionZ_ : LocalPositionZPlug = PlugDescriptor("localPositionZ")
	lpz_ : LocalPositionZPlug = PlugDescriptor("localPositionZ")
	node : Locator = None
	pass
class LocalScaleXPlug(Plug):
	parent : LocalScalePlug = PlugDescriptor("localScale")
	node : Locator = None
	pass
class LocalScaleYPlug(Plug):
	parent : LocalScalePlug = PlugDescriptor("localScale")
	node : Locator = None
	pass
class LocalScaleZPlug(Plug):
	parent : LocalScalePlug = PlugDescriptor("localScale")
	node : Locator = None
	pass
class LocalScalePlug(Plug):
	localScaleX_ : LocalScaleXPlug = PlugDescriptor("localScaleX")
	lsx_ : LocalScaleXPlug = PlugDescriptor("localScaleX")
	localScaleY_ : LocalScaleYPlug = PlugDescriptor("localScaleY")
	lsy_ : LocalScaleYPlug = PlugDescriptor("localScaleY")
	localScaleZ_ : LocalScaleZPlug = PlugDescriptor("localScaleZ")
	lsz_ : LocalScaleZPlug = PlugDescriptor("localScaleZ")
	node : Locator = None
	pass
class UnderWorldObjectPlug(Plug):
	node : Locator = None
	pass
class WorldPositionXPlug(Plug):
	parent : WorldPositionPlug = PlugDescriptor("worldPosition")
	node : Locator = None
	pass
class WorldPositionYPlug(Plug):
	parent : WorldPositionPlug = PlugDescriptor("worldPosition")
	node : Locator = None
	pass
class WorldPositionZPlug(Plug):
	parent : WorldPositionPlug = PlugDescriptor("worldPosition")
	node : Locator = None
	pass
class WorldPositionPlug(Plug):
	worldPositionX_ : WorldPositionXPlug = PlugDescriptor("worldPositionX")
	wpx_ : WorldPositionXPlug = PlugDescriptor("worldPositionX")
	worldPositionY_ : WorldPositionYPlug = PlugDescriptor("worldPositionY")
	wpy_ : WorldPositionYPlug = PlugDescriptor("worldPositionY")
	worldPositionZ_ : WorldPositionZPlug = PlugDescriptor("worldPositionZ")
	wpz_ : WorldPositionZPlug = PlugDescriptor("worldPositionZ")
	node : Locator = None
	pass
# endregion


# define node class
class Locator(GeometryShape):
	localPositionX_ : LocalPositionXPlug = PlugDescriptor("localPositionX")
	localPositionY_ : LocalPositionYPlug = PlugDescriptor("localPositionY")
	localPositionZ_ : LocalPositionZPlug = PlugDescriptor("localPositionZ")
	localPosition_ : LocalPositionPlug = PlugDescriptor("localPosition")
	localScaleX_ : LocalScaleXPlug = PlugDescriptor("localScaleX")
	localScaleY_ : LocalScaleYPlug = PlugDescriptor("localScaleY")
	localScaleZ_ : LocalScaleZPlug = PlugDescriptor("localScaleZ")
	localScale_ : LocalScalePlug = PlugDescriptor("localScale")
	underWorldObject_ : UnderWorldObjectPlug = PlugDescriptor("underWorldObject")
	worldPositionX_ : WorldPositionXPlug = PlugDescriptor("worldPositionX")
	worldPositionY_ : WorldPositionYPlug = PlugDescriptor("worldPositionY")
	worldPositionZ_ : WorldPositionZPlug = PlugDescriptor("worldPositionZ")
	worldPosition_ : WorldPositionPlug = PlugDescriptor("worldPosition")

	# node attributes

	typeName = "locator"
	apiTypeInt = 281
	apiTypeStr = "kLocator"
	typeIdInt = 1280262996
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["localPositionX", "localPositionY", "localPositionZ", "localPosition", "localScaleX", "localScaleY", "localScaleZ", "localScale", "underWorldObject", "worldPositionX", "worldPositionY", "worldPositionZ", "worldPosition"]
	nodeLeafPlugs = ["localPosition", "localScale", "underWorldObject", "worldPosition"]
	pass

