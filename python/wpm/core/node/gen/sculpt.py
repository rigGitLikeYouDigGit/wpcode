

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
GeometryFilter = retriever.getNodeCls("GeometryFilter")
assert GeometryFilter
if T.TYPE_CHECKING:
	from .. import GeometryFilter

# add node doc



# region plug type defs
class DropoffDistancePlug(Plug):
	node : Sculpt = None
	pass
class DropoffTypePlug(Plug):
	node : Sculpt = None
	pass
class ExtendedEndPlug(Plug):
	node : Sculpt = None
	pass
class InsideModePlug(Plug):
	node : Sculpt = None
	pass
class MaximumDisplacementPlug(Plug):
	node : Sculpt = None
	pass
class ModePlug(Plug):
	node : Sculpt = None
	pass
class SculptObjectGeometryPlug(Plug):
	node : Sculpt = None
	pass
class SculptObjectMatrixPlug(Plug):
	node : Sculpt = None
	pass
class StartPosXPlug(Plug):
	parent : StartPositionPlug = PlugDescriptor("startPosition")
	node : Sculpt = None
	pass
class StartPosYPlug(Plug):
	parent : StartPositionPlug = PlugDescriptor("startPosition")
	node : Sculpt = None
	pass
class StartPosZPlug(Plug):
	parent : StartPositionPlug = PlugDescriptor("startPosition")
	node : Sculpt = None
	pass
class StartPositionPlug(Plug):
	startPosX_ : StartPosXPlug = PlugDescriptor("startPosX")
	sx_ : StartPosXPlug = PlugDescriptor("startPosX")
	startPosY_ : StartPosYPlug = PlugDescriptor("startPosY")
	sy_ : StartPosYPlug = PlugDescriptor("startPosY")
	startPosZ_ : StartPosZPlug = PlugDescriptor("startPosZ")
	sz_ : StartPosZPlug = PlugDescriptor("startPosZ")
	node : Sculpt = None
	pass
# endregion


# define node class
class Sculpt(GeometryFilter):
	dropoffDistance_ : DropoffDistancePlug = PlugDescriptor("dropoffDistance")
	dropoffType_ : DropoffTypePlug = PlugDescriptor("dropoffType")
	extendedEnd_ : ExtendedEndPlug = PlugDescriptor("extendedEnd")
	insideMode_ : InsideModePlug = PlugDescriptor("insideMode")
	maximumDisplacement_ : MaximumDisplacementPlug = PlugDescriptor("maximumDisplacement")
	mode_ : ModePlug = PlugDescriptor("mode")
	sculptObjectGeometry_ : SculptObjectGeometryPlug = PlugDescriptor("sculptObjectGeometry")
	sculptObjectMatrix_ : SculptObjectMatrixPlug = PlugDescriptor("sculptObjectMatrix")
	startPosX_ : StartPosXPlug = PlugDescriptor("startPosX")
	startPosY_ : StartPosYPlug = PlugDescriptor("startPosY")
	startPosZ_ : StartPosZPlug = PlugDescriptor("startPosZ")
	startPosition_ : StartPositionPlug = PlugDescriptor("startPosition")

	# node attributes

	typeName = "sculpt"
	apiTypeInt = 342
	apiTypeStr = "kSculpt"
	typeIdInt = 1179861840
	MFnCls = om.MFnGeometryFilter
	pass

