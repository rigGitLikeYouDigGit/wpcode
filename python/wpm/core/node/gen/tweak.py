

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	GeometryFilter = Catalogue.GeometryFilter
else:
	from .. import retriever
	GeometryFilter = retriever.getNodeCls("GeometryFilter")
	assert GeometryFilter

# add node doc



# region plug type defs
class XValuePlug(Plug):
	parent : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : Tweak = None
	pass
class YValuePlug(Plug):
	parent : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : Tweak = None
	pass
class ZValuePlug(Plug):
	parent : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : Tweak = None
	pass
class ControlPointsPlug(Plug):
	parent : PlistPlug = PlugDescriptor("plist")
	xValue_ : XValuePlug = PlugDescriptor("xValue")
	xv_ : XValuePlug = PlugDescriptor("xValue")
	yValue_ : YValuePlug = PlugDescriptor("yValue")
	yv_ : YValuePlug = PlugDescriptor("yValue")
	zValue_ : ZValuePlug = PlugDescriptor("zValue")
	zv_ : ZValuePlug = PlugDescriptor("zValue")
	node : Tweak = None
	pass
class PlistPlug(Plug):
	controlPoints_ : ControlPointsPlug = PlugDescriptor("controlPoints")
	cp_ : ControlPointsPlug = PlugDescriptor("controlPoints")
	node : Tweak = None
	pass
class RelativeTweakPlug(Plug):
	node : Tweak = None
	pass
class XVertexPlug(Plug):
	parent : VertexPlug = PlugDescriptor("vertex")
	node : Tweak = None
	pass
class YVertexPlug(Plug):
	parent : VertexPlug = PlugDescriptor("vertex")
	node : Tweak = None
	pass
class ZVertexPlug(Plug):
	parent : VertexPlug = PlugDescriptor("vertex")
	node : Tweak = None
	pass
class VertexPlug(Plug):
	parent : VlistPlug = PlugDescriptor("vlist")
	xVertex_ : XVertexPlug = PlugDescriptor("xVertex")
	vx_ : XVertexPlug = PlugDescriptor("xVertex")
	yVertex_ : YVertexPlug = PlugDescriptor("yVertex")
	vy_ : YVertexPlug = PlugDescriptor("yVertex")
	zVertex_ : ZVertexPlug = PlugDescriptor("zVertex")
	vz_ : ZVertexPlug = PlugDescriptor("zVertex")
	node : Tweak = None
	pass
class VlistPlug(Plug):
	vertex_ : VertexPlug = PlugDescriptor("vertex")
	vt_ : VertexPlug = PlugDescriptor("vertex")
	node : Tweak = None
	pass
# endregion


# define node class
class Tweak(GeometryFilter):
	xValue_ : XValuePlug = PlugDescriptor("xValue")
	yValue_ : YValuePlug = PlugDescriptor("yValue")
	zValue_ : ZValuePlug = PlugDescriptor("zValue")
	controlPoints_ : ControlPointsPlug = PlugDescriptor("controlPoints")
	plist_ : PlistPlug = PlugDescriptor("plist")
	relativeTweak_ : RelativeTweakPlug = PlugDescriptor("relativeTweak")
	xVertex_ : XVertexPlug = PlugDescriptor("xVertex")
	yVertex_ : YVertexPlug = PlugDescriptor("yVertex")
	zVertex_ : ZVertexPlug = PlugDescriptor("zVertex")
	vertex_ : VertexPlug = PlugDescriptor("vertex")
	vlist_ : VlistPlug = PlugDescriptor("vlist")

	# node attributes

	typeName = "tweak"
	apiTypeInt = 345
	apiTypeStr = "kTweak"
	typeIdInt = 1179471956
	MFnCls = om.MFnGeometryFilter
	nodeLeafClassAttrs = ["xValue", "yValue", "zValue", "controlPoints", "plist", "relativeTweak", "xVertex", "yVertex", "zVertex", "vertex", "vlist"]
	nodeLeafPlugs = ["plist", "relativeTweak", "vlist"]
	pass

