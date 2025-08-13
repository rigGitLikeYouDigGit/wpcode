

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
_BASE_ = retriever.getNodeCls("_BASE_")
assert _BASE_
if T.TYPE_CHECKING:
	from .. import _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : BlendColors = None
	pass
class BlenderPlug(Plug):
	node : BlendColors = None
	pass
class Color1BPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : BlendColors = None
	pass
class Color1GPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : BlendColors = None
	pass
class Color1RPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : BlendColors = None
	pass
class Color1Plug(Plug):
	color1B_ : Color1BPlug = PlugDescriptor("color1B")
	c1b_ : Color1BPlug = PlugDescriptor("color1B")
	color1G_ : Color1GPlug = PlugDescriptor("color1G")
	c1g_ : Color1GPlug = PlugDescriptor("color1G")
	color1R_ : Color1RPlug = PlugDescriptor("color1R")
	c1r_ : Color1RPlug = PlugDescriptor("color1R")
	node : BlendColors = None
	pass
class Color2BPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : BlendColors = None
	pass
class Color2GPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : BlendColors = None
	pass
class Color2RPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : BlendColors = None
	pass
class Color2Plug(Plug):
	color2B_ : Color2BPlug = PlugDescriptor("color2B")
	c2b_ : Color2BPlug = PlugDescriptor("color2B")
	color2G_ : Color2GPlug = PlugDescriptor("color2G")
	c2g_ : Color2GPlug = PlugDescriptor("color2G")
	color2R_ : Color2RPlug = PlugDescriptor("color2R")
	c2r_ : Color2RPlug = PlugDescriptor("color2R")
	node : BlendColors = None
	pass
class OutputBPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : BlendColors = None
	pass
class OutputGPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : BlendColors = None
	pass
class OutputRPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : BlendColors = None
	pass
class OutputPlug(Plug):
	outputB_ : OutputBPlug = PlugDescriptor("outputB")
	opb_ : OutputBPlug = PlugDescriptor("outputB")
	outputG_ : OutputGPlug = PlugDescriptor("outputG")
	opg_ : OutputGPlug = PlugDescriptor("outputG")
	outputR_ : OutputRPlug = PlugDescriptor("outputR")
	opr_ : OutputRPlug = PlugDescriptor("outputR")
	node : BlendColors = None
	pass
class RenderPassModePlug(Plug):
	node : BlendColors = None
	pass
# endregion


# define node class
class BlendColors(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	blender_ : BlenderPlug = PlugDescriptor("blender")
	color1B_ : Color1BPlug = PlugDescriptor("color1B")
	color1G_ : Color1GPlug = PlugDescriptor("color1G")
	color1R_ : Color1RPlug = PlugDescriptor("color1R")
	color1_ : Color1Plug = PlugDescriptor("color1")
	color2B_ : Color2BPlug = PlugDescriptor("color2B")
	color2G_ : Color2GPlug = PlugDescriptor("color2G")
	color2R_ : Color2RPlug = PlugDescriptor("color2R")
	color2_ : Color2Plug = PlugDescriptor("color2")
	outputB_ : OutputBPlug = PlugDescriptor("outputB")
	outputG_ : OutputGPlug = PlugDescriptor("outputG")
	outputR_ : OutputRPlug = PlugDescriptor("outputR")
	output_ : OutputPlug = PlugDescriptor("output")
	renderPassMode_ : RenderPassModePlug = PlugDescriptor("renderPassMode")

	# node attributes

	typeName = "blendColors"
	apiTypeInt = 31
	apiTypeStr = "kBlendColors"
	typeIdInt = 1380076594
	MFnCls = om.MFnDependencyNode
	pass

