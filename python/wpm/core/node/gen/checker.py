

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
class Color1BPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : Checker = None
	pass
class Color1GPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : Checker = None
	pass
class Color1RPlug(Plug):
	parent : Color1Plug = PlugDescriptor("color1")
	node : Checker = None
	pass
class Color1Plug(Plug):
	color1B_ : Color1BPlug = PlugDescriptor("color1B")
	c1b_ : Color1BPlug = PlugDescriptor("color1B")
	color1G_ : Color1GPlug = PlugDescriptor("color1G")
	c1g_ : Color1GPlug = PlugDescriptor("color1G")
	color1R_ : Color1RPlug = PlugDescriptor("color1R")
	c1r_ : Color1RPlug = PlugDescriptor("color1R")
	node : Checker = None
	pass
class Color2BPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : Checker = None
	pass
class Color2GPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : Checker = None
	pass
class Color2RPlug(Plug):
	parent : Color2Plug = PlugDescriptor("color2")
	node : Checker = None
	pass
class Color2Plug(Plug):
	color2B_ : Color2BPlug = PlugDescriptor("color2B")
	c2b_ : Color2BPlug = PlugDescriptor("color2B")
	color2G_ : Color2GPlug = PlugDescriptor("color2G")
	c2g_ : Color2GPlug = PlugDescriptor("color2G")
	color2R_ : Color2RPlug = PlugDescriptor("color2R")
	c2r_ : Color2RPlug = PlugDescriptor("color2R")
	node : Checker = None
	pass
class ContrastPlug(Plug):
	node : Checker = None
	pass
# endregion


# define node class
class Checker(Texture2d):
	color1B_ : Color1BPlug = PlugDescriptor("color1B")
	color1G_ : Color1GPlug = PlugDescriptor("color1G")
	color1R_ : Color1RPlug = PlugDescriptor("color1R")
	color1_ : Color1Plug = PlugDescriptor("color1")
	color2B_ : Color2BPlug = PlugDescriptor("color2B")
	color2G_ : Color2GPlug = PlugDescriptor("color2G")
	color2R_ : Color2RPlug = PlugDescriptor("color2R")
	color2_ : Color2Plug = PlugDescriptor("color2")
	contrast_ : ContrastPlug = PlugDescriptor("contrast")

	# node attributes

	typeName = "checker"
	apiTypeInt = 498
	apiTypeStr = "kChecker"
	typeIdInt = 1381253960
	MFnCls = om.MFnDependencyNode
	pass

