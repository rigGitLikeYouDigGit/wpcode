

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class BaseColorNamePlug(Plug):
	node : BlendColorSets = None
	pass
class BlendFuncPlug(Plug):
	node : BlendColorSets = None
	pass
class BlendWeightAPlug(Plug):
	node : BlendColorSets = None
	pass
class BlendWeightBPlug(Plug):
	node : BlendColorSets = None
	pass
class BlendWeightCPlug(Plug):
	node : BlendColorSets = None
	pass
class BlendWeightDPlug(Plug):
	node : BlendColorSets = None
	pass
class DstColorNamePlug(Plug):
	node : BlendColorSets = None
	pass
class SrcColorNamePlug(Plug):
	node : BlendColorSets = None
	pass
# endregion


# define node class
class BlendColorSets(PolyModifier):
	baseColorName_ : BaseColorNamePlug = PlugDescriptor("baseColorName")
	blendFunc_ : BlendFuncPlug = PlugDescriptor("blendFunc")
	blendWeightA_ : BlendWeightAPlug = PlugDescriptor("blendWeightA")
	blendWeightB_ : BlendWeightBPlug = PlugDescriptor("blendWeightB")
	blendWeightC_ : BlendWeightCPlug = PlugDescriptor("blendWeightC")
	blendWeightD_ : BlendWeightDPlug = PlugDescriptor("blendWeightD")
	dstColorName_ : DstColorNamePlug = PlugDescriptor("dstColorName")
	srcColorName_ : SrcColorNamePlug = PlugDescriptor("srcColorName")

	# node attributes

	typeName = "blendColorSets"
	typeIdInt = 1346519891
	pass

