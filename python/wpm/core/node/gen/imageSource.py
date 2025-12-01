

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
	node : ImageSource = None
	pass
class FileHasAlphaPlug(Plug):
	node : ImageSource = None
	pass
class OutAlphaPlug(Plug):
	node : ImageSource = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : ImageSource = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : ImageSource = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : ImageSource = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : ImageSource = None
	pass
class OutSizeXPlug(Plug):
	parent : OutSizePlug = PlugDescriptor("outSize")
	node : ImageSource = None
	pass
class OutSizeYPlug(Plug):
	parent : OutSizePlug = PlugDescriptor("outSize")
	node : ImageSource = None
	pass
class OutSizePlug(Plug):
	outSizeX_ : OutSizeXPlug = PlugDescriptor("outSizeX")
	osx_ : OutSizeXPlug = PlugDescriptor("outSizeX")
	outSizeY_ : OutSizeYPlug = PlugDescriptor("outSizeY")
	osy_ : OutSizeYPlug = PlugDescriptor("outSizeY")
	node : ImageSource = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : ImageSource = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : ImageSource = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : ImageSource = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : ImageSource = None
	pass
# endregion


# define node class
class ImageSource(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	fileHasAlpha_ : FileHasAlphaPlug = PlugDescriptor("fileHasAlpha")
	outAlpha_ : OutAlphaPlug = PlugDescriptor("outAlpha")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	outSizeX_ : OutSizeXPlug = PlugDescriptor("outSizeX")
	outSizeY_ : OutSizeYPlug = PlugDescriptor("outSizeY")
	outSize_ : OutSizePlug = PlugDescriptor("outSize")
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	outTransparency_ : OutTransparencyPlug = PlugDescriptor("outTransparency")

	# node attributes

	typeName = "imageSource"
	apiTypeInt = 791
	apiTypeStr = "kImageSource"
	typeIdInt = 1380862291
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "fileHasAlpha", "outAlpha", "outColorB", "outColorG", "outColorR", "outColor", "outSizeX", "outSizeY", "outSize", "outTransparencyB", "outTransparencyG", "outTransparencyR", "outTransparency"]
	nodeLeafPlugs = ["binMembership", "fileHasAlpha", "outAlpha", "outColor", "outSize", "outTransparency"]
	pass

