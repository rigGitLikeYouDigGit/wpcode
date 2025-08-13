

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
	node : ViewColorManager = None
	pass
class ContrastPlug(Plug):
	node : ViewColorManager = None
	pass
class ContrastPivotPlug(Plug):
	node : ViewColorManager = None
	pass
class DisplayColorProfilePlug(Plug):
	node : ViewColorManager = None
	pass
class ExposurePlug(Plug):
	node : ViewColorManager = None
	pass
class ImageColorProfilePlug(Plug):
	node : ViewColorManager = None
	pass
class LutFilePlug(Plug):
	node : ViewColorManager = None
	pass
# endregion


# define node class
class ViewColorManager(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	contrast_ : ContrastPlug = PlugDescriptor("contrast")
	contrastPivot_ : ContrastPivotPlug = PlugDescriptor("contrastPivot")
	displayColorProfile_ : DisplayColorProfilePlug = PlugDescriptor("displayColorProfile")
	exposure_ : ExposurePlug = PlugDescriptor("exposure")
	imageColorProfile_ : ImageColorProfilePlug = PlugDescriptor("imageColorProfile")
	lutFile_ : LutFilePlug = PlugDescriptor("lutFile")

	# node attributes

	typeName = "viewColorManager"
	apiTypeInt = 671
	apiTypeStr = "kViewColorManager"
	typeIdInt = 1448559437
	MFnCls = om.MFnDependencyNode
	pass

