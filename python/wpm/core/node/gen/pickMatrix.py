

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
	node : PickMatrix = None
	pass
class InputMatrixPlug(Plug):
	node : PickMatrix = None
	pass
class OutputMatrixPlug(Plug):
	node : PickMatrix = None
	pass
class UseRotatePlug(Plug):
	node : PickMatrix = None
	pass
class UseScalePlug(Plug):
	node : PickMatrix = None
	pass
class UseShearPlug(Plug):
	node : PickMatrix = None
	pass
class UseTranslatePlug(Plug):
	node : PickMatrix = None
	pass
# endregion


# define node class
class PickMatrix(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inputMatrix_ : InputMatrixPlug = PlugDescriptor("inputMatrix")
	outputMatrix_ : OutputMatrixPlug = PlugDescriptor("outputMatrix")
	useRotate_ : UseRotatePlug = PlugDescriptor("useRotate")
	useScale_ : UseScalePlug = PlugDescriptor("useScale")
	useShear_ : UseShearPlug = PlugDescriptor("useShear")
	useTranslate_ : UseTranslatePlug = PlugDescriptor("useTranslate")

	# node attributes

	typeName = "pickMatrix"
	apiTypeInt = 1138
	apiTypeStr = "kPickMatrix"
	typeIdInt = 1347240276
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "inputMatrix", "outputMatrix", "useRotate", "useScale", "useShear", "useTranslate"]
	nodeLeafPlugs = ["binMembership", "inputMatrix", "outputMatrix", "useRotate", "useScale", "useShear", "useTranslate"]
	pass

