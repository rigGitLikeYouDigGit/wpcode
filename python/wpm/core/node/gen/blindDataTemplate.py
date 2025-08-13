

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
class BdPresetAttrPlug(Plug):
	parent : BdPresetElementsPlug = PlugDescriptor("bdPresetElements")
	node : BlindDataTemplate = None
	pass
class BdUserInfoNamePlug(Plug):
	parent : BdUserInfoPlug = PlugDescriptor("bdUserInfo")
	node : BlindDataTemplate = None
	pass
class BdUserInfoValuePlug(Plug):
	parent : BdUserInfoPlug = PlugDescriptor("bdUserInfo")
	node : BlindDataTemplate = None
	pass
class BdUserInfoPlug(Plug):
	bdUserInfoName_ : BdUserInfoNamePlug = PlugDescriptor("bdUserInfoName")
	bdun_ : BdUserInfoNamePlug = PlugDescriptor("bdUserInfoName")
	bdUserInfoValue_ : BdUserInfoValuePlug = PlugDescriptor("bdUserInfoValue")
	bduv_ : BdUserInfoValuePlug = PlugDescriptor("bdUserInfoValue")
	node : BlindDataTemplate = None
	pass
class BinMembershipPlug(Plug):
	node : BlindDataTemplate = None
	pass
class BdPresetValuePlug(Plug):
	parent : BdPresetElementsPlug = PlugDescriptor("bdPresetElements")
	node : BlindDataTemplate = None
	pass
class BdPresetElementsPlug(Plug):
	parent : BlindDataPresetsPlug = PlugDescriptor("blindDataPresets")
	bdPresetAttr_ : BdPresetAttrPlug = PlugDescriptor("bdPresetAttr")
	bdpa_ : BdPresetAttrPlug = PlugDescriptor("bdPresetAttr")
	bdPresetValue_ : BdPresetValuePlug = PlugDescriptor("bdPresetValue")
	bdpv_ : BdPresetValuePlug = PlugDescriptor("bdPresetValue")
	node : BlindDataTemplate = None
	pass
class BdPresetNamePlug(Plug):
	parent : BlindDataPresetsPlug = PlugDescriptor("blindDataPresets")
	node : BlindDataTemplate = None
	pass
class BlindDataPresetsPlug(Plug):
	bdPresetElements_ : BdPresetElementsPlug = PlugDescriptor("bdPresetElements")
	bdpe_ : BdPresetElementsPlug = PlugDescriptor("bdPresetElements")
	bdPresetName_ : BdPresetNamePlug = PlugDescriptor("bdPresetName")
	bdpn_ : BdPresetNamePlug = PlugDescriptor("bdPresetName")
	node : BlindDataTemplate = None
	pass
class TypeIdPlug(Plug):
	node : BlindDataTemplate = None
	pass
# endregion


# define node class
class BlindDataTemplate(_BASE_):
	bdPresetAttr_ : BdPresetAttrPlug = PlugDescriptor("bdPresetAttr")
	bdUserInfoName_ : BdUserInfoNamePlug = PlugDescriptor("bdUserInfoName")
	bdUserInfoValue_ : BdUserInfoValuePlug = PlugDescriptor("bdUserInfoValue")
	bdUserInfo_ : BdUserInfoPlug = PlugDescriptor("bdUserInfo")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	bdPresetValue_ : BdPresetValuePlug = PlugDescriptor("bdPresetValue")
	bdPresetElements_ : BdPresetElementsPlug = PlugDescriptor("bdPresetElements")
	bdPresetName_ : BdPresetNamePlug = PlugDescriptor("bdPresetName")
	blindDataPresets_ : BlindDataPresetsPlug = PlugDescriptor("blindDataPresets")
	typeId_ : TypeIdPlug = PlugDescriptor("typeId")

	# node attributes

	typeName = "blindDataTemplate"
	apiTypeInt = 757
	apiTypeStr = "kBlindDataTemplate"
	typeIdInt = 1112294484
	MFnCls = om.MFnDependencyNode
	pass

