

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
	node : Facade = None
	pass
class ConnectionPlug(Plug):
	node : Facade = None
	pass
class KeyWordsPlug(Plug):
	node : Facade = None
	pass
class SharedLibNamePlug(Plug):
	node : Facade = None
	pass
class UiNamePlug(Plug):
	node : Facade = None
	pass
class UiScriptPlug(Plug):
	node : Facade = None
	pass
class UniqueIDPlug(Plug):
	node : Facade = None
	pass
# endregion


# define node class
class Facade(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	connection_ : ConnectionPlug = PlugDescriptor("connection")
	keyWords_ : KeyWordsPlug = PlugDescriptor("keyWords")
	sharedLibName_ : SharedLibNamePlug = PlugDescriptor("sharedLibName")
	uiName_ : UiNamePlug = PlugDescriptor("uiName")
	uiScript_ : UiScriptPlug = PlugDescriptor("uiScript")
	uniqueID_ : UniqueIDPlug = PlugDescriptor("uniqueID")

	# node attributes

	typeName = "facade"
	apiTypeInt = 974
	apiTypeStr = "kFacade"
	typeIdInt = 1145455438
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "connection", "keyWords", "sharedLibName", "uiName", "uiScript", "uniqueID"]
	nodeLeafPlugs = ["binMembership", "connection", "keyWords", "sharedLibName", "uiName", "uiScript", "uniqueID"]
	pass

