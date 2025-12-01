

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
class ActiveProxyPlug(Plug):
	node : ProxyManager = None
	pass
class BinMembershipPlug(Plug):
	node : ProxyManager = None
	pass
class ProxyListPlug(Plug):
	node : ProxyManager = None
	pass
class SharedEditsOwnerPlug(Plug):
	node : ProxyManager = None
	pass
# endregion


# define node class
class ProxyManager(_BASE_):
	activeProxy_ : ActiveProxyPlug = PlugDescriptor("activeProxy")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	proxyList_ : ProxyListPlug = PlugDescriptor("proxyList")
	sharedEditsOwner_ : SharedEditsOwnerPlug = PlugDescriptor("sharedEditsOwner")

	# node attributes

	typeName = "proxyManager"
	apiTypeInt = 966
	apiTypeStr = "kProxyManager"
	typeIdInt = 1347964231
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["activeProxy", "binMembership", "proxyList", "sharedEditsOwner"]
	nodeLeafPlugs = ["activeProxy", "binMembership", "proxyList", "sharedEditsOwner"]
	pass

