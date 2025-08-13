

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
class AddAttrListPlug(Plug):
	node : Reference = None
	pass
class AssociatedNodePlug(Plug):
	node : Reference = None
	pass
class BinMembershipPlug(Plug):
	node : Reference = None
	pass
class BrokenConnectionListPlug(Plug):
	node : Reference = None
	pass
class ConnectionPlug(Plug):
	parent : ConnectionListPlug = PlugDescriptor("connectionList")
	node : Reference = None
	pass
class ConnectionAttrPlug(Plug):
	parent : ConnectionListPlug = PlugDescriptor("connectionList")
	node : Reference = None
	pass
class ConnectionListPlug(Plug):
	connection_ : ConnectionPlug = PlugDescriptor("connection")
	c_ : ConnectionPlug = PlugDescriptor("connection")
	connectionAttr_ : ConnectionAttrPlug = PlugDescriptor("connectionAttr")
	ca_ : ConnectionAttrPlug = PlugDescriptor("connectionAttr")
	node : Reference = None
	pass
class DeleteAttrListPlug(Plug):
	node : Reference = None
	pass
class EditsPlug(Plug):
	node : Reference = None
	pass
class FileNamesPlug(Plug):
	node : Reference = None
	pass
class FosterParentPlug(Plug):
	node : Reference = None
	pass
class FosterSiblingsPlug(Plug):
	node : Reference = None
	pass
class LockedPlug(Plug):
	node : Reference = None
	pass
class MultiParentPlug(Plug):
	parent : MultiParentListPlug = PlugDescriptor("multiParentList")
	node : Reference = None
	pass
class MultiParentListPlug(Plug):
	multiParent_ : MultiParentPlug = PlugDescriptor("multiParent")
	mp_ : MultiParentPlug = PlugDescriptor("multiParent")
	node : Reference = None
	pass
class ParentListPlug(Plug):
	node : Reference = None
	pass
class PlaceHolderListPlug(Plug):
	node : Reference = None
	pass
class PlaceHolderNamespacePlug(Plug):
	node : Reference = None
	pass
class ProxyMsgPlug(Plug):
	node : Reference = None
	pass
class ProxyTagPlug(Plug):
	node : Reference = None
	pass
class SetAttrListPlug(Plug):
	node : Reference = None
	pass
class SharedReferencePlug(Plug):
	node : Reference = None
	pass
class UnknownReferencePlug(Plug):
	node : Reference = None
	pass
# endregion


# define node class
class Reference(_BASE_):
	addAttrList_ : AddAttrListPlug = PlugDescriptor("addAttrList")
	associatedNode_ : AssociatedNodePlug = PlugDescriptor("associatedNode")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	brokenConnectionList_ : BrokenConnectionListPlug = PlugDescriptor("brokenConnectionList")
	connection_ : ConnectionPlug = PlugDescriptor("connection")
	connectionAttr_ : ConnectionAttrPlug = PlugDescriptor("connectionAttr")
	connectionList_ : ConnectionListPlug = PlugDescriptor("connectionList")
	deleteAttrList_ : DeleteAttrListPlug = PlugDescriptor("deleteAttrList")
	edits_ : EditsPlug = PlugDescriptor("edits")
	fileNames_ : FileNamesPlug = PlugDescriptor("fileNames")
	fosterParent_ : FosterParentPlug = PlugDescriptor("fosterParent")
	fosterSiblings_ : FosterSiblingsPlug = PlugDescriptor("fosterSiblings")
	locked_ : LockedPlug = PlugDescriptor("locked")
	multiParent_ : MultiParentPlug = PlugDescriptor("multiParent")
	multiParentList_ : MultiParentListPlug = PlugDescriptor("multiParentList")
	parentList_ : ParentListPlug = PlugDescriptor("parentList")
	placeHolderList_ : PlaceHolderListPlug = PlugDescriptor("placeHolderList")
	placeHolderNamespace_ : PlaceHolderNamespacePlug = PlugDescriptor("placeHolderNamespace")
	proxyMsg_ : ProxyMsgPlug = PlugDescriptor("proxyMsg")
	proxyTag_ : ProxyTagPlug = PlugDescriptor("proxyTag")
	setAttrList_ : SetAttrListPlug = PlugDescriptor("setAttrList")
	sharedReference_ : SharedReferencePlug = PlugDescriptor("sharedReference")
	unknownReference_ : UnknownReferencePlug = PlugDescriptor("unknownReference")

	# node attributes

	typeName = "reference"
	apiTypeInt = 755
	apiTypeStr = "kReference"
	typeIdInt = 1380271694
	MFnCls = om.MFnReference
	pass

