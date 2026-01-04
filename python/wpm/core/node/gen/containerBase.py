

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : ContainerBase = None
	pass
class BlackBoxPlug(Plug):
	node : ContainerBase = None
	pass
class BorderConnectionsPlug(Plug):
	node : ContainerBase = None
	pass
class ContainerTypePlug(Plug):
	node : ContainerBase = None
	pass
class CreationDatePlug(Plug):
	node : ContainerBase = None
	pass
class CreatorPlug(Plug):
	node : ContainerBase = None
	pass
class CustomTreatmentPlug(Plug):
	node : ContainerBase = None
	pass
class HyperLayoutPlug(Plug):
	node : ContainerBase = None
	pass
class IconNamePlug(Plug):
	node : ContainerBase = None
	pass
class IsCollapsedPlug(Plug):
	node : ContainerBase = None
	pass
class IsHierarchicalConnectionPlug(Plug):
	node : ContainerBase = None
	pass
class IsHierarchicalNodePlug(Plug):
	parent : PublishedNodeInfoPlug = PlugDescriptor("publishedNodeInfo")
	node : ContainerBase = None
	pass
class PublishedNodePlug(Plug):
	parent : PublishedNodeInfoPlug = PlugDescriptor("publishedNodeInfo")
	node : ContainerBase = None
	pass
class PublishedNodeTypePlug(Plug):
	parent : PublishedNodeInfoPlug = PlugDescriptor("publishedNodeInfo")
	node : ContainerBase = None
	pass
class PublishedNodeInfoPlug(Plug):
	isHierarchicalNode_ : IsHierarchicalNodePlug = PlugDescriptor("isHierarchicalNode")
	ihn_ : IsHierarchicalNodePlug = PlugDescriptor("isHierarchicalNode")
	publishedNode_ : PublishedNodePlug = PlugDescriptor("publishedNode")
	pnod_ : PublishedNodePlug = PlugDescriptor("publishedNode")
	publishedNodeType_ : PublishedNodeTypePlug = PlugDescriptor("publishedNodeType")
	pntp_ : PublishedNodeTypePlug = PlugDescriptor("publishedNodeType")
	node : ContainerBase = None
	pass
class RmbCommandPlug(Plug):
	node : ContainerBase = None
	pass
class TemplateNamePlug(Plug):
	node : ContainerBase = None
	pass
class TemplatePathPlug(Plug):
	node : ContainerBase = None
	pass
class TemplateVersionPlug(Plug):
	node : ContainerBase = None
	pass
class UiTreatmentPlug(Plug):
	node : ContainerBase = None
	pass
class ViewModePlug(Plug):
	node : ContainerBase = None
	pass
class ViewNamePlug(Plug):
	node : ContainerBase = None
	pass
# endregion


# define node class
class ContainerBase(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	blackBox_ : BlackBoxPlug = PlugDescriptor("blackBox")
	borderConnections_ : BorderConnectionsPlug = PlugDescriptor("borderConnections")
	containerType_ : ContainerTypePlug = PlugDescriptor("containerType")
	creationDate_ : CreationDatePlug = PlugDescriptor("creationDate")
	creator_ : CreatorPlug = PlugDescriptor("creator")
	customTreatment_ : CustomTreatmentPlug = PlugDescriptor("customTreatment")
	hyperLayout_ : HyperLayoutPlug = PlugDescriptor("hyperLayout")
	iconName_ : IconNamePlug = PlugDescriptor("iconName")
	isCollapsed_ : IsCollapsedPlug = PlugDescriptor("isCollapsed")
	isHierarchicalConnection_ : IsHierarchicalConnectionPlug = PlugDescriptor("isHierarchicalConnection")
	isHierarchicalNode_ : IsHierarchicalNodePlug = PlugDescriptor("isHierarchicalNode")
	publishedNode_ : PublishedNodePlug = PlugDescriptor("publishedNode")
	publishedNodeType_ : PublishedNodeTypePlug = PlugDescriptor("publishedNodeType")
	publishedNodeInfo_ : PublishedNodeInfoPlug = PlugDescriptor("publishedNodeInfo")
	rmbCommand_ : RmbCommandPlug = PlugDescriptor("rmbCommand")
	templateName_ : TemplateNamePlug = PlugDescriptor("templateName")
	templatePath_ : TemplatePathPlug = PlugDescriptor("templatePath")
	templateVersion_ : TemplateVersionPlug = PlugDescriptor("templateVersion")
	uiTreatment_ : UiTreatmentPlug = PlugDescriptor("uiTreatment")
	viewMode_ : ViewModePlug = PlugDescriptor("viewMode")
	viewName_ : ViewNamePlug = PlugDescriptor("viewName")

	# node attributes

	typeName = "containerBase"
	apiTypeInt = 1068
	apiTypeStr = "kContainerBase"
	typeIdInt = 1129267777
	MFnCls = om.MFnContainerNode
	nodeLeafClassAttrs = ["binMembership", "blackBox", "borderConnections", "containerType", "creationDate", "creator", "customTreatment", "hyperLayout", "iconName", "isCollapsed", "isHierarchicalConnection", "isHierarchicalNode", "publishedNode", "publishedNodeType", "publishedNodeInfo", "rmbCommand", "templateName", "templatePath", "templateVersion", "uiTreatment", "viewMode", "viewName"]
	nodeLeafPlugs = ["binMembership", "blackBox", "borderConnections", "containerType", "creationDate", "creator", "customTreatment", "hyperLayout", "iconName", "isCollapsed", "isHierarchicalConnection", "publishedNodeInfo", "rmbCommand", "templateName", "templatePath", "templateVersion", "uiTreatment", "viewMode", "viewName"]
	pass

