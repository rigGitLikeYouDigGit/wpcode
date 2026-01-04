

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ContainerBase = Catalogue.ContainerBase
else:
	from .. import retriever
	ContainerBase = retriever.getNodeCls("ContainerBase")
	assert ContainerBase

# add node doc



# region plug type defs
class CanBeChildPlug(Plug):
	node : Container = None
	pass
class CanBeParentPlug(Plug):
	node : Container = None
	pass
class IsHierarchicalChildPlug(Plug):
	node : Container = None
	pass
class IsHierarchicalParentPlug(Plug):
	node : Container = None
	pass
class IsHierarchicalRootPlug(Plug):
	node : Container = None
	pass
class RootTransformPlug(Plug):
	node : Container = None
	pass
# endregion


# define node class
class Container(ContainerBase):
	canBeChild_ : CanBeChildPlug = PlugDescriptor("canBeChild")
	canBeParent_ : CanBeParentPlug = PlugDescriptor("canBeParent")
	isHierarchicalChild_ : IsHierarchicalChildPlug = PlugDescriptor("isHierarchicalChild")
	isHierarchicalParent_ : IsHierarchicalParentPlug = PlugDescriptor("isHierarchicalParent")
	isHierarchicalRoot_ : IsHierarchicalRootPlug = PlugDescriptor("isHierarchicalRoot")
	rootTransform_ : RootTransformPlug = PlugDescriptor("rootTransform")

	# node attributes

	typeName = "container"
	apiTypeInt = 1013
	apiTypeStr = "kContainer"
	typeIdInt = 1129270868
	MFnCls = om.MFnContainerNode
	nodeLeafClassAttrs = ["canBeChild", "canBeParent", "isHierarchicalChild", "isHierarchicalParent", "isHierarchicalRoot", "rootTransform"]
	nodeLeafPlugs = ["canBeChild", "canBeParent", "isHierarchicalChild", "isHierarchicalParent", "isHierarchicalRoot", "rootTransform"]
	pass

