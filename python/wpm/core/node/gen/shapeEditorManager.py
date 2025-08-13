

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
	node : ShapeEditorManager = None
	pass
class ChildIndicesPlug(Plug):
	parent : BlendShapeDirectoryPlug = PlugDescriptor("blendShapeDirectory")
	node : ShapeEditorManager = None
	pass
class DirectoryNamePlug(Plug):
	parent : BlendShapeDirectoryPlug = PlugDescriptor("blendShapeDirectory")
	node : ShapeEditorManager = None
	pass
class DirectoryParentVisibilityPlug(Plug):
	parent : BlendShapeDirectoryPlug = PlugDescriptor("blendShapeDirectory")
	node : ShapeEditorManager = None
	pass
class DirectoryVisibilityPlug(Plug):
	parent : BlendShapeDirectoryPlug = PlugDescriptor("blendShapeDirectory")
	node : ShapeEditorManager = None
	pass
class ParentIndexPlug(Plug):
	parent : BlendShapeDirectoryPlug = PlugDescriptor("blendShapeDirectory")
	node : ShapeEditorManager = None
	pass
class BlendShapeDirectoryPlug(Plug):
	childIndices_ : ChildIndicesPlug = PlugDescriptor("childIndices")
	bscd_ : ChildIndicesPlug = PlugDescriptor("childIndices")
	directoryName_ : DirectoryNamePlug = PlugDescriptor("directoryName")
	bsdn_ : DirectoryNamePlug = PlugDescriptor("directoryName")
	directoryParentVisibility_ : DirectoryParentVisibilityPlug = PlugDescriptor("directoryParentVisibility")
	bdpv_ : DirectoryParentVisibilityPlug = PlugDescriptor("directoryParentVisibility")
	directoryVisibility_ : DirectoryVisibilityPlug = PlugDescriptor("directoryVisibility")
	bsdv_ : DirectoryVisibilityPlug = PlugDescriptor("directoryVisibility")
	parentIndex_ : ParentIndexPlug = PlugDescriptor("parentIndex")
	bspi_ : ParentIndexPlug = PlugDescriptor("parentIndex")
	node : ShapeEditorManager = None
	pass
class BlendShapeParentPlug(Plug):
	node : ShapeEditorManager = None
	pass
class FilterStringPlug(Plug):
	node : ShapeEditorManager = None
	pass
class OutBlendShapeVisibilityPlug(Plug):
	node : ShapeEditorManager = None
	pass
# endregion


# define node class
class ShapeEditorManager(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	childIndices_ : ChildIndicesPlug = PlugDescriptor("childIndices")
	directoryName_ : DirectoryNamePlug = PlugDescriptor("directoryName")
	directoryParentVisibility_ : DirectoryParentVisibilityPlug = PlugDescriptor("directoryParentVisibility")
	directoryVisibility_ : DirectoryVisibilityPlug = PlugDescriptor("directoryVisibility")
	parentIndex_ : ParentIndexPlug = PlugDescriptor("parentIndex")
	blendShapeDirectory_ : BlendShapeDirectoryPlug = PlugDescriptor("blendShapeDirectory")
	blendShapeParent_ : BlendShapeParentPlug = PlugDescriptor("blendShapeParent")
	filterString_ : FilterStringPlug = PlugDescriptor("filterString")
	outBlendShapeVisibility_ : OutBlendShapeVisibilityPlug = PlugDescriptor("outBlendShapeVisibility")

	# node attributes

	typeName = "shapeEditorManager"
	apiTypeInt = 1125
	apiTypeStr = "kShapeEditorManager"
	typeIdInt = 1396985164
	MFnCls = om.MFnDependencyNode
	pass

