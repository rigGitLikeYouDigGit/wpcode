

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
	node : PoseInterpolatorManager = None
	pass
class ChildIndicesPlug(Plug):
	parent : PoseInterpolatorDirectoryPlug = PlugDescriptor("poseInterpolatorDirectory")
	node : PoseInterpolatorManager = None
	pass
class DirectoryNamePlug(Plug):
	parent : PoseInterpolatorDirectoryPlug = PlugDescriptor("poseInterpolatorDirectory")
	node : PoseInterpolatorManager = None
	pass
class ParentIndexPlug(Plug):
	parent : PoseInterpolatorDirectoryPlug = PlugDescriptor("poseInterpolatorDirectory")
	node : PoseInterpolatorManager = None
	pass
class PoseInterpolatorDirectoryPlug(Plug):
	childIndices_ : ChildIndicesPlug = PlugDescriptor("childIndices")
	tpcd_ : ChildIndicesPlug = PlugDescriptor("childIndices")
	directoryName_ : DirectoryNamePlug = PlugDescriptor("directoryName")
	tpdn_ : DirectoryNamePlug = PlugDescriptor("directoryName")
	parentIndex_ : ParentIndexPlug = PlugDescriptor("parentIndex")
	tppi_ : ParentIndexPlug = PlugDescriptor("parentIndex")
	node : PoseInterpolatorManager = None
	pass
class PoseInterpolatorParentPlug(Plug):
	node : PoseInterpolatorManager = None
	pass
# endregion


# define node class
class PoseInterpolatorManager(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	childIndices_ : ChildIndicesPlug = PlugDescriptor("childIndices")
	directoryName_ : DirectoryNamePlug = PlugDescriptor("directoryName")
	parentIndex_ : ParentIndexPlug = PlugDescriptor("parentIndex")
	poseInterpolatorDirectory_ : PoseInterpolatorDirectoryPlug = PlugDescriptor("poseInterpolatorDirectory")
	poseInterpolatorParent_ : PoseInterpolatorParentPlug = PlugDescriptor("poseInterpolatorParent")

	# node attributes

	typeName = "poseInterpolatorManager"
	apiTypeInt = 1127
	apiTypeStr = "kPoseInterpolatorManager"
	typeIdInt = 1347634253
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "childIndices", "directoryName", "parentIndex", "poseInterpolatorDirectory", "poseInterpolatorParent"]
	nodeLeafPlugs = ["binMembership", "poseInterpolatorDirectory", "poseInterpolatorParent"]
	pass

