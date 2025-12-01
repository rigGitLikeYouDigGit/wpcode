

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ImageSource = retriever.getNodeCls("ImageSource")
assert ImageSource
if T.TYPE_CHECKING:
	from .. import ImageSource

# add node doc



# region plug type defs
class BackupAttributesPlug(Plug):
	parent : BackupPlug = PlugDescriptor("backup")
	node : RenderPass = None
	pass
class BackupDataPlug(Plug):
	parent : BackupPlug = PlugDescriptor("backup")
	node : RenderPass = None
	pass
class BackupPlug(Plug):
	backupAttributes_ : BackupAttributesPlug = PlugDescriptor("backupAttributes")
	ba_ : BackupAttributesPlug = PlugDescriptor("backupAttributes")
	backupData_ : BackupDataPlug = PlugDescriptor("backupData")
	bd_ : BackupDataPlug = PlugDescriptor("backupData")
	node : RenderPass = None
	pass
class ColorProfilePlug(Plug):
	node : RenderPass = None
	pass
class FilteringPlug(Plug):
	node : RenderPass = None
	pass
class FrameBufferTypePlug(Plug):
	node : RenderPass = None
	pass
class NumChannelsPlug(Plug):
	node : RenderPass = None
	pass
class OwnerPlug(Plug):
	node : RenderPass = None
	pass
class PassGroupNamePlug(Plug):
	node : RenderPass = None
	pass
class PassIDPlug(Plug):
	node : RenderPass = None
	pass
class RenderablePlug(Plug):
	node : RenderPass = None
	pass
# endregion


# define node class
class RenderPass(ImageSource):
	backupAttributes_ : BackupAttributesPlug = PlugDescriptor("backupAttributes")
	backupData_ : BackupDataPlug = PlugDescriptor("backupData")
	backup_ : BackupPlug = PlugDescriptor("backup")
	colorProfile_ : ColorProfilePlug = PlugDescriptor("colorProfile")
	filtering_ : FilteringPlug = PlugDescriptor("filtering")
	frameBufferType_ : FrameBufferTypePlug = PlugDescriptor("frameBufferType")
	numChannels_ : NumChannelsPlug = PlugDescriptor("numChannels")
	owner_ : OwnerPlug = PlugDescriptor("owner")
	passGroupName_ : PassGroupNamePlug = PlugDescriptor("passGroupName")
	passID_ : PassIDPlug = PlugDescriptor("passID")
	renderable_ : RenderablePlug = PlugDescriptor("renderable")

	# node attributes

	typeName = "renderPass"
	apiTypeInt = 783
	apiTypeStr = "kRenderPass"
	typeIdInt = 1380864083
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["backupAttributes", "backupData", "backup", "colorProfile", "filtering", "frameBufferType", "numChannels", "owner", "passGroupName", "passID", "renderable"]
	nodeLeafPlugs = ["backup", "colorProfile", "filtering", "frameBufferType", "numChannels", "owner", "passGroupName", "passID", "renderable"]
	pass

