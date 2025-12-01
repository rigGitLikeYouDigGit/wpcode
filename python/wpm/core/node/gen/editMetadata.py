

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
	node : EditMetadata = None
	pass
class StreamEditsPlug(Plug):
	parent : ChannelEditsPlug = PlugDescriptor("channelEdits")
	error_ : ErrorPlug = PlugDescriptor("error")
	err_ : ErrorPlug = PlugDescriptor("error")
	metadataEditType_ : MetadataEditTypePlug = PlugDescriptor("metadataEditType")
	met_ : MetadataEditTypePlug = PlugDescriptor("metadataEditType")
	metadataEditValue_ : MetadataEditValuePlug = PlugDescriptor("metadataEditValue")
	mev_ : MetadataEditValuePlug = PlugDescriptor("metadataEditValue")
	metadataIndexValue_ : MetadataIndexValuePlug = PlugDescriptor("metadataIndexValue")
	miv_ : MetadataIndexValuePlug = PlugDescriptor("metadataIndexValue")
	metadataMember_ : MetadataMemberPlug = PlugDescriptor("metadataMember")
	mm_ : MetadataMemberPlug = PlugDescriptor("metadataMember")
	node : EditMetadata = None
	pass
class StreamNamePlug(Plug):
	parent : ChannelEditsPlug = PlugDescriptor("channelEdits")
	node : EditMetadata = None
	pass
class ChannelEditsPlug(Plug):
	parent : EditsPlug = PlugDescriptor("edits")
	streamEdits_ : StreamEditsPlug = PlugDescriptor("streamEdits")
	se_ : StreamEditsPlug = PlugDescriptor("streamEdits")
	streamName_ : StreamNamePlug = PlugDescriptor("streamName")
	sn_ : StreamNamePlug = PlugDescriptor("streamName")
	node : EditMetadata = None
	pass
class ChannelNamePlug(Plug):
	parent : EditsPlug = PlugDescriptor("edits")
	node : EditMetadata = None
	pass
class EditsPlug(Plug):
	channelEdits_ : ChannelEditsPlug = PlugDescriptor("channelEdits")
	ce_ : ChannelEditsPlug = PlugDescriptor("channelEdits")
	channelName_ : ChannelNamePlug = PlugDescriptor("channelName")
	cn_ : ChannelNamePlug = PlugDescriptor("channelName")
	node : EditMetadata = None
	pass
class InDataPlug(Plug):
	node : EditMetadata = None
	pass
class OutDataPlug(Plug):
	node : EditMetadata = None
	pass
class ErrorPlug(Plug):
	parent : StreamEditsPlug = PlugDescriptor("streamEdits")
	node : EditMetadata = None
	pass
class MetadataEditTypePlug(Plug):
	parent : StreamEditsPlug = PlugDescriptor("streamEdits")
	node : EditMetadata = None
	pass
class MetadataEditValuePlug(Plug):
	parent : StreamEditsPlug = PlugDescriptor("streamEdits")
	node : EditMetadata = None
	pass
class MetadataIndexValuePlug(Plug):
	parent : StreamEditsPlug = PlugDescriptor("streamEdits")
	node : EditMetadata = None
	pass
class MetadataMemberPlug(Plug):
	parent : StreamEditsPlug = PlugDescriptor("streamEdits")
	node : EditMetadata = None
	pass
# endregion


# define node class
class EditMetadata(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	streamEdits_ : StreamEditsPlug = PlugDescriptor("streamEdits")
	streamName_ : StreamNamePlug = PlugDescriptor("streamName")
	channelEdits_ : ChannelEditsPlug = PlugDescriptor("channelEdits")
	channelName_ : ChannelNamePlug = PlugDescriptor("channelName")
	edits_ : EditsPlug = PlugDescriptor("edits")
	inData_ : InDataPlug = PlugDescriptor("inData")
	outData_ : OutDataPlug = PlugDescriptor("outData")
	error_ : ErrorPlug = PlugDescriptor("error")
	metadataEditType_ : MetadataEditTypePlug = PlugDescriptor("metadataEditType")
	metadataEditValue_ : MetadataEditValuePlug = PlugDescriptor("metadataEditValue")
	metadataIndexValue_ : MetadataIndexValuePlug = PlugDescriptor("metadataIndexValue")
	metadataMember_ : MetadataMemberPlug = PlugDescriptor("metadataMember")

	# node attributes

	typeName = "editMetadata"
	apiTypeInt = 1089
	apiTypeStr = "kEditMetadata"
	typeIdInt = 1162695748
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "streamEdits", "streamName", "channelEdits", "channelName", "edits", "inData", "outData", "error", "metadataEditType", "metadataEditValue", "metadataIndexValue", "metadataMember"]
	nodeLeafPlugs = ["binMembership", "edits", "inData", "outData"]
	pass

