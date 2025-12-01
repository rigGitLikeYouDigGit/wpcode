

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
	node : PrecompExport = None
	pass
class CameraPlug(Plug):
	parent : ExcludedExportItemListPlug = PlugDescriptor("excludedExportItemList")
	node : PrecompExport = None
	pass
class LayerPlug(Plug):
	parent : ExcludedExportItemListPlug = PlugDescriptor("excludedExportItemList")
	node : PrecompExport = None
	pass
class PassPlug(Plug):
	parent : ExcludedExportItemListPlug = PlugDescriptor("excludedExportItemList")
	node : PrecompExport = None
	pass
class ExcludedExportItemListPlug(Plug):
	camera_ : CameraPlug = PlugDescriptor("camera")
	ec_ : CameraPlug = PlugDescriptor("camera")
	layer_ : LayerPlug = PlugDescriptor("layer")
	el_ : LayerPlug = PlugDescriptor("layer")
	pass_ : PassPlug = PlugDescriptor("pass")
	ep_ : PassPlug = PlugDescriptor("pass")
	node : PrecompExport = None
	pass
class PreCompositingAnchorPlug(Plug):
	node : PrecompExport = None
	pass
class PreCompositingNotesPlug(Plug):
	node : PrecompExport = None
	pass
# endregion


# define node class
class PrecompExport(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	camera_ : CameraPlug = PlugDescriptor("camera")
	layer_ : LayerPlug = PlugDescriptor("layer")
	pass_ : PassPlug = PlugDescriptor("pass")
	excludedExportItemList_ : ExcludedExportItemListPlug = PlugDescriptor("excludedExportItemList")
	preCompositingAnchor_ : PreCompositingAnchorPlug = PlugDescriptor("preCompositingAnchor")
	preCompositingNotes_ : PreCompositingNotesPlug = PlugDescriptor("preCompositingNotes")

	# node attributes

	typeName = "precompExport"
	apiTypeInt = 788
	apiTypeStr = "kPrecompExport"
	typeIdInt = 1415072581
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "camera", "layer", "pass", "excludedExportItemList", "preCompositingAnchor", "preCompositingNotes"]
	nodeLeafPlugs = ["binMembership", "excludedExportItemList", "preCompositingAnchor", "preCompositingNotes"]
	pass

