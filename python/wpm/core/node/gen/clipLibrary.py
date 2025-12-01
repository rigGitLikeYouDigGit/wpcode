

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
class ActiveClipPlug(Plug):
	node : ClipLibrary = None
	pass
class BinMembershipPlug(Plug):
	node : ClipLibrary = None
	pass
class CharacterMappingPlug(Plug):
	parent : CharacterdataPlug = PlugDescriptor("characterdata")
	node : ClipLibrary = None
	pass
class ClipIndexMappingPlug(Plug):
	parent : CharacterdataPlug = PlugDescriptor("characterdata")
	node : ClipLibrary = None
	pass
class CharacterdataPlug(Plug):
	characterMapping_ : CharacterMappingPlug = PlugDescriptor("characterMapping")
	cm_ : CharacterMappingPlug = PlugDescriptor("characterMapping")
	clipIndexMapping_ : ClipIndexMappingPlug = PlugDescriptor("clipIndexMapping")
	cim_ : ClipIndexMappingPlug = PlugDescriptor("clipIndexMapping")
	node : ClipLibrary = None
	pass
class ClipPlug(Plug):
	node : ClipLibrary = None
	pass
class ClipEval_HiddenPlug(Plug):
	parent : ClipEvalPlug = PlugDescriptor("clipEval")
	node : ClipLibrary = None
	pass
class ClipEval_InmapFromPlug(Plug):
	parent : ClipEval_InmapPlug = PlugDescriptor("clipEval_Inmap")
	node : ClipLibrary = None
	pass
class ClipEval_InmapToPlug(Plug):
	parent : ClipEval_InmapPlug = PlugDescriptor("clipEval_Inmap")
	node : ClipLibrary = None
	pass
class ClipEval_InmapPlug(Plug):
	parent : ClipEvalPlug = PlugDescriptor("clipEval")
	clipEval_InmapFrom_ : ClipEval_InmapFromPlug = PlugDescriptor("clipEval_InmapFrom")
	cevif_ : ClipEval_InmapFromPlug = PlugDescriptor("clipEval_InmapFrom")
	clipEval_InmapTo_ : ClipEval_InmapToPlug = PlugDescriptor("clipEval_InmapTo")
	cevit_ : ClipEval_InmapToPlug = PlugDescriptor("clipEval_InmapTo")
	node : ClipLibrary = None
	pass
class ClipEval_OutmapFromPlug(Plug):
	parent : ClipEval_OutmapPlug = PlugDescriptor("clipEval_Outmap")
	node : ClipLibrary = None
	pass
class ClipEval_OutmapToPlug(Plug):
	parent : ClipEval_OutmapPlug = PlugDescriptor("clipEval_Outmap")
	node : ClipLibrary = None
	pass
class ClipEval_OutmapPlug(Plug):
	parent : ClipEvalPlug = PlugDescriptor("clipEval")
	clipEval_OutmapFrom_ : ClipEval_OutmapFromPlug = PlugDescriptor("clipEval_OutmapFrom")
	cevof_ : ClipEval_OutmapFromPlug = PlugDescriptor("clipEval_OutmapFrom")
	clipEval_OutmapTo_ : ClipEval_OutmapToPlug = PlugDescriptor("clipEval_OutmapTo")
	cevot_ : ClipEval_OutmapToPlug = PlugDescriptor("clipEval_OutmapTo")
	node : ClipLibrary = None
	pass
class ClipEval_RawPlug(Plug):
	parent : ClipEvalPlug = PlugDescriptor("clipEval")
	node : ClipLibrary = None
	pass
class ClipEvalPlug(Plug):
	parent : ClipEvalListPlug = PlugDescriptor("clipEvalList")
	clipEval_Hidden_ : ClipEval_HiddenPlug = PlugDescriptor("clipEval_Hidden")
	cevh_ : ClipEval_HiddenPlug = PlugDescriptor("clipEval_Hidden")
	clipEval_Inmap_ : ClipEval_InmapPlug = PlugDescriptor("clipEval_Inmap")
	cevi_ : ClipEval_InmapPlug = PlugDescriptor("clipEval_Inmap")
	clipEval_Outmap_ : ClipEval_OutmapPlug = PlugDescriptor("clipEval_Outmap")
	cevo_ : ClipEval_OutmapPlug = PlugDescriptor("clipEval_Outmap")
	clipEval_Raw_ : ClipEval_RawPlug = PlugDescriptor("clipEval_Raw")
	cevr_ : ClipEval_RawPlug = PlugDescriptor("clipEval_Raw")
	node : ClipLibrary = None
	pass
class ClipEvalListPlug(Plug):
	clipEval_ : ClipEvalPlug = PlugDescriptor("clipEval")
	cev_ : ClipEvalPlug = PlugDescriptor("clipEval")
	node : ClipLibrary = None
	pass
class ClipFunctionPlug(Plug):
	node : ClipLibrary = None
	pass
class ClipNamePlug(Plug):
	node : ClipLibrary = None
	pass
class DurationPlug(Plug):
	node : ClipLibrary = None
	pass
class SourceClipPlug(Plug):
	node : ClipLibrary = None
	pass
class StartPlug(Plug):
	node : ClipLibrary = None
	pass
# endregion


# define node class
class ClipLibrary(_BASE_):
	activeClip_ : ActiveClipPlug = PlugDescriptor("activeClip")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	characterMapping_ : CharacterMappingPlug = PlugDescriptor("characterMapping")
	clipIndexMapping_ : ClipIndexMappingPlug = PlugDescriptor("clipIndexMapping")
	characterdata_ : CharacterdataPlug = PlugDescriptor("characterdata")
	clip_ : ClipPlug = PlugDescriptor("clip")
	clipEval_Hidden_ : ClipEval_HiddenPlug = PlugDescriptor("clipEval_Hidden")
	clipEval_InmapFrom_ : ClipEval_InmapFromPlug = PlugDescriptor("clipEval_InmapFrom")
	clipEval_InmapTo_ : ClipEval_InmapToPlug = PlugDescriptor("clipEval_InmapTo")
	clipEval_Inmap_ : ClipEval_InmapPlug = PlugDescriptor("clipEval_Inmap")
	clipEval_OutmapFrom_ : ClipEval_OutmapFromPlug = PlugDescriptor("clipEval_OutmapFrom")
	clipEval_OutmapTo_ : ClipEval_OutmapToPlug = PlugDescriptor("clipEval_OutmapTo")
	clipEval_Outmap_ : ClipEval_OutmapPlug = PlugDescriptor("clipEval_Outmap")
	clipEval_Raw_ : ClipEval_RawPlug = PlugDescriptor("clipEval_Raw")
	clipEval_ : ClipEvalPlug = PlugDescriptor("clipEval")
	clipEvalList_ : ClipEvalListPlug = PlugDescriptor("clipEvalList")
	clipFunction_ : ClipFunctionPlug = PlugDescriptor("clipFunction")
	clipName_ : ClipNamePlug = PlugDescriptor("clipName")
	duration_ : DurationPlug = PlugDescriptor("duration")
	sourceClip_ : SourceClipPlug = PlugDescriptor("sourceClip")
	start_ : StartPlug = PlugDescriptor("start")

	# node attributes

	typeName = "clipLibrary"
	apiTypeInt = 780
	apiTypeStr = "kClipLibrary"
	typeIdInt = 1129072976
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["activeClip", "binMembership", "characterMapping", "clipIndexMapping", "characterdata", "clip", "clipEval_Hidden", "clipEval_InmapFrom", "clipEval_InmapTo", "clipEval_Inmap", "clipEval_OutmapFrom", "clipEval_OutmapTo", "clipEval_Outmap", "clipEval_Raw", "clipEval", "clipEvalList", "clipFunction", "clipName", "duration", "sourceClip", "start"]
	nodeLeafPlugs = ["activeClip", "binMembership", "characterdata", "clip", "clipEvalList", "clipFunction", "clipName", "duration", "sourceClip", "start"]
	pass

