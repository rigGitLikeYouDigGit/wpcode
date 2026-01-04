

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
class SourcePlug(Plug):
	parent : AnimationPlug = PlugDescriptor("animation")
	node : TimeEditorAnimSource = None
	pass
class SourcePathPlug(Plug):
	parent : AnimationPlug = PlugDescriptor("animation")
	node : TimeEditorAnimSource = None
	pass
class SourceValuePlug(Plug):
	parent : AnimationPlug = PlugDescriptor("animation")
	node : TimeEditorAnimSource = None
	pass
class TargetPlug(Plug):
	parent : AnimationPlug = PlugDescriptor("animation")
	node : TimeEditorAnimSource = None
	pass
class AnimationPlug(Plug):
	source_ : SourcePlug = PlugDescriptor("source")
	as_ : SourcePlug = PlugDescriptor("source")
	sourcePath_ : SourcePathPlug = PlugDescriptor("sourcePath")
	asp_ : SourcePathPlug = PlugDescriptor("sourcePath")
	sourceValue_ : SourceValuePlug = PlugDescriptor("sourceValue")
	asv_ : SourceValuePlug = PlugDescriptor("sourceValue")
	target_ : TargetPlug = PlugDescriptor("target")
	at_ : TargetPlug = PlugDescriptor("target")
	node : TimeEditorAnimSource = None
	pass
class BinMembershipPlug(Plug):
	node : TimeEditorAnimSource = None
	pass
class BlendShapeSourcePlug(Plug):
	node : TimeEditorAnimSource = None
	pass
class DurationPlug(Plug):
	node : TimeEditorAnimSource = None
	pass
class InitialClipAbsoluteDurationPlug(Plug):
	node : TimeEditorAnimSource = None
	pass
class InitialClipDurationPlug(Plug):
	node : TimeEditorAnimSource = None
	pass
class InitialClipStartPlug(Plug):
	node : TimeEditorAnimSource = None
	pass
class RostersPlug(Plug):
	node : TimeEditorAnimSource = None
	pass
class StartPlug(Plug):
	node : TimeEditorAnimSource = None
	pass
# endregion


# define node class
class TimeEditorAnimSource(_BASE_):
	source_ : SourcePlug = PlugDescriptor("source")
	sourcePath_ : SourcePathPlug = PlugDescriptor("sourcePath")
	sourceValue_ : SourceValuePlug = PlugDescriptor("sourceValue")
	target_ : TargetPlug = PlugDescriptor("target")
	animation_ : AnimationPlug = PlugDescriptor("animation")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	blendShapeSource_ : BlendShapeSourcePlug = PlugDescriptor("blendShapeSource")
	duration_ : DurationPlug = PlugDescriptor("duration")
	initialClipAbsoluteDuration_ : InitialClipAbsoluteDurationPlug = PlugDescriptor("initialClipAbsoluteDuration")
	initialClipDuration_ : InitialClipDurationPlug = PlugDescriptor("initialClipDuration")
	initialClipStart_ : InitialClipStartPlug = PlugDescriptor("initialClipStart")
	rosters_ : RostersPlug = PlugDescriptor("rosters")
	start_ : StartPlug = PlugDescriptor("start")

	# node attributes

	typeName = "timeEditorAnimSource"
	apiTypeInt = 1109
	apiTypeStr = "kTimeEditorAnimSource"
	typeIdInt = 1413824851
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["source", "sourcePath", "sourceValue", "target", "animation", "binMembership", "blendShapeSource", "duration", "initialClipAbsoluteDuration", "initialClipDuration", "initialClipStart", "rosters", "start"]
	nodeLeafPlugs = ["animation", "binMembership", "blendShapeSource", "duration", "initialClipAbsoluteDuration", "initialClipDuration", "initialClipStart", "rosters", "start"]
	pass

