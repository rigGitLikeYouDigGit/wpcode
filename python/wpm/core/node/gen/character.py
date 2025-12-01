

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ObjectSet = retriever.getNodeCls("ObjectSet")
assert ObjectSet
if T.TYPE_CHECKING:
	from .. import ObjectSet

# add node doc



# region plug type defs
class ActiveClipConnectedPlug(Plug):
	node : Character = None
	pass
class AngularClipValuesPlug(Plug):
	node : Character = None
	pass
class AngularValuesPlug(Plug):
	node : Character = None
	pass
class AnimationMappingPlug(Plug):
	node : Character = None
	pass
class ClipEvaluate_HiddenPlug(Plug):
	parent : ClipEvaluatePlug = PlugDescriptor("clipEvaluate")
	node : Character = None
	pass
class ClipEvaluate_InmapFromPlug(Plug):
	parent : ClipEvaluate_InmapPlug = PlugDescriptor("clipEvaluate_Inmap")
	node : Character = None
	pass
class ClipEvaluate_InmapToPlug(Plug):
	parent : ClipEvaluate_InmapPlug = PlugDescriptor("clipEvaluate_Inmap")
	node : Character = None
	pass
class ClipEvaluate_InmapPlug(Plug):
	parent : ClipEvaluatePlug = PlugDescriptor("clipEvaluate")
	clipEvaluate_InmapFrom_ : ClipEvaluate_InmapFromPlug = PlugDescriptor("clipEvaluate_InmapFrom")
	ceif_ : ClipEvaluate_InmapFromPlug = PlugDescriptor("clipEvaluate_InmapFrom")
	clipEvaluate_InmapTo_ : ClipEvaluate_InmapToPlug = PlugDescriptor("clipEvaluate_InmapTo")
	ceit_ : ClipEvaluate_InmapToPlug = PlugDescriptor("clipEvaluate_InmapTo")
	node : Character = None
	pass
class ClipEvaluate_OutmapFromPlug(Plug):
	parent : ClipEvaluate_OutmapPlug = PlugDescriptor("clipEvaluate_Outmap")
	node : Character = None
	pass
class ClipEvaluate_OutmapToPlug(Plug):
	parent : ClipEvaluate_OutmapPlug = PlugDescriptor("clipEvaluate_Outmap")
	node : Character = None
	pass
class ClipEvaluate_OutmapPlug(Plug):
	parent : ClipEvaluatePlug = PlugDescriptor("clipEvaluate")
	clipEvaluate_OutmapFrom_ : ClipEvaluate_OutmapFromPlug = PlugDescriptor("clipEvaluate_OutmapFrom")
	ceof_ : ClipEvaluate_OutmapFromPlug = PlugDescriptor("clipEvaluate_OutmapFrom")
	clipEvaluate_OutmapTo_ : ClipEvaluate_OutmapToPlug = PlugDescriptor("clipEvaluate_OutmapTo")
	ceot_ : ClipEvaluate_OutmapToPlug = PlugDescriptor("clipEvaluate_OutmapTo")
	node : Character = None
	pass
class ClipEvaluate_RawPlug(Plug):
	parent : ClipEvaluatePlug = PlugDescriptor("clipEvaluate")
	node : Character = None
	pass
class ClipEvaluatePlug(Plug):
	clipEvaluate_Hidden_ : ClipEvaluate_HiddenPlug = PlugDescriptor("clipEvaluate_Hidden")
	ceh_ : ClipEvaluate_HiddenPlug = PlugDescriptor("clipEvaluate_Hidden")
	clipEvaluate_Inmap_ : ClipEvaluate_InmapPlug = PlugDescriptor("clipEvaluate_Inmap")
	cei_ : ClipEvaluate_InmapPlug = PlugDescriptor("clipEvaluate_Inmap")
	clipEvaluate_Outmap_ : ClipEvaluate_OutmapPlug = PlugDescriptor("clipEvaluate_Outmap")
	ceo_ : ClipEvaluate_OutmapPlug = PlugDescriptor("clipEvaluate_Outmap")
	clipEvaluate_Raw_ : ClipEvaluate_RawPlug = PlugDescriptor("clipEvaluate_Raw")
	cer_ : ClipEvaluate_RawPlug = PlugDescriptor("clipEvaluate_Raw")
	node : Character = None
	pass
class ClipIndexMapPlug(Plug):
	node : Character = None
	pass
class ClipStatePercentEval_HiddenPlug(Plug):
	parent : ClipStatePercentEvalPlug = PlugDescriptor("clipStatePercentEval")
	node : Character = None
	pass
class ClipStatePercentEval_InmapFromPlug(Plug):
	parent : ClipStatePercentEval_InmapPlug = PlugDescriptor("clipStatePercentEval_Inmap")
	node : Character = None
	pass
class ClipStatePercentEval_InmapToPlug(Plug):
	parent : ClipStatePercentEval_InmapPlug = PlugDescriptor("clipStatePercentEval_Inmap")
	node : Character = None
	pass
class ClipStatePercentEval_InmapPlug(Plug):
	parent : ClipStatePercentEvalPlug = PlugDescriptor("clipStatePercentEval")
	clipStatePercentEval_InmapFrom_ : ClipStatePercentEval_InmapFromPlug = PlugDescriptor("clipStatePercentEval_InmapFrom")
	cspeif_ : ClipStatePercentEval_InmapFromPlug = PlugDescriptor("clipStatePercentEval_InmapFrom")
	clipStatePercentEval_InmapTo_ : ClipStatePercentEval_InmapToPlug = PlugDescriptor("clipStatePercentEval_InmapTo")
	cspeit_ : ClipStatePercentEval_InmapToPlug = PlugDescriptor("clipStatePercentEval_InmapTo")
	node : Character = None
	pass
class ClipStatePercentEval_OutmapFromPlug(Plug):
	parent : ClipStatePercentEval_OutmapPlug = PlugDescriptor("clipStatePercentEval_Outmap")
	node : Character = None
	pass
class ClipStatePercentEval_OutmapToPlug(Plug):
	parent : ClipStatePercentEval_OutmapPlug = PlugDescriptor("clipStatePercentEval_Outmap")
	node : Character = None
	pass
class ClipStatePercentEval_OutmapPlug(Plug):
	parent : ClipStatePercentEvalPlug = PlugDescriptor("clipStatePercentEval")
	clipStatePercentEval_OutmapFrom_ : ClipStatePercentEval_OutmapFromPlug = PlugDescriptor("clipStatePercentEval_OutmapFrom")
	cspeof_ : ClipStatePercentEval_OutmapFromPlug = PlugDescriptor("clipStatePercentEval_OutmapFrom")
	clipStatePercentEval_OutmapTo_ : ClipStatePercentEval_OutmapToPlug = PlugDescriptor("clipStatePercentEval_OutmapTo")
	cspeot_ : ClipStatePercentEval_OutmapToPlug = PlugDescriptor("clipStatePercentEval_OutmapTo")
	node : Character = None
	pass
class ClipStatePercentEval_RawPlug(Plug):
	parent : ClipStatePercentEvalPlug = PlugDescriptor("clipStatePercentEval")
	node : Character = None
	pass
class ClipStatePercentEvalPlug(Plug):
	clipStatePercentEval_Hidden_ : ClipStatePercentEval_HiddenPlug = PlugDescriptor("clipStatePercentEval_Hidden")
	cspeh_ : ClipStatePercentEval_HiddenPlug = PlugDescriptor("clipStatePercentEval_Hidden")
	clipStatePercentEval_Inmap_ : ClipStatePercentEval_InmapPlug = PlugDescriptor("clipStatePercentEval_Inmap")
	cspei_ : ClipStatePercentEval_InmapPlug = PlugDescriptor("clipStatePercentEval_Inmap")
	clipStatePercentEval_Outmap_ : ClipStatePercentEval_OutmapPlug = PlugDescriptor("clipStatePercentEval_Outmap")
	cspeo_ : ClipStatePercentEval_OutmapPlug = PlugDescriptor("clipStatePercentEval_Outmap")
	clipStatePercentEval_Raw_ : ClipStatePercentEval_RawPlug = PlugDescriptor("clipStatePercentEval_Raw")
	csper_ : ClipStatePercentEval_RawPlug = PlugDescriptor("clipStatePercentEval_Raw")
	node : Character = None
	pass
class CopyAngularValuesPlug(Plug):
	node : Character = None
	pass
class CopyLinearValuesPlug(Plug):
	node : Character = None
	pass
class CopyTimeValuesPlug(Plug):
	node : Character = None
	pass
class CopyUnitlessValuesPlug(Plug):
	node : Character = None
	pass
class EvalCharacterKeysPlug(Plug):
	node : Character = None
	pass
class LinearClipValuesPlug(Plug):
	node : Character = None
	pass
class LinearValuesPlug(Plug):
	node : Character = None
	pass
class MatchNodePlug(Plug):
	node : Character = None
	pass
class OffsetNodePlug(Plug):
	node : Character = None
	pass
class OffsetObjectLocalXFormPlug(Plug):
	node : Character = None
	pass
class OffsetObjectLocalXFormsPlug(Plug):
	node : Character = None
	pass
class OffsetObjectsPlug(Plug):
	node : Character = None
	pass
class ReferenceMappingPlug(Plug):
	node : Character = None
	pass
class TimeClipValuesPlug(Plug):
	node : Character = None
	pass
class TimeValuesPlug(Plug):
	node : Character = None
	pass
class TimelineClipEndPlug(Plug):
	node : Character = None
	pass
class TimelineClipStartPlug(Plug):
	node : Character = None
	pass
class TranslationOffsetIndexXPlug(Plug):
	parent : TranslationOffsetIndicesPlug = PlugDescriptor("translationOffsetIndices")
	node : Character = None
	pass
class TranslationOffsetYPlug(Plug):
	parent : TranslationOffsetIndicesPlug = PlugDescriptor("translationOffsetIndices")
	node : Character = None
	pass
class TranslationOffsetZPlug(Plug):
	parent : TranslationOffsetIndicesPlug = PlugDescriptor("translationOffsetIndices")
	node : Character = None
	pass
class TranslationOffsetIndicesPlug(Plug):
	translationOffsetIndexX_ : TranslationOffsetIndexXPlug = PlugDescriptor("translationOffsetIndexX")
	tox_ : TranslationOffsetIndexXPlug = PlugDescriptor("translationOffsetIndexX")
	translationOffsetY_ : TranslationOffsetYPlug = PlugDescriptor("translationOffsetY")
	toy_ : TranslationOffsetYPlug = PlugDescriptor("translationOffsetY")
	translationOffsetZ_ : TranslationOffsetZPlug = PlugDescriptor("translationOffsetZ")
	toz_ : TranslationOffsetZPlug = PlugDescriptor("translationOffsetZ")
	node : Character = None
	pass
class TranslationOffsetIndicesXPlug(Plug):
	node : Character = None
	pass
class TranslationOffsetIndicesYPlug(Plug):
	node : Character = None
	pass
class TranslationOffsetIndicesZPlug(Plug):
	node : Character = None
	pass
class UnitlessClipValuesPlug(Plug):
	node : Character = None
	pass
class UnitlessValuesPlug(Plug):
	node : Character = None
	pass
# endregion


# define node class
class Character(ObjectSet):
	activeClipConnected_ : ActiveClipConnectedPlug = PlugDescriptor("activeClipConnected")
	angularClipValues_ : AngularClipValuesPlug = PlugDescriptor("angularClipValues")
	angularValues_ : AngularValuesPlug = PlugDescriptor("angularValues")
	animationMapping_ : AnimationMappingPlug = PlugDescriptor("animationMapping")
	clipEvaluate_Hidden_ : ClipEvaluate_HiddenPlug = PlugDescriptor("clipEvaluate_Hidden")
	clipEvaluate_InmapFrom_ : ClipEvaluate_InmapFromPlug = PlugDescriptor("clipEvaluate_InmapFrom")
	clipEvaluate_InmapTo_ : ClipEvaluate_InmapToPlug = PlugDescriptor("clipEvaluate_InmapTo")
	clipEvaluate_Inmap_ : ClipEvaluate_InmapPlug = PlugDescriptor("clipEvaluate_Inmap")
	clipEvaluate_OutmapFrom_ : ClipEvaluate_OutmapFromPlug = PlugDescriptor("clipEvaluate_OutmapFrom")
	clipEvaluate_OutmapTo_ : ClipEvaluate_OutmapToPlug = PlugDescriptor("clipEvaluate_OutmapTo")
	clipEvaluate_Outmap_ : ClipEvaluate_OutmapPlug = PlugDescriptor("clipEvaluate_Outmap")
	clipEvaluate_Raw_ : ClipEvaluate_RawPlug = PlugDescriptor("clipEvaluate_Raw")
	clipEvaluate_ : ClipEvaluatePlug = PlugDescriptor("clipEvaluate")
	clipIndexMap_ : ClipIndexMapPlug = PlugDescriptor("clipIndexMap")
	clipStatePercentEval_Hidden_ : ClipStatePercentEval_HiddenPlug = PlugDescriptor("clipStatePercentEval_Hidden")
	clipStatePercentEval_InmapFrom_ : ClipStatePercentEval_InmapFromPlug = PlugDescriptor("clipStatePercentEval_InmapFrom")
	clipStatePercentEval_InmapTo_ : ClipStatePercentEval_InmapToPlug = PlugDescriptor("clipStatePercentEval_InmapTo")
	clipStatePercentEval_Inmap_ : ClipStatePercentEval_InmapPlug = PlugDescriptor("clipStatePercentEval_Inmap")
	clipStatePercentEval_OutmapFrom_ : ClipStatePercentEval_OutmapFromPlug = PlugDescriptor("clipStatePercentEval_OutmapFrom")
	clipStatePercentEval_OutmapTo_ : ClipStatePercentEval_OutmapToPlug = PlugDescriptor("clipStatePercentEval_OutmapTo")
	clipStatePercentEval_Outmap_ : ClipStatePercentEval_OutmapPlug = PlugDescriptor("clipStatePercentEval_Outmap")
	clipStatePercentEval_Raw_ : ClipStatePercentEval_RawPlug = PlugDescriptor("clipStatePercentEval_Raw")
	clipStatePercentEval_ : ClipStatePercentEvalPlug = PlugDescriptor("clipStatePercentEval")
	copyAngularValues_ : CopyAngularValuesPlug = PlugDescriptor("copyAngularValues")
	copyLinearValues_ : CopyLinearValuesPlug = PlugDescriptor("copyLinearValues")
	copyTimeValues_ : CopyTimeValuesPlug = PlugDescriptor("copyTimeValues")
	copyUnitlessValues_ : CopyUnitlessValuesPlug = PlugDescriptor("copyUnitlessValues")
	evalCharacterKeys_ : EvalCharacterKeysPlug = PlugDescriptor("evalCharacterKeys")
	linearClipValues_ : LinearClipValuesPlug = PlugDescriptor("linearClipValues")
	linearValues_ : LinearValuesPlug = PlugDescriptor("linearValues")
	matchNode_ : MatchNodePlug = PlugDescriptor("matchNode")
	offsetNode_ : OffsetNodePlug = PlugDescriptor("offsetNode")
	offsetObjectLocalXForm_ : OffsetObjectLocalXFormPlug = PlugDescriptor("offsetObjectLocalXForm")
	offsetObjectLocalXForms_ : OffsetObjectLocalXFormsPlug = PlugDescriptor("offsetObjectLocalXForms")
	offsetObjects_ : OffsetObjectsPlug = PlugDescriptor("offsetObjects")
	referenceMapping_ : ReferenceMappingPlug = PlugDescriptor("referenceMapping")
	timeClipValues_ : TimeClipValuesPlug = PlugDescriptor("timeClipValues")
	timeValues_ : TimeValuesPlug = PlugDescriptor("timeValues")
	timelineClipEnd_ : TimelineClipEndPlug = PlugDescriptor("timelineClipEnd")
	timelineClipStart_ : TimelineClipStartPlug = PlugDescriptor("timelineClipStart")
	translationOffsetIndexX_ : TranslationOffsetIndexXPlug = PlugDescriptor("translationOffsetIndexX")
	translationOffsetY_ : TranslationOffsetYPlug = PlugDescriptor("translationOffsetY")
	translationOffsetZ_ : TranslationOffsetZPlug = PlugDescriptor("translationOffsetZ")
	translationOffsetIndices_ : TranslationOffsetIndicesPlug = PlugDescriptor("translationOffsetIndices")
	translationOffsetIndicesX_ : TranslationOffsetIndicesXPlug = PlugDescriptor("translationOffsetIndicesX")
	translationOffsetIndicesY_ : TranslationOffsetIndicesYPlug = PlugDescriptor("translationOffsetIndicesY")
	translationOffsetIndicesZ_ : TranslationOffsetIndicesZPlug = PlugDescriptor("translationOffsetIndicesZ")
	unitlessClipValues_ : UnitlessClipValuesPlug = PlugDescriptor("unitlessClipValues")
	unitlessValues_ : UnitlessValuesPlug = PlugDescriptor("unitlessValues")

	# node attributes

	typeName = "character"
	apiTypeInt = 688
	apiTypeStr = "kCharacter"
	typeIdInt = 1128808786
	MFnCls = om.MFnSet
	nodeLeafClassAttrs = ["activeClipConnected", "angularClipValues", "angularValues", "animationMapping", "clipEvaluate_Hidden", "clipEvaluate_InmapFrom", "clipEvaluate_InmapTo", "clipEvaluate_Inmap", "clipEvaluate_OutmapFrom", "clipEvaluate_OutmapTo", "clipEvaluate_Outmap", "clipEvaluate_Raw", "clipEvaluate", "clipIndexMap", "clipStatePercentEval_Hidden", "clipStatePercentEval_InmapFrom", "clipStatePercentEval_InmapTo", "clipStatePercentEval_Inmap", "clipStatePercentEval_OutmapFrom", "clipStatePercentEval_OutmapTo", "clipStatePercentEval_Outmap", "clipStatePercentEval_Raw", "clipStatePercentEval", "copyAngularValues", "copyLinearValues", "copyTimeValues", "copyUnitlessValues", "evalCharacterKeys", "linearClipValues", "linearValues", "matchNode", "offsetNode", "offsetObjectLocalXForm", "offsetObjectLocalXForms", "offsetObjects", "referenceMapping", "timeClipValues", "timeValues", "timelineClipEnd", "timelineClipStart", "translationOffsetIndexX", "translationOffsetY", "translationOffsetZ", "translationOffsetIndices", "translationOffsetIndicesX", "translationOffsetIndicesY", "translationOffsetIndicesZ", "unitlessClipValues", "unitlessValues"]
	nodeLeafPlugs = ["activeClipConnected", "angularClipValues", "angularValues", "animationMapping", "clipEvaluate", "clipIndexMap", "clipStatePercentEval", "copyAngularValues", "copyLinearValues", "copyTimeValues", "copyUnitlessValues", "evalCharacterKeys", "linearClipValues", "linearValues", "matchNode", "offsetNode", "offsetObjectLocalXForm", "offsetObjectLocalXForms", "offsetObjects", "referenceMapping", "timeClipValues", "timeValues", "timelineClipEnd", "timelineClipStart", "translationOffsetIndices", "translationOffsetIndicesX", "translationOffsetIndicesY", "translationOffsetIndicesZ", "unitlessClipValues", "unitlessValues"]
	pass

