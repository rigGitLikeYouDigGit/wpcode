

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
class AllOnPlug(Plug):
	node : DynController = None
	pass
class AllOnWhenRunPlug(Plug):
	node : DynController = None
	pass
class AutoCreatePlug(Plug):
	node : DynController = None
	pass
class BinMembershipPlug(Plug):
	node : DynController = None
	pass
class BreakRunupPlug(Plug):
	node : DynController = None
	pass
class CacheTimePlug(Plug):
	node : DynController = None
	pass
class CurrEvalTimePlug(Plug):
	node : DynController = None
	pass
class DoRunupPlug(Plug):
	node : DynController = None
	pass
class EvalTimePlug(Plug):
	node : DynController = None
	pass
class FirstEvalPlug(Plug):
	node : DynController = None
	pass
class LastEvalTimePlug(Plug):
	node : DynController = None
	pass
class MakeDirtyPlug(Plug):
	node : DynController = None
	pass
class OutputPlug(Plug):
	node : DynController = None
	pass
class OversamplePlug(Plug):
	node : DynController = None
	pass
class ParticleCachePlug(Plug):
	node : DynController = None
	pass
class ParticleLODPlug(Plug):
	node : DynController = None
	pass
class ParticlesOnPlug(Plug):
	node : DynController = None
	pass
class RigidOnPlug(Plug):
	node : DynController = None
	pass
class SeedPlug(Plug):
	node : DynController = None
	pass
class StartFramePlug(Plug):
	node : DynController = None
	pass
class StartRunupPlug(Plug):
	node : DynController = None
	pass
class StartTimePlug(Plug):
	node : DynController = None
	pass
class TraceDepthPlug(Plug):
	node : DynController = None
	pass
# endregion


# define node class
class DynController(_BASE_):
	allOn_ : AllOnPlug = PlugDescriptor("allOn")
	allOnWhenRun_ : AllOnWhenRunPlug = PlugDescriptor("allOnWhenRun")
	autoCreate_ : AutoCreatePlug = PlugDescriptor("autoCreate")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	breakRunup_ : BreakRunupPlug = PlugDescriptor("breakRunup")
	cacheTime_ : CacheTimePlug = PlugDescriptor("cacheTime")
	currEvalTime_ : CurrEvalTimePlug = PlugDescriptor("currEvalTime")
	doRunup_ : DoRunupPlug = PlugDescriptor("doRunup")
	evalTime_ : EvalTimePlug = PlugDescriptor("evalTime")
	firstEval_ : FirstEvalPlug = PlugDescriptor("firstEval")
	lastEvalTime_ : LastEvalTimePlug = PlugDescriptor("lastEvalTime")
	makeDirty_ : MakeDirtyPlug = PlugDescriptor("makeDirty")
	output_ : OutputPlug = PlugDescriptor("output")
	oversample_ : OversamplePlug = PlugDescriptor("oversample")
	particleCache_ : ParticleCachePlug = PlugDescriptor("particleCache")
	particleLOD_ : ParticleLODPlug = PlugDescriptor("particleLOD")
	particlesOn_ : ParticlesOnPlug = PlugDescriptor("particlesOn")
	rigidOn_ : RigidOnPlug = PlugDescriptor("rigidOn")
	seed_ : SeedPlug = PlugDescriptor("seed")
	startFrame_ : StartFramePlug = PlugDescriptor("startFrame")
	startRunup_ : StartRunupPlug = PlugDescriptor("startRunup")
	startTime_ : StartTimePlug = PlugDescriptor("startTime")
	traceDepth_ : TraceDepthPlug = PlugDescriptor("traceDepth")

	# node attributes

	typeName = "dynController"
	typeIdInt = 1497584716
	nodeLeafClassAttrs = ["allOn", "allOnWhenRun", "autoCreate", "binMembership", "breakRunup", "cacheTime", "currEvalTime", "doRunup", "evalTime", "firstEval", "lastEvalTime", "makeDirty", "output", "oversample", "particleCache", "particleLOD", "particlesOn", "rigidOn", "seed", "startFrame", "startRunup", "startTime", "traceDepth"]
	nodeLeafPlugs = ["allOn", "allOnWhenRun", "autoCreate", "binMembership", "breakRunup", "cacheTime", "currEvalTime", "doRunup", "evalTime", "firstEval", "lastEvalTime", "makeDirty", "output", "oversample", "particleCache", "particleLOD", "particlesOn", "rigidOn", "seed", "startFrame", "startRunup", "startTime", "traceDepth"]
	pass

