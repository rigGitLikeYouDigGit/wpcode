

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
	node : ParticleAgeMapper = None
	pass
class FoldAtEndPlug(Plug):
	node : ParticleAgeMapper = None
	pass
class OutUCoordPlug(Plug):
	parent : OutUvCoordPlug = PlugDescriptor("outUvCoord")
	node : ParticleAgeMapper = None
	pass
class OutVCoordPlug(Plug):
	parent : OutUvCoordPlug = PlugDescriptor("outUvCoord")
	node : ParticleAgeMapper = None
	pass
class OutUvCoordPlug(Plug):
	outUCoord_ : OutUCoordPlug = PlugDescriptor("outUCoord")
	ouc_ : OutUCoordPlug = PlugDescriptor("outUCoord")
	outVCoord_ : OutVCoordPlug = PlugDescriptor("outVCoord")
	ovc_ : OutVCoordPlug = PlugDescriptor("outVCoord")
	node : ParticleAgeMapper = None
	pass
class ParticleAgePlug(Plug):
	node : ParticleAgeMapper = None
	pass
class ParticleLifespanPlug(Plug):
	node : ParticleAgeMapper = None
	pass
class RelativeAgePlug(Plug):
	node : ParticleAgeMapper = None
	pass
class TimeScalePlug(Plug):
	node : ParticleAgeMapper = None
	pass
# endregion


# define node class
class ParticleAgeMapper(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	foldAtEnd_ : FoldAtEndPlug = PlugDescriptor("foldAtEnd")
	outUCoord_ : OutUCoordPlug = PlugDescriptor("outUCoord")
	outVCoord_ : OutVCoordPlug = PlugDescriptor("outVCoord")
	outUvCoord_ : OutUvCoordPlug = PlugDescriptor("outUvCoord")
	particleAge_ : ParticleAgePlug = PlugDescriptor("particleAge")
	particleLifespan_ : ParticleLifespanPlug = PlugDescriptor("particleLifespan")
	relativeAge_ : RelativeAgePlug = PlugDescriptor("relativeAge")
	timeScale_ : TimeScalePlug = PlugDescriptor("timeScale")

	# node attributes

	typeName = "particleAgeMapper"
	apiTypeInt = 451
	apiTypeStr = "kParticleAgeMapper"
	typeIdInt = 1346456897
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "foldAtEnd", "outUCoord", "outVCoord", "outUvCoord", "particleAge", "particleLifespan", "relativeAge", "timeScale"]
	nodeLeafPlugs = ["binMembership", "foldAtEnd", "outUvCoord", "particleAge", "particleLifespan", "relativeAge", "timeScale"]
	pass

