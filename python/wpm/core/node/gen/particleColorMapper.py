

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
	node : ParticleColorMapper = None
	pass
class ParticleColorBPlug(Plug):
	parent : ParticleColorPlug = PlugDescriptor("particleColor")
	node : ParticleColorMapper = None
	pass
class ParticleColorGPlug(Plug):
	parent : ParticleColorPlug = PlugDescriptor("particleColor")
	node : ParticleColorMapper = None
	pass
class ParticleColorRPlug(Plug):
	parent : ParticleColorPlug = PlugDescriptor("particleColor")
	node : ParticleColorMapper = None
	pass
class ParticleColorPlug(Plug):
	particleColorB_ : ParticleColorBPlug = PlugDescriptor("particleColorB")
	pcb_ : ParticleColorBPlug = PlugDescriptor("particleColorB")
	particleColorG_ : ParticleColorGPlug = PlugDescriptor("particleColorG")
	pcg_ : ParticleColorGPlug = PlugDescriptor("particleColorG")
	particleColorR_ : ParticleColorRPlug = PlugDescriptor("particleColorR")
	pcr_ : ParticleColorRPlug = PlugDescriptor("particleColorR")
	node : ParticleColorMapper = None
	pass
# endregion


# define node class
class ParticleColorMapper(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	particleColorB_ : ParticleColorBPlug = PlugDescriptor("particleColorB")
	particleColorG_ : ParticleColorGPlug = PlugDescriptor("particleColorG")
	particleColorR_ : ParticleColorRPlug = PlugDescriptor("particleColorR")
	particleColor_ : ParticleColorPlug = PlugDescriptor("particleColor")

	# node attributes

	typeName = "particleColorMapper"
	apiTypeInt = 453
	apiTypeStr = "kParticleColorMapper"
	typeIdInt = 1346587969
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "particleColorB", "particleColorG", "particleColorR", "particleColor"]
	nodeLeafPlugs = ["binMembership", "particleColor"]
	pass

