

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
	node : ParticleIncandMapper = None
	pass
class ParticleIncandescenceBPlug(Plug):
	parent : ParticleIncandescencePlug = PlugDescriptor("particleIncandescence")
	node : ParticleIncandMapper = None
	pass
class ParticleIncandescenceGPlug(Plug):
	parent : ParticleIncandescencePlug = PlugDescriptor("particleIncandescence")
	node : ParticleIncandMapper = None
	pass
class ParticleIncandescenceRPlug(Plug):
	parent : ParticleIncandescencePlug = PlugDescriptor("particleIncandescence")
	node : ParticleIncandMapper = None
	pass
class ParticleIncandescencePlug(Plug):
	particleIncandescenceB_ : ParticleIncandescenceBPlug = PlugDescriptor("particleIncandescenceB")
	pib_ : ParticleIncandescenceBPlug = PlugDescriptor("particleIncandescenceB")
	particleIncandescenceG_ : ParticleIncandescenceGPlug = PlugDescriptor("particleIncandescenceG")
	pig_ : ParticleIncandescenceGPlug = PlugDescriptor("particleIncandescenceG")
	particleIncandescenceR_ : ParticleIncandescenceRPlug = PlugDescriptor("particleIncandescenceR")
	pir_ : ParticleIncandescenceRPlug = PlugDescriptor("particleIncandescenceR")
	node : ParticleIncandMapper = None
	pass
# endregion


# define node class
class ParticleIncandMapper(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	particleIncandescenceB_ : ParticleIncandescenceBPlug = PlugDescriptor("particleIncandescenceB")
	particleIncandescenceG_ : ParticleIncandescenceGPlug = PlugDescriptor("particleIncandescenceG")
	particleIncandescenceR_ : ParticleIncandescenceRPlug = PlugDescriptor("particleIncandescenceR")
	particleIncandescence_ : ParticleIncandescencePlug = PlugDescriptor("particleIncandescence")

	# node attributes

	typeName = "particleIncandMapper"
	typeIdInt = 1346981185
	pass

