

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
	node : ParticleTranspMapper = None
	pass
class ParticleTransparencyBPlug(Plug):
	parent : ParticleTransparencyPlug = PlugDescriptor("particleTransparency")
	node : ParticleTranspMapper = None
	pass
class ParticleTransparencyGPlug(Plug):
	parent : ParticleTransparencyPlug = PlugDescriptor("particleTransparency")
	node : ParticleTranspMapper = None
	pass
class ParticleTransparencyRPlug(Plug):
	parent : ParticleTransparencyPlug = PlugDescriptor("particleTransparency")
	node : ParticleTranspMapper = None
	pass
class ParticleTransparencyPlug(Plug):
	particleTransparencyB_ : ParticleTransparencyBPlug = PlugDescriptor("particleTransparencyB")
	ptb_ : ParticleTransparencyBPlug = PlugDescriptor("particleTransparencyB")
	particleTransparencyG_ : ParticleTransparencyGPlug = PlugDescriptor("particleTransparencyG")
	ptg_ : ParticleTransparencyGPlug = PlugDescriptor("particleTransparencyG")
	particleTransparencyR_ : ParticleTransparencyRPlug = PlugDescriptor("particleTransparencyR")
	ptr_ : ParticleTransparencyRPlug = PlugDescriptor("particleTransparencyR")
	node : ParticleTranspMapper = None
	pass
# endregion


# define node class
class ParticleTranspMapper(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	particleTransparencyB_ : ParticleTransparencyBPlug = PlugDescriptor("particleTransparencyB")
	particleTransparencyG_ : ParticleTransparencyGPlug = PlugDescriptor("particleTransparencyG")
	particleTransparencyR_ : ParticleTransparencyRPlug = PlugDescriptor("particleTransparencyR")
	particleTransparency_ : ParticleTransparencyPlug = PlugDescriptor("particleTransparency")

	# node attributes

	typeName = "particleTranspMapper"
	typeIdInt = 1347702081
	nodeLeafClassAttrs = ["binMembership", "particleTransparencyB", "particleTransparencyG", "particleTransparencyR", "particleTransparency"]
	nodeLeafPlugs = ["binMembership", "particleTransparency"]
	pass

