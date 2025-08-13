

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Field = retriever.getNodeCls("Field")
assert Field
if T.TYPE_CHECKING:
	from .. import Field

# add node doc



# region plug type defs
class FrequencyPlug(Plug):
	node : TurbulenceField = None
	pass
class InterpolationTypePlug(Plug):
	node : TurbulenceField = None
	pass
class NoiseLevelPlug(Plug):
	node : TurbulenceField = None
	pass
class NoiseRatioPlug(Plug):
	node : TurbulenceField = None
	pass
class PhaseXPlug(Plug):
	node : TurbulenceField = None
	pass
class PhaseYPlug(Plug):
	node : TurbulenceField = None
	pass
class PhaseZPlug(Plug):
	node : TurbulenceField = None
	pass
# endregion


# define node class
class TurbulenceField(Field):
	frequency_ : FrequencyPlug = PlugDescriptor("frequency")
	interpolationType_ : InterpolationTypePlug = PlugDescriptor("interpolationType")
	noiseLevel_ : NoiseLevelPlug = PlugDescriptor("noiseLevel")
	noiseRatio_ : NoiseRatioPlug = PlugDescriptor("noiseRatio")
	phaseX_ : PhaseXPlug = PlugDescriptor("phaseX")
	phaseY_ : PhaseYPlug = PlugDescriptor("phaseY")
	phaseZ_ : PhaseZPlug = PlugDescriptor("phaseZ")

	# node attributes

	typeName = "turbulenceField"
	typeIdInt = 1498699090
	pass

