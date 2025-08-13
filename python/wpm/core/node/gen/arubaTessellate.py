

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
assert AbstractBaseCreate
if T.TYPE_CHECKING:
	from .. import AbstractBaseCreate

# add node doc



# region plug type defs
class AdaptivePlug(Plug):
	node : ArubaTessellate = None
	pass
class ChordalDeviationPlug(Plug):
	node : ArubaTessellate = None
	pass
class InputSurfacePlug(Plug):
	node : ArubaTessellate = None
	pass
class MaxChordLengthPlug(Plug):
	node : ArubaTessellate = None
	pass
class MinChordLengthPlug(Plug):
	node : ArubaTessellate = None
	pass
class NormalTolerancePlug(Plug):
	node : ArubaTessellate = None
	pass
class OutMeshPlug(Plug):
	node : ArubaTessellate = None
	pass
class SampleTypePlug(Plug):
	node : ArubaTessellate = None
	pass
class SamplesPlug(Plug):
	node : ArubaTessellate = None
	pass
class TolerancePlug(Plug):
	node : ArubaTessellate = None
	pass
# endregion


# define node class
class ArubaTessellate(AbstractBaseCreate):
	adaptive_ : AdaptivePlug = PlugDescriptor("adaptive")
	chordalDeviation_ : ChordalDeviationPlug = PlugDescriptor("chordalDeviation")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	maxChordLength_ : MaxChordLengthPlug = PlugDescriptor("maxChordLength")
	minChordLength_ : MinChordLengthPlug = PlugDescriptor("minChordLength")
	normalTolerance_ : NormalTolerancePlug = PlugDescriptor("normalTolerance")
	outMesh_ : OutMeshPlug = PlugDescriptor("outMesh")
	sampleType_ : SampleTypePlug = PlugDescriptor("sampleType")
	samples_ : SamplesPlug = PlugDescriptor("samples")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")

	# node attributes

	typeName = "arubaTessellate"
	typeIdInt = 1096041811
	pass

