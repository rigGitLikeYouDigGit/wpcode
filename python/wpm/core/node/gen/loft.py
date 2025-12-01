

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
class AutoReversePlug(Plug):
	node : Loft = None
	pass
class ClosePlug(Plug):
	node : Loft = None
	pass
class CreateCuspPlug(Plug):
	node : Loft = None
	pass
class DegreePlug(Plug):
	node : Loft = None
	pass
class InputCurvePlug(Plug):
	node : Loft = None
	pass
class OutputSurfacePlug(Plug):
	node : Loft = None
	pass
class ReversePlug(Plug):
	node : Loft = None
	pass
class ReverseSurfaceNormalsPlug(Plug):
	node : Loft = None
	pass
class SectionSpansPlug(Plug):
	node : Loft = None
	pass
class UniformPlug(Plug):
	node : Loft = None
	pass
# endregion


# define node class
class Loft(AbstractBaseCreate):
	autoReverse_ : AutoReversePlug = PlugDescriptor("autoReverse")
	close_ : ClosePlug = PlugDescriptor("close")
	createCusp_ : CreateCuspPlug = PlugDescriptor("createCusp")
	degree_ : DegreePlug = PlugDescriptor("degree")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	reverse_ : ReversePlug = PlugDescriptor("reverse")
	reverseSurfaceNormals_ : ReverseSurfaceNormalsPlug = PlugDescriptor("reverseSurfaceNormals")
	sectionSpans_ : SectionSpansPlug = PlugDescriptor("sectionSpans")
	uniform_ : UniformPlug = PlugDescriptor("uniform")

	# node attributes

	typeName = "loft"
	typeIdInt = 1314081614
	nodeLeafClassAttrs = ["autoReverse", "close", "createCusp", "degree", "inputCurve", "outputSurface", "reverse", "reverseSurfaceNormals", "sectionSpans", "uniform"]
	nodeLeafPlugs = ["autoReverse", "close", "createCusp", "degree", "inputCurve", "outputSurface", "reverse", "reverseSurfaceNormals", "sectionSpans", "uniform"]
	pass

