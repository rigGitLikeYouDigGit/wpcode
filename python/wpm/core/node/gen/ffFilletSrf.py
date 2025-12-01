

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
class BiasPlug(Plug):
	node : FfFilletSrf = None
	pass
class DepthPlug(Plug):
	node : FfFilletSrf = None
	pass
class LeftCurvePlug(Plug):
	node : FfFilletSrf = None
	pass
class OutputSurfacePlug(Plug):
	node : FfFilletSrf = None
	pass
class PositionTolerancePlug(Plug):
	node : FfFilletSrf = None
	pass
class RightCurvePlug(Plug):
	node : FfFilletSrf = None
	pass
class TangentTolerancePlug(Plug):
	node : FfFilletSrf = None
	pass
# endregion


# define node class
class FfFilletSrf(AbstractBaseCreate):
	bias_ : BiasPlug = PlugDescriptor("bias")
	depth_ : DepthPlug = PlugDescriptor("depth")
	leftCurve_ : LeftCurvePlug = PlugDescriptor("leftCurve")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	positionTolerance_ : PositionTolerancePlug = PlugDescriptor("positionTolerance")
	rightCurve_ : RightCurvePlug = PlugDescriptor("rightCurve")
	tangentTolerance_ : TangentTolerancePlug = PlugDescriptor("tangentTolerance")

	# node attributes

	typeName = "ffFilletSrf"
	typeIdInt = 1313228371
	nodeLeafClassAttrs = ["bias", "depth", "leftCurve", "outputSurface", "positionTolerance", "rightCurve", "tangentTolerance"]
	nodeLeafPlugs = ["bias", "depth", "leftCurve", "outputSurface", "positionTolerance", "rightCurve", "tangentTolerance"]
	pass

