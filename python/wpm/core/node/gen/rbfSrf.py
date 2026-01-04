

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AbstractBaseCreate = Catalogue.AbstractBaseCreate
else:
	from .. import retriever
	AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
	assert AbstractBaseCreate

# add node doc



# region plug type defs
class OutputSurfacePlug(Plug):
	node : RbfSrf = None
	pass
class PositionTolerancePlug(Plug):
	node : RbfSrf = None
	pass
class PrimaryRadiusPlug(Plug):
	node : RbfSrf = None
	pass
class PrimarySurfacePlug(Plug):
	node : RbfSrf = None
	pass
class SecondaryRadiusPlug(Plug):
	node : RbfSrf = None
	pass
class SecondarySurfacePlug(Plug):
	node : RbfSrf = None
	pass
class TangentTolerancePlug(Plug):
	node : RbfSrf = None
	pass
class TrimCurveOnPrimaryPlug(Plug):
	node : RbfSrf = None
	pass
class TrimCurveOnSecondaryPlug(Plug):
	node : RbfSrf = None
	pass
# endregion


# define node class
class RbfSrf(AbstractBaseCreate):
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	positionTolerance_ : PositionTolerancePlug = PlugDescriptor("positionTolerance")
	primaryRadius_ : PrimaryRadiusPlug = PlugDescriptor("primaryRadius")
	primarySurface_ : PrimarySurfacePlug = PlugDescriptor("primarySurface")
	secondaryRadius_ : SecondaryRadiusPlug = PlugDescriptor("secondaryRadius")
	secondarySurface_ : SecondarySurfacePlug = PlugDescriptor("secondarySurface")
	tangentTolerance_ : TangentTolerancePlug = PlugDescriptor("tangentTolerance")
	trimCurveOnPrimary_ : TrimCurveOnPrimaryPlug = PlugDescriptor("trimCurveOnPrimary")
	trimCurveOnSecondary_ : TrimCurveOnSecondaryPlug = PlugDescriptor("trimCurveOnSecondary")

	# node attributes

	typeName = "rbfSrf"
	typeIdInt = 1314013766
	nodeLeafClassAttrs = ["outputSurface", "positionTolerance", "primaryRadius", "primarySurface", "secondaryRadius", "secondarySurface", "tangentTolerance", "trimCurveOnPrimary", "trimCurveOnSecondary"]
	nodeLeafPlugs = ["outputSurface", "positionTolerance", "primaryRadius", "primarySurface", "secondaryRadius", "secondarySurface", "tangentTolerance", "trimCurveOnPrimary", "trimCurveOnSecondary"]
	pass

