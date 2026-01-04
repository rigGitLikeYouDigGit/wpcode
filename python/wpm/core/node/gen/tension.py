

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	WeightGeometryFilter = Catalogue.WeightGeometryFilter
else:
	from .. import retriever
	WeightGeometryFilter = retriever.getNodeCls("WeightGeometryFilter")
	assert WeightGeometryFilter

# add node doc



# region plug type defs
class BendStrengthPlug(Plug):
	node : Tension = None
	pass
class CacheBindPositionsPlug(Plug):
	parent : CachePlug = PlugDescriptor("cache")
	node : Tension = None
	pass
class CachePlug(Plug):
	cacheBindPositions_ : CacheBindPositionsPlug = PlugDescriptor("cacheBindPositions")
	cbp_ : CacheBindPositionsPlug = PlugDescriptor("cacheBindPositions")
	node : Tension = None
	pass
class CacheSetupPlug(Plug):
	node : Tension = None
	pass
class InwardConstraintPlug(Plug):
	node : Tension = None
	pass
class OutwardConstraintPlug(Plug):
	node : Tension = None
	pass
class PinBorderVerticesPlug(Plug):
	node : Tension = None
	pass
class RelativePlug(Plug):
	node : Tension = None
	pass
class ShearStrengthPlug(Plug):
	node : Tension = None
	pass
class SmoothingIterationsPlug(Plug):
	node : Tension = None
	pass
class SmoothingStepPlug(Plug):
	node : Tension = None
	pass
class SquashConstraintPlug(Plug):
	node : Tension = None
	pass
class StretchConstraintPlug(Plug):
	node : Tension = None
	pass
# endregion


# define node class
class Tension(WeightGeometryFilter):
	bendStrength_ : BendStrengthPlug = PlugDescriptor("bendStrength")
	cacheBindPositions_ : CacheBindPositionsPlug = PlugDescriptor("cacheBindPositions")
	cache_ : CachePlug = PlugDescriptor("cache")
	cacheSetup_ : CacheSetupPlug = PlugDescriptor("cacheSetup")
	inwardConstraint_ : InwardConstraintPlug = PlugDescriptor("inwardConstraint")
	outwardConstraint_ : OutwardConstraintPlug = PlugDescriptor("outwardConstraint")
	pinBorderVertices_ : PinBorderVerticesPlug = PlugDescriptor("pinBorderVertices")
	relative_ : RelativePlug = PlugDescriptor("relative")
	shearStrength_ : ShearStrengthPlug = PlugDescriptor("shearStrength")
	smoothingIterations_ : SmoothingIterationsPlug = PlugDescriptor("smoothingIterations")
	smoothingStep_ : SmoothingStepPlug = PlugDescriptor("smoothingStep")
	squashConstraint_ : SquashConstraintPlug = PlugDescriptor("squashConstraint")
	stretchConstraint_ : StretchConstraintPlug = PlugDescriptor("stretchConstraint")

	# node attributes

	typeName = "tension"
	apiTypeInt = 351
	apiTypeStr = "kTension"
	typeIdInt = 1413828179
	MFnCls = om.MFnGeometryFilter
	nodeLeafClassAttrs = ["bendStrength", "cacheBindPositions", "cache", "cacheSetup", "inwardConstraint", "outwardConstraint", "pinBorderVertices", "relative", "shearStrength", "smoothingIterations", "smoothingStep", "squashConstraint", "stretchConstraint"]
	nodeLeafPlugs = ["bendStrength", "cache", "cacheSetup", "inwardConstraint", "outwardConstraint", "pinBorderVertices", "relative", "shearStrength", "smoothingIterations", "smoothingStep", "squashConstraint", "stretchConstraint"]
	pass

