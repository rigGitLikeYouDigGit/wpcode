

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
GeometryFilter = retriever.getNodeCls("GeometryFilter")
assert GeometryFilter
if T.TYPE_CHECKING:
	from .. import GeometryFilter

# add node doc



# region plug type defs
class AdjustedLowerBaseLatticeMatrixPlug(Plug):
	node : JointLattice = None
	pass
class AdjustedUpperBaseLatticeMatrixPlug(Plug):
	node : JointLattice = None
	pass
class BaseLatticeMatrixPlug(Plug):
	node : JointLattice = None
	pass
class BendMagnitudePlug(Plug):
	node : JointLattice = None
	pass
class BendVectorXPlug(Plug):
	parent : BendVectorPlug = PlugDescriptor("bendVector")
	node : JointLattice = None
	pass
class BendVectorYPlug(Plug):
	parent : BendVectorPlug = PlugDescriptor("bendVector")
	node : JointLattice = None
	pass
class BendVectorZPlug(Plug):
	parent : BendVectorPlug = PlugDescriptor("bendVector")
	node : JointLattice = None
	pass
class BendVectorPlug(Plug):
	bendVectorX_ : BendVectorXPlug = PlugDescriptor("bendVectorX")
	bx_ : BendVectorXPlug = PlugDescriptor("bendVectorX")
	bendVectorY_ : BendVectorYPlug = PlugDescriptor("bendVectorY")
	by_ : BendVectorYPlug = PlugDescriptor("bendVectorY")
	bendVectorZ_ : BendVectorZPlug = PlugDescriptor("bendVectorZ")
	bz_ : BendVectorZPlug = PlugDescriptor("bendVectorZ")
	node : JointLattice = None
	pass
class CreasingPlug(Plug):
	node : JointLattice = None
	pass
class DeformedLatticeMatrixPlug(Plug):
	node : JointLattice = None
	pass
class InitialLowerMatrixPlug(Plug):
	node : JointLattice = None
	pass
class InitialUpperMatrixPlug(Plug):
	node : JointLattice = None
	pass
class LengthInPlug(Plug):
	node : JointLattice = None
	pass
class LengthOutPlug(Plug):
	node : JointLattice = None
	pass
class LowerMatrixPlug(Plug):
	node : JointLattice = None
	pass
class RoundingPlug(Plug):
	node : JointLattice = None
	pass
class UpperMatrixPlug(Plug):
	node : JointLattice = None
	pass
class WidthLeftPlug(Plug):
	node : JointLattice = None
	pass
class WidthRightPlug(Plug):
	node : JointLattice = None
	pass
# endregion


# define node class
class JointLattice(GeometryFilter):
	adjustedLowerBaseLatticeMatrix_ : AdjustedLowerBaseLatticeMatrixPlug = PlugDescriptor("adjustedLowerBaseLatticeMatrix")
	adjustedUpperBaseLatticeMatrix_ : AdjustedUpperBaseLatticeMatrixPlug = PlugDescriptor("adjustedUpperBaseLatticeMatrix")
	baseLatticeMatrix_ : BaseLatticeMatrixPlug = PlugDescriptor("baseLatticeMatrix")
	bendMagnitude_ : BendMagnitudePlug = PlugDescriptor("bendMagnitude")
	bendVectorX_ : BendVectorXPlug = PlugDescriptor("bendVectorX")
	bendVectorY_ : BendVectorYPlug = PlugDescriptor("bendVectorY")
	bendVectorZ_ : BendVectorZPlug = PlugDescriptor("bendVectorZ")
	bendVector_ : BendVectorPlug = PlugDescriptor("bendVector")
	creasing_ : CreasingPlug = PlugDescriptor("creasing")
	deformedLatticeMatrix_ : DeformedLatticeMatrixPlug = PlugDescriptor("deformedLatticeMatrix")
	initialLowerMatrix_ : InitialLowerMatrixPlug = PlugDescriptor("initialLowerMatrix")
	initialUpperMatrix_ : InitialUpperMatrixPlug = PlugDescriptor("initialUpperMatrix")
	lengthIn_ : LengthInPlug = PlugDescriptor("lengthIn")
	lengthOut_ : LengthOutPlug = PlugDescriptor("lengthOut")
	lowerMatrix_ : LowerMatrixPlug = PlugDescriptor("lowerMatrix")
	rounding_ : RoundingPlug = PlugDescriptor("rounding")
	upperMatrix_ : UpperMatrixPlug = PlugDescriptor("upperMatrix")
	widthLeft_ : WidthLeftPlug = PlugDescriptor("widthLeft")
	widthRight_ : WidthRightPlug = PlugDescriptor("widthRight")

	# node attributes

	typeName = "jointLattice"
	typeIdInt = 1178748236
	nodeLeafClassAttrs = ["adjustedLowerBaseLatticeMatrix", "adjustedUpperBaseLatticeMatrix", "baseLatticeMatrix", "bendMagnitude", "bendVectorX", "bendVectorY", "bendVectorZ", "bendVector", "creasing", "deformedLatticeMatrix", "initialLowerMatrix", "initialUpperMatrix", "lengthIn", "lengthOut", "lowerMatrix", "rounding", "upperMatrix", "widthLeft", "widthRight"]
	nodeLeafPlugs = ["adjustedLowerBaseLatticeMatrix", "adjustedUpperBaseLatticeMatrix", "baseLatticeMatrix", "bendMagnitude", "bendVector", "creasing", "deformedLatticeMatrix", "initialLowerMatrix", "initialUpperMatrix", "lengthIn", "lengthOut", "lowerMatrix", "rounding", "upperMatrix", "widthLeft", "widthRight"]
	pass

