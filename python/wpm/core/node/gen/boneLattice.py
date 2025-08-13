

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
class AdjustedUpperBaseLatticeMatrixPlug(Plug):
	node : BoneLattice = None
	pass
class BaseLatticeMatrixPlug(Plug):
	node : BoneLattice = None
	pass
class BendMagnitudePlug(Plug):
	node : BoneLattice = None
	pass
class BendVectorXPlug(Plug):
	parent : BendVectorPlug = PlugDescriptor("bendVector")
	node : BoneLattice = None
	pass
class BendVectorYPlug(Plug):
	parent : BendVectorPlug = PlugDescriptor("bendVector")
	node : BoneLattice = None
	pass
class BendVectorZPlug(Plug):
	parent : BendVectorPlug = PlugDescriptor("bendVector")
	node : BoneLattice = None
	pass
class BendVectorPlug(Plug):
	bendVectorX_ : BendVectorXPlug = PlugDescriptor("bendVectorX")
	bx_ : BendVectorXPlug = PlugDescriptor("bendVectorX")
	bendVectorY_ : BendVectorYPlug = PlugDescriptor("bendVectorY")
	by_ : BendVectorYPlug = PlugDescriptor("bendVectorY")
	bendVectorZ_ : BendVectorZPlug = PlugDescriptor("bendVectorZ")
	bz_ : BendVectorZPlug = PlugDescriptor("bendVectorZ")
	node : BoneLattice = None
	pass
class BicepPlug(Plug):
	node : BoneLattice = None
	pass
class DeformedLatticeMatrixPlug(Plug):
	node : BoneLattice = None
	pass
class InitialUpperMatrixPlug(Plug):
	node : BoneLattice = None
	pass
class LengthInPlug(Plug):
	node : BoneLattice = None
	pass
class LengthOutPlug(Plug):
	node : BoneLattice = None
	pass
class TricepPlug(Plug):
	node : BoneLattice = None
	pass
class UpperMatrixPlug(Plug):
	node : BoneLattice = None
	pass
class WidthLeftPlug(Plug):
	node : BoneLattice = None
	pass
class WidthRightPlug(Plug):
	node : BoneLattice = None
	pass
# endregion


# define node class
class BoneLattice(GeometryFilter):
	adjustedUpperBaseLatticeMatrix_ : AdjustedUpperBaseLatticeMatrixPlug = PlugDescriptor("adjustedUpperBaseLatticeMatrix")
	baseLatticeMatrix_ : BaseLatticeMatrixPlug = PlugDescriptor("baseLatticeMatrix")
	bendMagnitude_ : BendMagnitudePlug = PlugDescriptor("bendMagnitude")
	bendVectorX_ : BendVectorXPlug = PlugDescriptor("bendVectorX")
	bendVectorY_ : BendVectorYPlug = PlugDescriptor("bendVectorY")
	bendVectorZ_ : BendVectorZPlug = PlugDescriptor("bendVectorZ")
	bendVector_ : BendVectorPlug = PlugDescriptor("bendVector")
	bicep_ : BicepPlug = PlugDescriptor("bicep")
	deformedLatticeMatrix_ : DeformedLatticeMatrixPlug = PlugDescriptor("deformedLatticeMatrix")
	initialUpperMatrix_ : InitialUpperMatrixPlug = PlugDescriptor("initialUpperMatrix")
	lengthIn_ : LengthInPlug = PlugDescriptor("lengthIn")
	lengthOut_ : LengthOutPlug = PlugDescriptor("lengthOut")
	tricep_ : TricepPlug = PlugDescriptor("tricep")
	upperMatrix_ : UpperMatrixPlug = PlugDescriptor("upperMatrix")
	widthLeft_ : WidthLeftPlug = PlugDescriptor("widthLeft")
	widthRight_ : WidthRightPlug = PlugDescriptor("widthRight")

	# node attributes

	typeName = "boneLattice"
	typeIdInt = 1178752332
	pass

