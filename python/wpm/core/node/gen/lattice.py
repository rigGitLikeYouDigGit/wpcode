

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ControlPoint = retriever.getNodeCls("ControlPoint")
assert ControlPoint
if T.TYPE_CHECKING:
	from .. import ControlPoint

# add node doc



# region plug type defs
class CachedPlug(Plug):
	node : Lattice = None
	pass
class DispLatticePlug(Plug):
	node : Lattice = None
	pass
class DispPointsPlug(Plug):
	node : Lattice = None
	pass
class DisplayControlPlug(Plug):
	node : Lattice = None
	pass
class LatticeInputPlug(Plug):
	node : Lattice = None
	pass
class LatticeOutputPlug(Plug):
	node : Lattice = None
	pass
class LatticePointMovedPlug(Plug):
	node : Lattice = None
	pass
class OriginXPlug(Plug):
	parent : OriginPlug = PlugDescriptor("origin")
	node : Lattice = None
	pass
class OriginYPlug(Plug):
	parent : OriginPlug = PlugDescriptor("origin")
	node : Lattice = None
	pass
class OriginZPlug(Plug):
	parent : OriginPlug = PlugDescriptor("origin")
	node : Lattice = None
	pass
class OriginPlug(Plug):
	originX_ : OriginXPlug = PlugDescriptor("originX")
	ox_ : OriginXPlug = PlugDescriptor("originX")
	originY_ : OriginYPlug = PlugDescriptor("originY")
	oy_ : OriginYPlug = PlugDescriptor("originY")
	originZ_ : OriginZPlug = PlugDescriptor("originZ")
	oz_ : OriginZPlug = PlugDescriptor("originZ")
	node : Lattice = None
	pass
class SDivisionsPlug(Plug):
	node : Lattice = None
	pass
class TDivisionsPlug(Plug):
	node : Lattice = None
	pass
class UDivisionsPlug(Plug):
	node : Lattice = None
	pass
class WorldLatticePlug(Plug):
	node : Lattice = None
	pass
# endregion


# define node class
class Lattice(ControlPoint):
	cached_ : CachedPlug = PlugDescriptor("cached")
	dispLattice_ : DispLatticePlug = PlugDescriptor("dispLattice")
	dispPoints_ : DispPointsPlug = PlugDescriptor("dispPoints")
	displayControl_ : DisplayControlPlug = PlugDescriptor("displayControl")
	latticeInput_ : LatticeInputPlug = PlugDescriptor("latticeInput")
	latticeOutput_ : LatticeOutputPlug = PlugDescriptor("latticeOutput")
	latticePointMoved_ : LatticePointMovedPlug = PlugDescriptor("latticePointMoved")
	originX_ : OriginXPlug = PlugDescriptor("originX")
	originY_ : OriginYPlug = PlugDescriptor("originY")
	originZ_ : OriginZPlug = PlugDescriptor("originZ")
	origin_ : OriginPlug = PlugDescriptor("origin")
	sDivisions_ : SDivisionsPlug = PlugDescriptor("sDivisions")
	tDivisions_ : TDivisionsPlug = PlugDescriptor("tDivisions")
	uDivisions_ : UDivisionsPlug = PlugDescriptor("uDivisions")
	worldLattice_ : WorldLatticePlug = PlugDescriptor("worldLattice")

	# node attributes

	typeName = "lattice"
	apiTypeInt = 279
	apiTypeStr = "kLattice"
	typeIdInt = 1179402580
	MFnCls = om.MFnDagNode
	pass

