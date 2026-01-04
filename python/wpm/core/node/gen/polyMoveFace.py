

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyMoveEdge = Catalogue.PolyMoveEdge
else:
	from .. import retriever
	PolyMoveEdge = retriever.getNodeCls("PolyMoveEdge")
	assert PolyMoveEdge

# add node doc



# region plug type defs
class AttractionPlug(Plug):
	node : PolyMoveFace = None
	pass
class GravityXPlug(Plug):
	parent : GravityPlug = PlugDescriptor("gravity")
	node : PolyMoveFace = None
	pass
class GravityYPlug(Plug):
	parent : GravityPlug = PlugDescriptor("gravity")
	node : PolyMoveFace = None
	pass
class GravityZPlug(Plug):
	parent : GravityPlug = PlugDescriptor("gravity")
	node : PolyMoveFace = None
	pass
class GravityPlug(Plug):
	gravityX_ : GravityXPlug = PlugDescriptor("gravityX")
	gx_ : GravityXPlug = PlugDescriptor("gravityX")
	gravityY_ : GravityYPlug = PlugDescriptor("gravityY")
	gy_ : GravityYPlug = PlugDescriptor("gravityY")
	gravityZ_ : GravityZPlug = PlugDescriptor("gravityZ")
	gz_ : GravityZPlug = PlugDescriptor("gravityZ")
	node : PolyMoveFace = None
	pass
class MagnXPlug(Plug):
	parent : MagnetPlug = PlugDescriptor("magnet")
	node : PolyMoveFace = None
	pass
class MagnYPlug(Plug):
	parent : MagnetPlug = PlugDescriptor("magnet")
	node : PolyMoveFace = None
	pass
class MagnZPlug(Plug):
	parent : MagnetPlug = PlugDescriptor("magnet")
	node : PolyMoveFace = None
	pass
class MagnetPlug(Plug):
	magnX_ : MagnXPlug = PlugDescriptor("magnX")
	mx_ : MagnXPlug = PlugDescriptor("magnX")
	magnY_ : MagnYPlug = PlugDescriptor("magnY")
	my_ : MagnYPlug = PlugDescriptor("magnY")
	magnZ_ : MagnZPlug = PlugDescriptor("magnZ")
	mz_ : MagnZPlug = PlugDescriptor("magnZ")
	node : PolyMoveFace = None
	pass
class OffsetPlug(Plug):
	node : PolyMoveFace = None
	pass
class WeightPlug(Plug):
	node : PolyMoveFace = None
	pass
# endregion


# define node class
class PolyMoveFace(PolyMoveEdge):
	attraction_ : AttractionPlug = PlugDescriptor("attraction")
	gravityX_ : GravityXPlug = PlugDescriptor("gravityX")
	gravityY_ : GravityYPlug = PlugDescriptor("gravityY")
	gravityZ_ : GravityZPlug = PlugDescriptor("gravityZ")
	gravity_ : GravityPlug = PlugDescriptor("gravity")
	magnX_ : MagnXPlug = PlugDescriptor("magnX")
	magnY_ : MagnYPlug = PlugDescriptor("magnY")
	magnZ_ : MagnZPlug = PlugDescriptor("magnZ")
	magnet_ : MagnetPlug = PlugDescriptor("magnet")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	weight_ : WeightPlug = PlugDescriptor("weight")

	# node attributes

	typeName = "polyMoveFace"
	typeIdInt = 1347243846
	nodeLeafClassAttrs = ["attraction", "gravityX", "gravityY", "gravityZ", "gravity", "magnX", "magnY", "magnZ", "magnet", "offset", "weight"]
	nodeLeafPlugs = ["attraction", "gravity", "magnet", "offset", "weight"]
	pass

