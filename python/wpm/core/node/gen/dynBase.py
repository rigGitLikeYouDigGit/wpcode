

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Transform = Catalogue.Transform
else:
	from .. import retriever
	Transform = retriever.getNodeCls("Transform")
	assert Transform

# add node doc



# region plug type defs
class FromWherePlug(Plug):
	node : DynBase = None
	pass
class OwnerPlug(Plug):
	node : DynBase = None
	pass
class OwnerCentroidXPlug(Plug):
	parent : OwnerCentroidPlug = PlugDescriptor("ownerCentroid")
	node : DynBase = None
	pass
class OwnerCentroidYPlug(Plug):
	parent : OwnerCentroidPlug = PlugDescriptor("ownerCentroid")
	node : DynBase = None
	pass
class OwnerCentroidZPlug(Plug):
	parent : OwnerCentroidPlug = PlugDescriptor("ownerCentroid")
	node : DynBase = None
	pass
class OwnerCentroidPlug(Plug):
	ownerCentroidX_ : OwnerCentroidXPlug = PlugDescriptor("ownerCentroidX")
	ocx_ : OwnerCentroidXPlug = PlugDescriptor("ownerCentroidX")
	ownerCentroidY_ : OwnerCentroidYPlug = PlugDescriptor("ownerCentroidY")
	ocy_ : OwnerCentroidYPlug = PlugDescriptor("ownerCentroidY")
	ownerCentroidZ_ : OwnerCentroidZPlug = PlugDescriptor("ownerCentroidZ")
	ocz_ : OwnerCentroidZPlug = PlugDescriptor("ownerCentroidZ")
	node : DynBase = None
	pass
class OwnerPosDataPlug(Plug):
	node : DynBase = None
	pass
class OwnerVelDataPlug(Plug):
	node : DynBase = None
	pass
class PositionalPlug(Plug):
	node : DynBase = None
	pass
class SubsetIdPlug(Plug):
	node : DynBase = None
	pass
# endregion


# define node class
class DynBase(Transform):
	fromWhere_ : FromWherePlug = PlugDescriptor("fromWhere")
	owner_ : OwnerPlug = PlugDescriptor("owner")
	ownerCentroidX_ : OwnerCentroidXPlug = PlugDescriptor("ownerCentroidX")
	ownerCentroidY_ : OwnerCentroidYPlug = PlugDescriptor("ownerCentroidY")
	ownerCentroidZ_ : OwnerCentroidZPlug = PlugDescriptor("ownerCentroidZ")
	ownerCentroid_ : OwnerCentroidPlug = PlugDescriptor("ownerCentroid")
	ownerPosData_ : OwnerPosDataPlug = PlugDescriptor("ownerPosData")
	ownerVelData_ : OwnerVelDataPlug = PlugDescriptor("ownerVelData")
	positional_ : PositionalPlug = PlugDescriptor("positional")
	subsetId_ : SubsetIdPlug = PlugDescriptor("subsetId")

	# node attributes

	typeName = "dynBase"
	typeIdInt = 1497462136
	nodeLeafClassAttrs = ["fromWhere", "owner", "ownerCentroidX", "ownerCentroidY", "ownerCentroidZ", "ownerCentroid", "ownerPosData", "ownerVelData", "positional", "subsetId"]
	nodeLeafPlugs = ["fromWhere", "owner", "ownerCentroid", "ownerPosData", "ownerVelData", "positional", "subsetId"]
	pass

