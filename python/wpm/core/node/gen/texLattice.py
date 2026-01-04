

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : TexLattice = None
	pass
class BoundingBoxLeftPlug(Plug):
	parent : BoundingBoxInfPlug = PlugDescriptor("boundingBoxInf")
	node : TexLattice = None
	pass
class BoundingBoxTopPlug(Plug):
	parent : BoundingBoxInfPlug = PlugDescriptor("boundingBoxInf")
	node : TexLattice = None
	pass
class BoundingBoxInfPlug(Plug):
	boundingBoxLeft_ : BoundingBoxLeftPlug = PlugDescriptor("boundingBoxLeft")
	bbxl_ : BoundingBoxLeftPlug = PlugDescriptor("boundingBoxLeft")
	boundingBoxTop_ : BoundingBoxTopPlug = PlugDescriptor("boundingBoxTop")
	bbxt_ : BoundingBoxTopPlug = PlugDescriptor("boundingBoxTop")
	node : TexLattice = None
	pass
class BoundingBoxBottomPlug(Plug):
	parent : BoundingBoxSupPlug = PlugDescriptor("boundingBoxSup")
	node : TexLattice = None
	pass
class BoundingBoxRightPlug(Plug):
	parent : BoundingBoxSupPlug = PlugDescriptor("boundingBoxSup")
	node : TexLattice = None
	pass
class BoundingBoxSupPlug(Plug):
	boundingBoxBottom_ : BoundingBoxBottomPlug = PlugDescriptor("boundingBoxBottom")
	bbxb_ : BoundingBoxBottomPlug = PlugDescriptor("boundingBoxBottom")
	boundingBoxRight_ : BoundingBoxRightPlug = PlugDescriptor("boundingBoxRight")
	bbxr_ : BoundingBoxRightPlug = PlugDescriptor("boundingBoxRight")
	node : TexLattice = None
	pass
class LatticeHeightPlug(Plug):
	node : TexLattice = None
	pass
class LatticePointXPlug(Plug):
	parent : LatticePointPlug = PlugDescriptor("latticePoint")
	node : TexLattice = None
	pass
class LatticePointYPlug(Plug):
	parent : LatticePointPlug = PlugDescriptor("latticePoint")
	node : TexLattice = None
	pass
class LatticePointPlug(Plug):
	latticePointX_ : LatticePointXPlug = PlugDescriptor("latticePointX")
	lpx_ : LatticePointXPlug = PlugDescriptor("latticePointX")
	latticePointY_ : LatticePointYPlug = PlugDescriptor("latticePointY")
	lpy_ : LatticePointYPlug = PlugDescriptor("latticePointY")
	node : TexLattice = None
	pass
class LatticeWidthPlug(Plug):
	node : TexLattice = None
	pass
# endregion


# define node class
class TexLattice(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	boundingBoxLeft_ : BoundingBoxLeftPlug = PlugDescriptor("boundingBoxLeft")
	boundingBoxTop_ : BoundingBoxTopPlug = PlugDescriptor("boundingBoxTop")
	boundingBoxInf_ : BoundingBoxInfPlug = PlugDescriptor("boundingBoxInf")
	boundingBoxBottom_ : BoundingBoxBottomPlug = PlugDescriptor("boundingBoxBottom")
	boundingBoxRight_ : BoundingBoxRightPlug = PlugDescriptor("boundingBoxRight")
	boundingBoxSup_ : BoundingBoxSupPlug = PlugDescriptor("boundingBoxSup")
	latticeHeight_ : LatticeHeightPlug = PlugDescriptor("latticeHeight")
	latticePointX_ : LatticePointXPlug = PlugDescriptor("latticePointX")
	latticePointY_ : LatticePointYPlug = PlugDescriptor("latticePointY")
	latticePoint_ : LatticePointPlug = PlugDescriptor("latticePoint")
	latticeWidth_ : LatticeWidthPlug = PlugDescriptor("latticeWidth")

	# node attributes

	typeName = "texLattice"
	apiTypeInt = 200
	apiTypeStr = "kTexLattice"
	typeIdInt = 1415072852
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "boundingBoxLeft", "boundingBoxTop", "boundingBoxInf", "boundingBoxBottom", "boundingBoxRight", "boundingBoxSup", "latticeHeight", "latticePointX", "latticePointY", "latticePoint", "latticeWidth"]
	nodeLeafPlugs = ["binMembership", "boundingBoxInf", "boundingBoxSup", "latticeHeight", "latticePoint", "latticeWidth"]
	pass

