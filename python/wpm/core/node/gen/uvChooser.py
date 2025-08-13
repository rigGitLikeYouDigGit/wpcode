

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
_BASE_ = retriever.getNodeCls("_BASE_")
assert _BASE_
if T.TYPE_CHECKING:
	from .. import _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : UvChooser = None
	pass
class InfoBitsPlug(Plug):
	node : UvChooser = None
	pass
class OutUPlug(Plug):
	parent : OutUvPlug = PlugDescriptor("outUv")
	node : UvChooser = None
	pass
class OutVPlug(Plug):
	parent : OutUvPlug = PlugDescriptor("outUv")
	node : UvChooser = None
	pass
class OutUvPlug(Plug):
	outU_ : OutUPlug = PlugDescriptor("outU")
	ou_ : OutUPlug = PlugDescriptor("outU")
	outV_ : OutVPlug = PlugDescriptor("outV")
	ov_ : OutVPlug = PlugDescriptor("outV")
	node : UvChooser = None
	pass
class OutVertexCameraOneXPlug(Plug):
	parent : OutVertexCameraOnePlug = PlugDescriptor("outVertexCameraOne")
	node : UvChooser = None
	pass
class OutVertexCameraOneYPlug(Plug):
	parent : OutVertexCameraOnePlug = PlugDescriptor("outVertexCameraOne")
	node : UvChooser = None
	pass
class OutVertexCameraOneZPlug(Plug):
	parent : OutVertexCameraOnePlug = PlugDescriptor("outVertexCameraOne")
	node : UvChooser = None
	pass
class OutVertexCameraOnePlug(Plug):
	outVertexCameraOneX_ : OutVertexCameraOneXPlug = PlugDescriptor("outVertexCameraOneX")
	o1x_ : OutVertexCameraOneXPlug = PlugDescriptor("outVertexCameraOneX")
	outVertexCameraOneY_ : OutVertexCameraOneYPlug = PlugDescriptor("outVertexCameraOneY")
	o1y_ : OutVertexCameraOneYPlug = PlugDescriptor("outVertexCameraOneY")
	outVertexCameraOneZ_ : OutVertexCameraOneZPlug = PlugDescriptor("outVertexCameraOneZ")
	o1z_ : OutVertexCameraOneZPlug = PlugDescriptor("outVertexCameraOneZ")
	node : UvChooser = None
	pass
class OutVertexUvOneUPlug(Plug):
	parent : OutVertexUvOnePlug = PlugDescriptor("outVertexUvOne")
	node : UvChooser = None
	pass
class OutVertexUvOneVPlug(Plug):
	parent : OutVertexUvOnePlug = PlugDescriptor("outVertexUvOne")
	node : UvChooser = None
	pass
class OutVertexUvOnePlug(Plug):
	outVertexUvOneU_ : OutVertexUvOneUPlug = PlugDescriptor("outVertexUvOneU")
	o1u_ : OutVertexUvOneUPlug = PlugDescriptor("outVertexUvOneU")
	outVertexUvOneV_ : OutVertexUvOneVPlug = PlugDescriptor("outVertexUvOneV")
	o1v_ : OutVertexUvOneVPlug = PlugDescriptor("outVertexUvOneV")
	node : UvChooser = None
	pass
class OutVertexUvThreeUPlug(Plug):
	parent : OutVertexUvThreePlug = PlugDescriptor("outVertexUvThree")
	node : UvChooser = None
	pass
class OutVertexUvThreeVPlug(Plug):
	parent : OutVertexUvThreePlug = PlugDescriptor("outVertexUvThree")
	node : UvChooser = None
	pass
class OutVertexUvThreePlug(Plug):
	outVertexUvThreeU_ : OutVertexUvThreeUPlug = PlugDescriptor("outVertexUvThreeU")
	o3u_ : OutVertexUvThreeUPlug = PlugDescriptor("outVertexUvThreeU")
	outVertexUvThreeV_ : OutVertexUvThreeVPlug = PlugDescriptor("outVertexUvThreeV")
	o3v_ : OutVertexUvThreeVPlug = PlugDescriptor("outVertexUvThreeV")
	node : UvChooser = None
	pass
class OutVertexUvTwoUPlug(Plug):
	parent : OutVertexUvTwoPlug = PlugDescriptor("outVertexUvTwo")
	node : UvChooser = None
	pass
class OutVertexUvTwoVPlug(Plug):
	parent : OutVertexUvTwoPlug = PlugDescriptor("outVertexUvTwo")
	node : UvChooser = None
	pass
class OutVertexUvTwoPlug(Plug):
	outVertexUvTwoU_ : OutVertexUvTwoUPlug = PlugDescriptor("outVertexUvTwoU")
	o2u_ : OutVertexUvTwoUPlug = PlugDescriptor("outVertexUvTwoU")
	outVertexUvTwoV_ : OutVertexUvTwoVPlug = PlugDescriptor("outVertexUvTwoV")
	o2v_ : OutVertexUvTwoVPlug = PlugDescriptor("outVertexUvTwoV")
	node : UvChooser = None
	pass
class SCoordPlug(Plug):
	parent : StCoordPlug = PlugDescriptor("stCoord")
	node : UvChooser = None
	pass
class TCoordPlug(Plug):
	parent : StCoordPlug = PlugDescriptor("stCoord")
	node : UvChooser = None
	pass
class StCoordPlug(Plug):
	sCoord_ : SCoordPlug = PlugDescriptor("sCoord")
	s_ : SCoordPlug = PlugDescriptor("sCoord")
	tCoord_ : TCoordPlug = PlugDescriptor("tCoord")
	t_ : TCoordPlug = PlugDescriptor("tCoord")
	node : UvChooser = None
	pass
class UCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : UvChooser = None
	pass
class VCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : UvChooser = None
	pass
class UvCoordPlug(Plug):
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	u_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	v_ : VCoordPlug = PlugDescriptor("vCoord")
	node : UvChooser = None
	pass
class UvSetsPlug(Plug):
	node : UvChooser = None
	pass
class VertexCameraOneXPlug(Plug):
	parent : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	node : UvChooser = None
	pass
class VertexCameraOneYPlug(Plug):
	parent : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	node : UvChooser = None
	pass
class VertexCameraOneZPlug(Plug):
	parent : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	node : UvChooser = None
	pass
class VertexCameraOnePlug(Plug):
	vertexCameraOneX_ : VertexCameraOneXPlug = PlugDescriptor("vertexCameraOneX")
	c1x_ : VertexCameraOneXPlug = PlugDescriptor("vertexCameraOneX")
	vertexCameraOneY_ : VertexCameraOneYPlug = PlugDescriptor("vertexCameraOneY")
	c1y_ : VertexCameraOneYPlug = PlugDescriptor("vertexCameraOneY")
	vertexCameraOneZ_ : VertexCameraOneZPlug = PlugDescriptor("vertexCameraOneZ")
	c1z_ : VertexCameraOneZPlug = PlugDescriptor("vertexCameraOneZ")
	node : UvChooser = None
	pass
class VertexStOneSPlug(Plug):
	parent : VertexStOnePlug = PlugDescriptor("vertexStOne")
	node : UvChooser = None
	pass
class VertexStOneTPlug(Plug):
	parent : VertexStOnePlug = PlugDescriptor("vertexStOne")
	node : UvChooser = None
	pass
class VertexStOnePlug(Plug):
	vertexStOneS_ : VertexStOneSPlug = PlugDescriptor("vertexStOneS")
	s1s_ : VertexStOneSPlug = PlugDescriptor("vertexStOneS")
	vertexStOneT_ : VertexStOneTPlug = PlugDescriptor("vertexStOneT")
	s1t_ : VertexStOneTPlug = PlugDescriptor("vertexStOneT")
	node : UvChooser = None
	pass
class VertexStThreeSPlug(Plug):
	parent : VertexStThreePlug = PlugDescriptor("vertexStThree")
	node : UvChooser = None
	pass
class VertexStThreeTPlug(Plug):
	parent : VertexStThreePlug = PlugDescriptor("vertexStThree")
	node : UvChooser = None
	pass
class VertexStThreePlug(Plug):
	vertexStThreeS_ : VertexStThreeSPlug = PlugDescriptor("vertexStThreeS")
	s3s_ : VertexStThreeSPlug = PlugDescriptor("vertexStThreeS")
	vertexStThreeT_ : VertexStThreeTPlug = PlugDescriptor("vertexStThreeT")
	s3t_ : VertexStThreeTPlug = PlugDescriptor("vertexStThreeT")
	node : UvChooser = None
	pass
class VertexStTwoSPlug(Plug):
	parent : VertexStTwoPlug = PlugDescriptor("vertexStTwo")
	node : UvChooser = None
	pass
class VertexStTwoTPlug(Plug):
	parent : VertexStTwoPlug = PlugDescriptor("vertexStTwo")
	node : UvChooser = None
	pass
class VertexStTwoPlug(Plug):
	vertexStTwoS_ : VertexStTwoSPlug = PlugDescriptor("vertexStTwoS")
	s2s_ : VertexStTwoSPlug = PlugDescriptor("vertexStTwoS")
	vertexStTwoT_ : VertexStTwoTPlug = PlugDescriptor("vertexStTwoT")
	s2t_ : VertexStTwoTPlug = PlugDescriptor("vertexStTwoT")
	node : UvChooser = None
	pass
class VertexUvOneUPlug(Plug):
	parent : VertexUvOnePlug = PlugDescriptor("vertexUvOne")
	node : UvChooser = None
	pass
class VertexUvOneVPlug(Plug):
	parent : VertexUvOnePlug = PlugDescriptor("vertexUvOne")
	node : UvChooser = None
	pass
class VertexUvOnePlug(Plug):
	vertexUvOneU_ : VertexUvOneUPlug = PlugDescriptor("vertexUvOneU")
	t1u_ : VertexUvOneUPlug = PlugDescriptor("vertexUvOneU")
	vertexUvOneV_ : VertexUvOneVPlug = PlugDescriptor("vertexUvOneV")
	t1v_ : VertexUvOneVPlug = PlugDescriptor("vertexUvOneV")
	node : UvChooser = None
	pass
class VertexUvThreeUPlug(Plug):
	parent : VertexUvThreePlug = PlugDescriptor("vertexUvThree")
	node : UvChooser = None
	pass
class VertexUvThreeVPlug(Plug):
	parent : VertexUvThreePlug = PlugDescriptor("vertexUvThree")
	node : UvChooser = None
	pass
class VertexUvThreePlug(Plug):
	vertexUvThreeU_ : VertexUvThreeUPlug = PlugDescriptor("vertexUvThreeU")
	t3u_ : VertexUvThreeUPlug = PlugDescriptor("vertexUvThreeU")
	vertexUvThreeV_ : VertexUvThreeVPlug = PlugDescriptor("vertexUvThreeV")
	t3v_ : VertexUvThreeVPlug = PlugDescriptor("vertexUvThreeV")
	node : UvChooser = None
	pass
class VertexUvTwoUPlug(Plug):
	parent : VertexUvTwoPlug = PlugDescriptor("vertexUvTwo")
	node : UvChooser = None
	pass
class VertexUvTwoVPlug(Plug):
	parent : VertexUvTwoPlug = PlugDescriptor("vertexUvTwo")
	node : UvChooser = None
	pass
class VertexUvTwoPlug(Plug):
	vertexUvTwoU_ : VertexUvTwoUPlug = PlugDescriptor("vertexUvTwoU")
	t2u_ : VertexUvTwoUPlug = PlugDescriptor("vertexUvTwoU")
	vertexUvTwoV_ : VertexUvTwoVPlug = PlugDescriptor("vertexUvTwoV")
	t2v_ : VertexUvTwoVPlug = PlugDescriptor("vertexUvTwoV")
	node : UvChooser = None
	pass
# endregion


# define node class
class UvChooser(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	infoBits_ : InfoBitsPlug = PlugDescriptor("infoBits")
	outU_ : OutUPlug = PlugDescriptor("outU")
	outV_ : OutVPlug = PlugDescriptor("outV")
	outUv_ : OutUvPlug = PlugDescriptor("outUv")
	outVertexCameraOneX_ : OutVertexCameraOneXPlug = PlugDescriptor("outVertexCameraOneX")
	outVertexCameraOneY_ : OutVertexCameraOneYPlug = PlugDescriptor("outVertexCameraOneY")
	outVertexCameraOneZ_ : OutVertexCameraOneZPlug = PlugDescriptor("outVertexCameraOneZ")
	outVertexCameraOne_ : OutVertexCameraOnePlug = PlugDescriptor("outVertexCameraOne")
	outVertexUvOneU_ : OutVertexUvOneUPlug = PlugDescriptor("outVertexUvOneU")
	outVertexUvOneV_ : OutVertexUvOneVPlug = PlugDescriptor("outVertexUvOneV")
	outVertexUvOne_ : OutVertexUvOnePlug = PlugDescriptor("outVertexUvOne")
	outVertexUvThreeU_ : OutVertexUvThreeUPlug = PlugDescriptor("outVertexUvThreeU")
	outVertexUvThreeV_ : OutVertexUvThreeVPlug = PlugDescriptor("outVertexUvThreeV")
	outVertexUvThree_ : OutVertexUvThreePlug = PlugDescriptor("outVertexUvThree")
	outVertexUvTwoU_ : OutVertexUvTwoUPlug = PlugDescriptor("outVertexUvTwoU")
	outVertexUvTwoV_ : OutVertexUvTwoVPlug = PlugDescriptor("outVertexUvTwoV")
	outVertexUvTwo_ : OutVertexUvTwoPlug = PlugDescriptor("outVertexUvTwo")
	sCoord_ : SCoordPlug = PlugDescriptor("sCoord")
	tCoord_ : TCoordPlug = PlugDescriptor("tCoord")
	stCoord_ : StCoordPlug = PlugDescriptor("stCoord")
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	uvCoord_ : UvCoordPlug = PlugDescriptor("uvCoord")
	uvSets_ : UvSetsPlug = PlugDescriptor("uvSets")
	vertexCameraOneX_ : VertexCameraOneXPlug = PlugDescriptor("vertexCameraOneX")
	vertexCameraOneY_ : VertexCameraOneYPlug = PlugDescriptor("vertexCameraOneY")
	vertexCameraOneZ_ : VertexCameraOneZPlug = PlugDescriptor("vertexCameraOneZ")
	vertexCameraOne_ : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	vertexStOneS_ : VertexStOneSPlug = PlugDescriptor("vertexStOneS")
	vertexStOneT_ : VertexStOneTPlug = PlugDescriptor("vertexStOneT")
	vertexStOne_ : VertexStOnePlug = PlugDescriptor("vertexStOne")
	vertexStThreeS_ : VertexStThreeSPlug = PlugDescriptor("vertexStThreeS")
	vertexStThreeT_ : VertexStThreeTPlug = PlugDescriptor("vertexStThreeT")
	vertexStThree_ : VertexStThreePlug = PlugDescriptor("vertexStThree")
	vertexStTwoS_ : VertexStTwoSPlug = PlugDescriptor("vertexStTwoS")
	vertexStTwoT_ : VertexStTwoTPlug = PlugDescriptor("vertexStTwoT")
	vertexStTwo_ : VertexStTwoPlug = PlugDescriptor("vertexStTwo")
	vertexUvOneU_ : VertexUvOneUPlug = PlugDescriptor("vertexUvOneU")
	vertexUvOneV_ : VertexUvOneVPlug = PlugDescriptor("vertexUvOneV")
	vertexUvOne_ : VertexUvOnePlug = PlugDescriptor("vertexUvOne")
	vertexUvThreeU_ : VertexUvThreeUPlug = PlugDescriptor("vertexUvThreeU")
	vertexUvThreeV_ : VertexUvThreeVPlug = PlugDescriptor("vertexUvThreeV")
	vertexUvThree_ : VertexUvThreePlug = PlugDescriptor("vertexUvThree")
	vertexUvTwoU_ : VertexUvTwoUPlug = PlugDescriptor("vertexUvTwoU")
	vertexUvTwoV_ : VertexUvTwoVPlug = PlugDescriptor("vertexUvTwoV")
	vertexUvTwo_ : VertexUvTwoPlug = PlugDescriptor("vertexUvTwo")

	# node attributes

	typeName = "uvChooser"
	apiTypeInt = 797
	apiTypeStr = "kUvChooser"
	typeIdInt = 1431716680
	MFnCls = om.MFnDependencyNode
	pass

