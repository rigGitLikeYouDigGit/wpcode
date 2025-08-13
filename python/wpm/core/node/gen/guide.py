

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
class AutoGuidePlug(Plug):
	node : Guide = None
	pass
class BendAnglePlug(Plug):
	node : Guide = None
	pass
class BendMagnitudePlug(Plug):
	node : Guide = None
	pass
class BendVectorXPlug(Plug):
	parent : BendVectorPlug = PlugDescriptor("bendVector")
	node : Guide = None
	pass
class BendVectorYPlug(Plug):
	parent : BendVectorPlug = PlugDescriptor("bendVector")
	node : Guide = None
	pass
class BendVectorZPlug(Plug):
	parent : BendVectorPlug = PlugDescriptor("bendVector")
	node : Guide = None
	pass
class BendVectorPlug(Plug):
	bendVectorX_ : BendVectorXPlug = PlugDescriptor("bendVectorX")
	bx_ : BendVectorXPlug = PlugDescriptor("bendVectorX")
	bendVectorY_ : BendVectorYPlug = PlugDescriptor("bendVectorY")
	by_ : BendVectorYPlug = PlugDescriptor("bendVectorY")
	bendVectorZ_ : BendVectorZPlug = PlugDescriptor("bendVectorZ")
	bz_ : BendVectorZPlug = PlugDescriptor("bendVectorZ")
	node : Guide = None
	pass
class BinMembershipPlug(Plug):
	node : Guide = None
	pass
class JointAboveMatrixPlug(Plug):
	node : Guide = None
	pass
class JointBelowMatrixPlug(Plug):
	node : Guide = None
	pass
class JointGuideAxisPlug(Plug):
	node : Guide = None
	pass
class JointXformMatrixPlug(Plug):
	node : Guide = None
	pass
class MaxXYZPlug(Plug):
	node : Guide = None
	pass
class RotateXPlug(Plug):
	node : Guide = None
	pass
class RotateYPlug(Plug):
	node : Guide = None
	pass
class RotateZPlug(Plug):
	node : Guide = None
	pass
# endregion


# define node class
class Guide(_BASE_):
	autoGuide_ : AutoGuidePlug = PlugDescriptor("autoGuide")
	bendAngle_ : BendAnglePlug = PlugDescriptor("bendAngle")
	bendMagnitude_ : BendMagnitudePlug = PlugDescriptor("bendMagnitude")
	bendVectorX_ : BendVectorXPlug = PlugDescriptor("bendVectorX")
	bendVectorY_ : BendVectorYPlug = PlugDescriptor("bendVectorY")
	bendVectorZ_ : BendVectorZPlug = PlugDescriptor("bendVectorZ")
	bendVector_ : BendVectorPlug = PlugDescriptor("bendVector")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	jointAboveMatrix_ : JointAboveMatrixPlug = PlugDescriptor("jointAboveMatrix")
	jointBelowMatrix_ : JointBelowMatrixPlug = PlugDescriptor("jointBelowMatrix")
	jointGuideAxis_ : JointGuideAxisPlug = PlugDescriptor("jointGuideAxis")
	jointXformMatrix_ : JointXformMatrixPlug = PlugDescriptor("jointXformMatrix")
	maxXYZ_ : MaxXYZPlug = PlugDescriptor("maxXYZ")
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")

	# node attributes

	typeName = "guide"
	apiTypeInt = 358
	apiTypeStr = "kGuide"
	typeIdInt = 1179080009
	MFnCls = om.MFnDependencyNode
	pass

