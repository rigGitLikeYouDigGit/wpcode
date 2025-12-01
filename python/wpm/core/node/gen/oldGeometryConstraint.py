

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
	node : OldGeometryConstraint = None
	pass
class GeometryPlug(Plug):
	node : OldGeometryConstraint = None
	pass
class InputMatrixPlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : OldGeometryConstraint = None
	pass
class InputRotPivotXPlug(Plug):
	parent : InputRotPivotPlug = PlugDescriptor("inputRotPivot")
	node : OldGeometryConstraint = None
	pass
class InputRotPivotYPlug(Plug):
	parent : InputRotPivotPlug = PlugDescriptor("inputRotPivot")
	node : OldGeometryConstraint = None
	pass
class InputRotPivotZPlug(Plug):
	parent : InputRotPivotPlug = PlugDescriptor("inputRotPivot")
	node : OldGeometryConstraint = None
	pass
class InputRotPivotPlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	inputRotPivotX_ : InputRotPivotXPlug = PlugDescriptor("inputRotPivotX")
	irpx_ : InputRotPivotXPlug = PlugDescriptor("inputRotPivotX")
	inputRotPivotY_ : InputRotPivotYPlug = PlugDescriptor("inputRotPivotY")
	irpy_ : InputRotPivotYPlug = PlugDescriptor("inputRotPivotY")
	inputRotPivotZ_ : InputRotPivotZPlug = PlugDescriptor("inputRotPivotZ")
	irpz_ : InputRotPivotZPlug = PlugDescriptor("inputRotPivotZ")
	node : OldGeometryConstraint = None
	pass
class InputRotTransXPlug(Plug):
	parent : InputRotTransPlug = PlugDescriptor("inputRotTrans")
	node : OldGeometryConstraint = None
	pass
class InputRotTransYPlug(Plug):
	parent : InputRotTransPlug = PlugDescriptor("inputRotTrans")
	node : OldGeometryConstraint = None
	pass
class InputRotTransZPlug(Plug):
	parent : InputRotTransPlug = PlugDescriptor("inputRotTrans")
	node : OldGeometryConstraint = None
	pass
class InputRotTransPlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	inputRotTransX_ : InputRotTransXPlug = PlugDescriptor("inputRotTransX")
	irtx_ : InputRotTransXPlug = PlugDescriptor("inputRotTransX")
	inputRotTransY_ : InputRotTransYPlug = PlugDescriptor("inputRotTransY")
	irty_ : InputRotTransYPlug = PlugDescriptor("inputRotTransY")
	inputRotTransZ_ : InputRotTransZPlug = PlugDescriptor("inputRotTransZ")
	irtz_ : InputRotTransZPlug = PlugDescriptor("inputRotTransZ")
	node : OldGeometryConstraint = None
	pass
class InputTransXPlug(Plug):
	parent : InputTransPlug = PlugDescriptor("inputTrans")
	node : OldGeometryConstraint = None
	pass
class InputTransYPlug(Plug):
	parent : InputTransPlug = PlugDescriptor("inputTrans")
	node : OldGeometryConstraint = None
	pass
class InputTransZPlug(Plug):
	parent : InputTransPlug = PlugDescriptor("inputTrans")
	node : OldGeometryConstraint = None
	pass
class InputTransPlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	inputTransX_ : InputTransXPlug = PlugDescriptor("inputTransX")
	itx_ : InputTransXPlug = PlugDescriptor("inputTransX")
	inputTransY_ : InputTransYPlug = PlugDescriptor("inputTransY")
	ity_ : InputTransYPlug = PlugDescriptor("inputTransY")
	inputTransZ_ : InputTransZPlug = PlugDescriptor("inputTransZ")
	itz_ : InputTransZPlug = PlugDescriptor("inputTransZ")
	node : OldGeometryConstraint = None
	pass
class WeightPlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : OldGeometryConstraint = None
	pass
class InputPlug(Plug):
	inputMatrix_ : InputMatrixPlug = PlugDescriptor("inputMatrix")
	im_ : InputMatrixPlug = PlugDescriptor("inputMatrix")
	inputRotPivot_ : InputRotPivotPlug = PlugDescriptor("inputRotPivot")
	irp_ : InputRotPivotPlug = PlugDescriptor("inputRotPivot")
	inputRotTrans_ : InputRotTransPlug = PlugDescriptor("inputRotTrans")
	irt_ : InputRotTransPlug = PlugDescriptor("inputRotTrans")
	inputTrans_ : InputTransPlug = PlugDescriptor("inputTrans")
	it_ : InputTransPlug = PlugDescriptor("inputTrans")
	weight_ : WeightPlug = PlugDescriptor("weight")
	w_ : WeightPlug = PlugDescriptor("weight")
	node : OldGeometryConstraint = None
	pass
class ObjectRotPivotXPlug(Plug):
	parent : ObjectRotPivotPlug = PlugDescriptor("objectRotPivot")
	node : OldGeometryConstraint = None
	pass
class ObjectRotPivotYPlug(Plug):
	parent : ObjectRotPivotPlug = PlugDescriptor("objectRotPivot")
	node : OldGeometryConstraint = None
	pass
class ObjectRotPivotZPlug(Plug):
	parent : ObjectRotPivotPlug = PlugDescriptor("objectRotPivot")
	node : OldGeometryConstraint = None
	pass
class ObjectRotPivotPlug(Plug):
	objectRotPivotX_ : ObjectRotPivotXPlug = PlugDescriptor("objectRotPivotX")
	orpx_ : ObjectRotPivotXPlug = PlugDescriptor("objectRotPivotX")
	objectRotPivotY_ : ObjectRotPivotYPlug = PlugDescriptor("objectRotPivotY")
	orpy_ : ObjectRotPivotYPlug = PlugDescriptor("objectRotPivotY")
	objectRotPivotZ_ : ObjectRotPivotZPlug = PlugDescriptor("objectRotPivotZ")
	orpz_ : ObjectRotPivotZPlug = PlugDescriptor("objectRotPivotZ")
	node : OldGeometryConstraint = None
	pass
class ObjectRotTransXPlug(Plug):
	parent : ObjectRotTransPlug = PlugDescriptor("objectRotTrans")
	node : OldGeometryConstraint = None
	pass
class ObjectRotTransYPlug(Plug):
	parent : ObjectRotTransPlug = PlugDescriptor("objectRotTrans")
	node : OldGeometryConstraint = None
	pass
class ObjectRotTransZPlug(Plug):
	parent : ObjectRotTransPlug = PlugDescriptor("objectRotTrans")
	node : OldGeometryConstraint = None
	pass
class ObjectRotTransPlug(Plug):
	objectRotTransX_ : ObjectRotTransXPlug = PlugDescriptor("objectRotTransX")
	ortx_ : ObjectRotTransXPlug = PlugDescriptor("objectRotTransX")
	objectRotTransY_ : ObjectRotTransYPlug = PlugDescriptor("objectRotTransY")
	orty_ : ObjectRotTransYPlug = PlugDescriptor("objectRotTransY")
	objectRotTransZ_ : ObjectRotTransZPlug = PlugDescriptor("objectRotTransZ")
	ortz_ : ObjectRotTransZPlug = PlugDescriptor("objectRotTransZ")
	node : OldGeometryConstraint = None
	pass
class OutputXPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : OldGeometryConstraint = None
	pass
class OutputYPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : OldGeometryConstraint = None
	pass
class OutputZPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : OldGeometryConstraint = None
	pass
class OutputPlug(Plug):
	outputX_ : OutputXPlug = PlugDescriptor("outputX")
	ox_ : OutputXPlug = PlugDescriptor("outputX")
	outputY_ : OutputYPlug = PlugDescriptor("outputY")
	oy_ : OutputYPlug = PlugDescriptor("outputY")
	outputZ_ : OutputZPlug = PlugDescriptor("outputZ")
	oz_ : OutputZPlug = PlugDescriptor("outputZ")
	node : OldGeometryConstraint = None
	pass
class ParentInverseMatrixPlug(Plug):
	node : OldGeometryConstraint = None
	pass
# endregion


# define node class
class OldGeometryConstraint(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	geometry_ : GeometryPlug = PlugDescriptor("geometry")
	inputMatrix_ : InputMatrixPlug = PlugDescriptor("inputMatrix")
	inputRotPivotX_ : InputRotPivotXPlug = PlugDescriptor("inputRotPivotX")
	inputRotPivotY_ : InputRotPivotYPlug = PlugDescriptor("inputRotPivotY")
	inputRotPivotZ_ : InputRotPivotZPlug = PlugDescriptor("inputRotPivotZ")
	inputRotPivot_ : InputRotPivotPlug = PlugDescriptor("inputRotPivot")
	inputRotTransX_ : InputRotTransXPlug = PlugDescriptor("inputRotTransX")
	inputRotTransY_ : InputRotTransYPlug = PlugDescriptor("inputRotTransY")
	inputRotTransZ_ : InputRotTransZPlug = PlugDescriptor("inputRotTransZ")
	inputRotTrans_ : InputRotTransPlug = PlugDescriptor("inputRotTrans")
	inputTransX_ : InputTransXPlug = PlugDescriptor("inputTransX")
	inputTransY_ : InputTransYPlug = PlugDescriptor("inputTransY")
	inputTransZ_ : InputTransZPlug = PlugDescriptor("inputTransZ")
	inputTrans_ : InputTransPlug = PlugDescriptor("inputTrans")
	weight_ : WeightPlug = PlugDescriptor("weight")
	input_ : InputPlug = PlugDescriptor("input")
	objectRotPivotX_ : ObjectRotPivotXPlug = PlugDescriptor("objectRotPivotX")
	objectRotPivotY_ : ObjectRotPivotYPlug = PlugDescriptor("objectRotPivotY")
	objectRotPivotZ_ : ObjectRotPivotZPlug = PlugDescriptor("objectRotPivotZ")
	objectRotPivot_ : ObjectRotPivotPlug = PlugDescriptor("objectRotPivot")
	objectRotTransX_ : ObjectRotTransXPlug = PlugDescriptor("objectRotTransX")
	objectRotTransY_ : ObjectRotTransYPlug = PlugDescriptor("objectRotTransY")
	objectRotTransZ_ : ObjectRotTransZPlug = PlugDescriptor("objectRotTransZ")
	objectRotTrans_ : ObjectRotTransPlug = PlugDescriptor("objectRotTrans")
	outputX_ : OutputXPlug = PlugDescriptor("outputX")
	outputY_ : OutputYPlug = PlugDescriptor("outputY")
	outputZ_ : OutputZPlug = PlugDescriptor("outputZ")
	output_ : OutputPlug = PlugDescriptor("output")
	parentInverseMatrix_ : ParentInverseMatrixPlug = PlugDescriptor("parentInverseMatrix")

	# node attributes

	typeName = "oldGeometryConstraint"
	apiTypeInt = 449
	apiTypeStr = "kOldGeometryConstraint"
	typeIdInt = 1145523523
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "geometry", "inputMatrix", "inputRotPivotX", "inputRotPivotY", "inputRotPivotZ", "inputRotPivot", "inputRotTransX", "inputRotTransY", "inputRotTransZ", "inputRotTrans", "inputTransX", "inputTransY", "inputTransZ", "inputTrans", "weight", "input", "objectRotPivotX", "objectRotPivotY", "objectRotPivotZ", "objectRotPivot", "objectRotTransX", "objectRotTransY", "objectRotTransZ", "objectRotTrans", "outputX", "outputY", "outputZ", "output", "parentInverseMatrix"]
	nodeLeafPlugs = ["binMembership", "geometry", "input", "objectRotPivot", "objectRotTrans", "output", "parentInverseMatrix"]
	pass

