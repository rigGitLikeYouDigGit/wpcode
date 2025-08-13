

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
assert AbstractBaseCreate
if T.TYPE_CHECKING:
	from .. import AbstractBaseCreate

# add node doc



# region plug type defs
class InputCachePlug(Plug):
	node : LeastSquaresModifier = None
	pass
class InputNurbsObjectPlug(Plug):
	node : LeastSquaresModifier = None
	pass
class ObjectModifierPlug(Plug):
	node : LeastSquaresModifier = None
	pass
class OutputNurbsObjectPlug(Plug):
	node : LeastSquaresModifier = None
	pass
class PointConstraintVPlug(Plug):
	parent : PointConstraintUVWPlug = PlugDescriptor("pointConstraintUVW")
	node : LeastSquaresModifier = None
	pass
class PointConstraintWPlug(Plug):
	parent : PointConstraintUVWPlug = PlugDescriptor("pointConstraintUVW")
	node : LeastSquaresModifier = None
	pass
class PointConstraintUVWPlug(Plug):
	parent : PointConstraintPlug = PlugDescriptor("pointConstraint")
	pointConstraintU_ : PointConstraintUPlug = PlugDescriptor("pointConstraintU")
	pcu_ : PointConstraintUPlug = PlugDescriptor("pointConstraintU")
	pointConstraintV_ : PointConstraintVPlug = PlugDescriptor("pointConstraintV")
	pcv_ : PointConstraintVPlug = PlugDescriptor("pointConstraintV")
	pointConstraintW_ : PointConstraintWPlug = PlugDescriptor("pointConstraintW")
	pcw_ : PointConstraintWPlug = PlugDescriptor("pointConstraintW")
	node : LeastSquaresModifier = None
	pass
class PointPositionYPlug(Plug):
	parent : PointPositionXYZPlug = PlugDescriptor("pointPositionXYZ")
	node : LeastSquaresModifier = None
	pass
class PointPositionZPlug(Plug):
	parent : PointPositionXYZPlug = PlugDescriptor("pointPositionXYZ")
	node : LeastSquaresModifier = None
	pass
class PointPositionXYZPlug(Plug):
	parent : PointConstraintPlug = PlugDescriptor("pointConstraint")
	pointPositionX_ : PointPositionXPlug = PlugDescriptor("pointPositionX")
	ppx_ : PointPositionXPlug = PlugDescriptor("pointPositionX")
	pointPositionY_ : PointPositionYPlug = PlugDescriptor("pointPositionY")
	ppy_ : PointPositionYPlug = PlugDescriptor("pointPositionY")
	pointPositionZ_ : PointPositionZPlug = PlugDescriptor("pointPositionZ")
	ppz_ : PointPositionZPlug = PlugDescriptor("pointPositionZ")
	node : LeastSquaresModifier = None
	pass
class PointWeightPlug(Plug):
	parent : PointConstraintPlug = PlugDescriptor("pointConstraint")
	node : LeastSquaresModifier = None
	pass
class PointConstraintPlug(Plug):
	pointConstraintUVW_ : PointConstraintUVWPlug = PlugDescriptor("pointConstraintUVW")
	puv_ : PointConstraintUVWPlug = PlugDescriptor("pointConstraintUVW")
	pointPositionXYZ_ : PointPositionXYZPlug = PlugDescriptor("pointPositionXYZ")
	xyz_ : PointPositionXYZPlug = PlugDescriptor("pointPositionXYZ")
	pointWeight_ : PointWeightPlug = PlugDescriptor("pointWeight")
	pw_ : PointWeightPlug = PlugDescriptor("pointWeight")
	node : LeastSquaresModifier = None
	pass
class PointConstraintUPlug(Plug):
	parent : PointConstraintUVWPlug = PlugDescriptor("pointConstraintUVW")
	node : LeastSquaresModifier = None
	pass
class PointPositionXPlug(Plug):
	parent : PointPositionXYZPlug = PlugDescriptor("pointPositionXYZ")
	node : LeastSquaresModifier = None
	pass
class PointSymbolicIndexPlug(Plug):
	node : LeastSquaresModifier = None
	pass
class ResetModifierPlug(Plug):
	node : LeastSquaresModifier = None
	pass
class UpdatePointModifierPlug(Plug):
	node : LeastSquaresModifier = None
	pass
class WorldSpaceToObjectSpacePlug(Plug):
	node : LeastSquaresModifier = None
	pass
# endregion


# define node class
class LeastSquaresModifier(AbstractBaseCreate):
	inputCache_ : InputCachePlug = PlugDescriptor("inputCache")
	inputNurbsObject_ : InputNurbsObjectPlug = PlugDescriptor("inputNurbsObject")
	objectModifier_ : ObjectModifierPlug = PlugDescriptor("objectModifier")
	outputNurbsObject_ : OutputNurbsObjectPlug = PlugDescriptor("outputNurbsObject")
	pointConstraintV_ : PointConstraintVPlug = PlugDescriptor("pointConstraintV")
	pointConstraintW_ : PointConstraintWPlug = PlugDescriptor("pointConstraintW")
	pointConstraintUVW_ : PointConstraintUVWPlug = PlugDescriptor("pointConstraintUVW")
	pointPositionY_ : PointPositionYPlug = PlugDescriptor("pointPositionY")
	pointPositionZ_ : PointPositionZPlug = PlugDescriptor("pointPositionZ")
	pointPositionXYZ_ : PointPositionXYZPlug = PlugDescriptor("pointPositionXYZ")
	pointWeight_ : PointWeightPlug = PlugDescriptor("pointWeight")
	pointConstraint_ : PointConstraintPlug = PlugDescriptor("pointConstraint")
	pointConstraintU_ : PointConstraintUPlug = PlugDescriptor("pointConstraintU")
	pointPositionX_ : PointPositionXPlug = PlugDescriptor("pointPositionX")
	pointSymbolicIndex_ : PointSymbolicIndexPlug = PlugDescriptor("pointSymbolicIndex")
	resetModifier_ : ResetModifierPlug = PlugDescriptor("resetModifier")
	updatePointModifier_ : UpdatePointModifierPlug = PlugDescriptor("updatePointModifier")
	worldSpaceToObjectSpace_ : WorldSpaceToObjectSpacePlug = PlugDescriptor("worldSpaceToObjectSpace")

	# node attributes

	typeName = "leastSquaresModifier"
	typeIdInt = 1313624909
	pass

