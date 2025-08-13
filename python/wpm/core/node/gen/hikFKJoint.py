

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Joint = retriever.getNodeCls("Joint")
assert Joint
if T.TYPE_CHECKING:
	from .. import Joint

# add node doc



# region plug type defs
class AltConstraintTargetGXPlug(Plug):
	node : HikFKJoint = None
	pass
class AlternateGXPlug(Plug):
	node : HikFKJoint = None
	pass
class LookPlug(Plug):
	node : HikFKJoint = None
	pass
class RotateOffsetXPlug(Plug):
	parent : RotateOffsetPlug = PlugDescriptor("rotateOffset")
	node : HikFKJoint = None
	pass
class RotateOffsetYPlug(Plug):
	parent : RotateOffsetPlug = PlugDescriptor("rotateOffset")
	node : HikFKJoint = None
	pass
class RotateOffsetZPlug(Plug):
	parent : RotateOffsetPlug = PlugDescriptor("rotateOffset")
	node : HikFKJoint = None
	pass
class RotateOffsetPlug(Plug):
	rotateOffsetX_ : RotateOffsetXPlug = PlugDescriptor("rotateOffsetX")
	rox_ : RotateOffsetXPlug = PlugDescriptor("rotateOffsetX")
	rotateOffsetY_ : RotateOffsetYPlug = PlugDescriptor("rotateOffsetY")
	roy_ : RotateOffsetYPlug = PlugDescriptor("rotateOffsetY")
	rotateOffsetZ_ : RotateOffsetZPlug = PlugDescriptor("rotateOffsetZ")
	roz_ : RotateOffsetZPlug = PlugDescriptor("rotateOffsetZ")
	node : HikFKJoint = None
	pass
class ScaleOffsetXPlug(Plug):
	parent : ScaleOffsetPlug = PlugDescriptor("scaleOffset")
	node : HikFKJoint = None
	pass
class ScaleOffsetYPlug(Plug):
	parent : ScaleOffsetPlug = PlugDescriptor("scaleOffset")
	node : HikFKJoint = None
	pass
class ScaleOffsetZPlug(Plug):
	parent : ScaleOffsetPlug = PlugDescriptor("scaleOffset")
	node : HikFKJoint = None
	pass
class ScaleOffsetPlug(Plug):
	scaleOffsetX_ : ScaleOffsetXPlug = PlugDescriptor("scaleOffsetX")
	sox_ : ScaleOffsetXPlug = PlugDescriptor("scaleOffsetX")
	scaleOffsetY_ : ScaleOffsetYPlug = PlugDescriptor("scaleOffsetY")
	soy_ : ScaleOffsetYPlug = PlugDescriptor("scaleOffsetY")
	scaleOffsetZ_ : ScaleOffsetZPlug = PlugDescriptor("scaleOffsetZ")
	soz_ : ScaleOffsetZPlug = PlugDescriptor("scaleOffsetZ")
	node : HikFKJoint = None
	pass
class TranslateOffsetXPlug(Plug):
	parent : TranslateOffsetPlug = PlugDescriptor("translateOffset")
	node : HikFKJoint = None
	pass
class TranslateOffsetYPlug(Plug):
	parent : TranslateOffsetPlug = PlugDescriptor("translateOffset")
	node : HikFKJoint = None
	pass
class TranslateOffsetZPlug(Plug):
	parent : TranslateOffsetPlug = PlugDescriptor("translateOffset")
	node : HikFKJoint = None
	pass
class TranslateOffsetPlug(Plug):
	translateOffsetX_ : TranslateOffsetXPlug = PlugDescriptor("translateOffsetX")
	tox_ : TranslateOffsetXPlug = PlugDescriptor("translateOffsetX")
	translateOffsetY_ : TranslateOffsetYPlug = PlugDescriptor("translateOffsetY")
	toy_ : TranslateOffsetYPlug = PlugDescriptor("translateOffsetY")
	translateOffsetZ_ : TranslateOffsetZPlug = PlugDescriptor("translateOffsetZ")
	toz_ : TranslateOffsetZPlug = PlugDescriptor("translateOffsetZ")
	node : HikFKJoint = None
	pass
class UseAlternateGXPlug(Plug):
	node : HikFKJoint = None
	pass
# endregion


# define node class
class HikFKJoint(Joint):
	altConstraintTargetGX_ : AltConstraintTargetGXPlug = PlugDescriptor("altConstraintTargetGX")
	alternateGX_ : AlternateGXPlug = PlugDescriptor("alternateGX")
	look_ : LookPlug = PlugDescriptor("look")
	rotateOffsetX_ : RotateOffsetXPlug = PlugDescriptor("rotateOffsetX")
	rotateOffsetY_ : RotateOffsetYPlug = PlugDescriptor("rotateOffsetY")
	rotateOffsetZ_ : RotateOffsetZPlug = PlugDescriptor("rotateOffsetZ")
	rotateOffset_ : RotateOffsetPlug = PlugDescriptor("rotateOffset")
	scaleOffsetX_ : ScaleOffsetXPlug = PlugDescriptor("scaleOffsetX")
	scaleOffsetY_ : ScaleOffsetYPlug = PlugDescriptor("scaleOffsetY")
	scaleOffsetZ_ : ScaleOffsetZPlug = PlugDescriptor("scaleOffsetZ")
	scaleOffset_ : ScaleOffsetPlug = PlugDescriptor("scaleOffset")
	translateOffsetX_ : TranslateOffsetXPlug = PlugDescriptor("translateOffsetX")
	translateOffsetY_ : TranslateOffsetYPlug = PlugDescriptor("translateOffsetY")
	translateOffsetZ_ : TranslateOffsetZPlug = PlugDescriptor("translateOffsetZ")
	translateOffset_ : TranslateOffsetPlug = PlugDescriptor("translateOffset")
	useAlternateGX_ : UseAlternateGXPlug = PlugDescriptor("useAlternateGX")

	# node attributes

	typeName = "hikFKJoint"
	apiTypeInt = 963
	apiTypeStr = "kHikFKJoint"
	typeIdInt = 1247037771
	MFnCls = om.MFnTransform
	pass

