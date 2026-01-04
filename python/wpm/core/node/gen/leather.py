

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Texture3d = Catalogue.Texture3d
else:
	from .. import retriever
	Texture3d = retriever.getNodeCls("Texture3d")
	assert Texture3d

# add node doc



# region plug type defs
class CellColorBPlug(Plug):
	parent : CellColorPlug = PlugDescriptor("cellColor")
	node : Leather = None
	pass
class CellColorGPlug(Plug):
	parent : CellColorPlug = PlugDescriptor("cellColor")
	node : Leather = None
	pass
class CellColorRPlug(Plug):
	parent : CellColorPlug = PlugDescriptor("cellColor")
	node : Leather = None
	pass
class CellColorPlug(Plug):
	cellColorB_ : CellColorBPlug = PlugDescriptor("cellColorB")
	ceb_ : CellColorBPlug = PlugDescriptor("cellColorB")
	cellColorG_ : CellColorGPlug = PlugDescriptor("cellColorG")
	ceg_ : CellColorGPlug = PlugDescriptor("cellColorG")
	cellColorR_ : CellColorRPlug = PlugDescriptor("cellColorR")
	cer_ : CellColorRPlug = PlugDescriptor("cellColorR")
	node : Leather = None
	pass
class CellSizePlug(Plug):
	node : Leather = None
	pass
class CreaseColorBPlug(Plug):
	parent : CreaseColorPlug = PlugDescriptor("creaseColor")
	node : Leather = None
	pass
class CreaseColorGPlug(Plug):
	parent : CreaseColorPlug = PlugDescriptor("creaseColor")
	node : Leather = None
	pass
class CreaseColorRPlug(Plug):
	parent : CreaseColorPlug = PlugDescriptor("creaseColor")
	node : Leather = None
	pass
class CreaseColorPlug(Plug):
	creaseColorB_ : CreaseColorBPlug = PlugDescriptor("creaseColorB")
	crb_ : CreaseColorBPlug = PlugDescriptor("creaseColorB")
	creaseColorG_ : CreaseColorGPlug = PlugDescriptor("creaseColorG")
	crg_ : CreaseColorGPlug = PlugDescriptor("creaseColorG")
	creaseColorR_ : CreaseColorRPlug = PlugDescriptor("creaseColorR")
	crr_ : CreaseColorRPlug = PlugDescriptor("creaseColorR")
	node : Leather = None
	pass
class CreasesPlug(Plug):
	node : Leather = None
	pass
class DensityPlug(Plug):
	node : Leather = None
	pass
class RandomnessPlug(Plug):
	node : Leather = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Leather = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Leather = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Leather = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : Leather = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Leather = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Leather = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Leather = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	roy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	roz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : Leather = None
	pass
class SpottynessPlug(Plug):
	node : Leather = None
	pass
class ThresholdPlug(Plug):
	node : Leather = None
	pass
# endregion


# define node class
class Leather(Texture3d):
	cellColorB_ : CellColorBPlug = PlugDescriptor("cellColorB")
	cellColorG_ : CellColorGPlug = PlugDescriptor("cellColorG")
	cellColorR_ : CellColorRPlug = PlugDescriptor("cellColorR")
	cellColor_ : CellColorPlug = PlugDescriptor("cellColor")
	cellSize_ : CellSizePlug = PlugDescriptor("cellSize")
	creaseColorB_ : CreaseColorBPlug = PlugDescriptor("creaseColorB")
	creaseColorG_ : CreaseColorGPlug = PlugDescriptor("creaseColorG")
	creaseColorR_ : CreaseColorRPlug = PlugDescriptor("creaseColorR")
	creaseColor_ : CreaseColorPlug = PlugDescriptor("creaseColor")
	creases_ : CreasesPlug = PlugDescriptor("creases")
	density_ : DensityPlug = PlugDescriptor("density")
	randomness_ : RandomnessPlug = PlugDescriptor("randomness")
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	refPointCamera_ : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	refPointObj_ : RefPointObjPlug = PlugDescriptor("refPointObj")
	spottyness_ : SpottynessPlug = PlugDescriptor("spottyness")
	threshold_ : ThresholdPlug = PlugDescriptor("threshold")

	# node attributes

	typeName = "leather"
	apiTypeInt = 512
	apiTypeStr = "kLeather"
	typeIdInt = 1381256261
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["cellColorB", "cellColorG", "cellColorR", "cellColor", "cellSize", "creaseColorB", "creaseColorG", "creaseColorR", "creaseColor", "creases", "density", "randomness", "refPointCameraX", "refPointCameraY", "refPointCameraZ", "refPointCamera", "refPointObjX", "refPointObjY", "refPointObjZ", "refPointObj", "spottyness", "threshold"]
	nodeLeafPlugs = ["cellColor", "cellSize", "creaseColor", "creases", "density", "randomness", "refPointCamera", "refPointObj", "spottyness", "threshold"]
	pass

