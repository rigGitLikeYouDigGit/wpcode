

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifierWorld = Catalogue.PolyModifierWorld
else:
	from .. import retriever
	PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
	assert PolyModifierWorld

# add node doc



# region plug type defs
class LocalTranslateXPlug(Plug):
	parent : LocalTranslatePlug = PlugDescriptor("localTranslate")
	node : PolyPoke = None
	pass
class LocalTranslateYPlug(Plug):
	parent : LocalTranslatePlug = PlugDescriptor("localTranslate")
	node : PolyPoke = None
	pass
class LocalTranslateZPlug(Plug):
	parent : LocalTranslatePlug = PlugDescriptor("localTranslate")
	node : PolyPoke = None
	pass
class LocalTranslatePlug(Plug):
	localTranslateX_ : LocalTranslateXPlug = PlugDescriptor("localTranslateX")
	ltx_ : LocalTranslateXPlug = PlugDescriptor("localTranslateX")
	localTranslateY_ : LocalTranslateYPlug = PlugDescriptor("localTranslateY")
	lty_ : LocalTranslateYPlug = PlugDescriptor("localTranslateY")
	localTranslateZ_ : LocalTranslateZPlug = PlugDescriptor("localTranslateZ")
	ltz_ : LocalTranslateZPlug = PlugDescriptor("localTranslateZ")
	node : PolyPoke = None
	pass
class MatrixPlug(Plug):
	node : PolyPoke = None
	pass
class Maya70Plug(Plug):
	node : PolyPoke = None
	pass
class TranslateXPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : PolyPoke = None
	pass
class TranslateYPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : PolyPoke = None
	pass
class TranslateZPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : PolyPoke = None
	pass
class TranslatePlug(Plug):
	translateX_ : TranslateXPlug = PlugDescriptor("translateX")
	tx_ : TranslateXPlug = PlugDescriptor("translateX")
	translateY_ : TranslateYPlug = PlugDescriptor("translateY")
	ty_ : TranslateYPlug = PlugDescriptor("translateY")
	translateZ_ : TranslateZPlug = PlugDescriptor("translateZ")
	tz_ : TranslateZPlug = PlugDescriptor("translateZ")
	node : PolyPoke = None
	pass
# endregion


# define node class
class PolyPoke(PolyModifierWorld):
	localTranslateX_ : LocalTranslateXPlug = PlugDescriptor("localTranslateX")
	localTranslateY_ : LocalTranslateYPlug = PlugDescriptor("localTranslateY")
	localTranslateZ_ : LocalTranslateZPlug = PlugDescriptor("localTranslateZ")
	localTranslate_ : LocalTranslatePlug = PlugDescriptor("localTranslate")
	matrix_ : MatrixPlug = PlugDescriptor("matrix")
	maya70_ : Maya70Plug = PlugDescriptor("maya70")
	translateX_ : TranslateXPlug = PlugDescriptor("translateX")
	translateY_ : TranslateYPlug = PlugDescriptor("translateY")
	translateZ_ : TranslateZPlug = PlugDescriptor("translateZ")
	translate_ : TranslatePlug = PlugDescriptor("translate")

	# node attributes

	typeName = "polyPoke"
	apiTypeInt = 902
	apiTypeStr = "kPolyPoke"
	typeIdInt = 1347440715
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["localTranslateX", "localTranslateY", "localTranslateZ", "localTranslate", "matrix", "maya70", "translateX", "translateY", "translateZ", "translate"]
	nodeLeafPlugs = ["localTranslate", "matrix", "maya70", "translate"]
	pass

