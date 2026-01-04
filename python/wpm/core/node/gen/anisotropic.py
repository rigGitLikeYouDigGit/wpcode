

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Reflect = Catalogue.Reflect
else:
	from .. import retriever
	Reflect = retriever.getNodeCls("Reflect")
	assert Reflect

# add node doc



# region plug type defs
class AnglePlug(Plug):
	node : Anisotropic = None
	pass
class AnisotropicReflectivityPlug(Plug):
	node : Anisotropic = None
	pass
class FresnelRefractiveIndexPlug(Plug):
	node : Anisotropic = None
	pass
class RoughnessPlug(Plug):
	node : Anisotropic = None
	pass
class SpreadXPlug(Plug):
	node : Anisotropic = None
	pass
class SpreadYPlug(Plug):
	node : Anisotropic = None
	pass
class TangentUCameraXPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : Anisotropic = None
	pass
class TangentUCameraYPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : Anisotropic = None
	pass
class TangentUCameraZPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : Anisotropic = None
	pass
class TangentUCameraPlug(Plug):
	tangentUCameraX_ : TangentUCameraXPlug = PlugDescriptor("tangentUCameraX")
	utnx_ : TangentUCameraXPlug = PlugDescriptor("tangentUCameraX")
	tangentUCameraY_ : TangentUCameraYPlug = PlugDescriptor("tangentUCameraY")
	utny_ : TangentUCameraYPlug = PlugDescriptor("tangentUCameraY")
	tangentUCameraZ_ : TangentUCameraZPlug = PlugDescriptor("tangentUCameraZ")
	utnz_ : TangentUCameraZPlug = PlugDescriptor("tangentUCameraZ")
	node : Anisotropic = None
	pass
class TangentVCameraXPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : Anisotropic = None
	pass
class TangentVCameraYPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : Anisotropic = None
	pass
class TangentVCameraZPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : Anisotropic = None
	pass
class TangentVCameraPlug(Plug):
	tangentVCameraX_ : TangentVCameraXPlug = PlugDescriptor("tangentVCameraX")
	vtnx_ : TangentVCameraXPlug = PlugDescriptor("tangentVCameraX")
	tangentVCameraY_ : TangentVCameraYPlug = PlugDescriptor("tangentVCameraY")
	vtny_ : TangentVCameraYPlug = PlugDescriptor("tangentVCameraY")
	tangentVCameraZ_ : TangentVCameraZPlug = PlugDescriptor("tangentVCameraZ")
	vtnz_ : TangentVCameraZPlug = PlugDescriptor("tangentVCameraZ")
	node : Anisotropic = None
	pass
# endregion


# define node class
class Anisotropic(Reflect):
	angle_ : AnglePlug = PlugDescriptor("angle")
	anisotropicReflectivity_ : AnisotropicReflectivityPlug = PlugDescriptor("anisotropicReflectivity")
	fresnelRefractiveIndex_ : FresnelRefractiveIndexPlug = PlugDescriptor("fresnelRefractiveIndex")
	roughness_ : RoughnessPlug = PlugDescriptor("roughness")
	spreadX_ : SpreadXPlug = PlugDescriptor("spreadX")
	spreadY_ : SpreadYPlug = PlugDescriptor("spreadY")
	tangentUCameraX_ : TangentUCameraXPlug = PlugDescriptor("tangentUCameraX")
	tangentUCameraY_ : TangentUCameraYPlug = PlugDescriptor("tangentUCameraY")
	tangentUCameraZ_ : TangentUCameraZPlug = PlugDescriptor("tangentUCameraZ")
	tangentUCamera_ : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	tangentVCameraX_ : TangentVCameraXPlug = PlugDescriptor("tangentVCameraX")
	tangentVCameraY_ : TangentVCameraYPlug = PlugDescriptor("tangentVCameraY")
	tangentVCameraZ_ : TangentVCameraZPlug = PlugDescriptor("tangentVCameraZ")
	tangentVCamera_ : TangentVCameraPlug = PlugDescriptor("tangentVCamera")

	# node attributes

	typeName = "anisotropic"
	typeIdInt = 1380011593
	nodeLeafClassAttrs = ["angle", "anisotropicReflectivity", "fresnelRefractiveIndex", "roughness", "spreadX", "spreadY", "tangentUCameraX", "tangentUCameraY", "tangentUCameraZ", "tangentUCamera", "tangentVCameraX", "tangentVCameraY", "tangentVCameraZ", "tangentVCamera"]
	nodeLeafPlugs = ["angle", "anisotropicReflectivity", "fresnelRefractiveIndex", "roughness", "spreadX", "spreadY", "tangentUCamera", "tangentVCamera"]
	pass

