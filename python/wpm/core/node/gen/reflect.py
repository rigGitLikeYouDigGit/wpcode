

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Lambert = Catalogue.Lambert
else:
	from .. import retriever
	Lambert = retriever.getNodeCls("Lambert")
	assert Lambert

# add node doc



# region plug type defs
class ReflectedColorBPlug(Plug):
	parent : ReflectedColorPlug = PlugDescriptor("reflectedColor")
	node : Reflect = None
	pass
class ReflectedColorGPlug(Plug):
	parent : ReflectedColorPlug = PlugDescriptor("reflectedColor")
	node : Reflect = None
	pass
class ReflectedColorRPlug(Plug):
	parent : ReflectedColorPlug = PlugDescriptor("reflectedColor")
	node : Reflect = None
	pass
class ReflectedColorPlug(Plug):
	reflectedColorB_ : ReflectedColorBPlug = PlugDescriptor("reflectedColorB")
	rb_ : ReflectedColorBPlug = PlugDescriptor("reflectedColorB")
	reflectedColorG_ : ReflectedColorGPlug = PlugDescriptor("reflectedColorG")
	rg_ : ReflectedColorGPlug = PlugDescriptor("reflectedColorG")
	reflectedColorR_ : ReflectedColorRPlug = PlugDescriptor("reflectedColorR")
	rr_ : ReflectedColorRPlug = PlugDescriptor("reflectedColorR")
	node : Reflect = None
	pass
class ReflectionLimitPlug(Plug):
	node : Reflect = None
	pass
class ReflectionSpecularityPlug(Plug):
	node : Reflect = None
	pass
class ReflectivityPlug(Plug):
	node : Reflect = None
	pass
class SpecularColorBPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : Reflect = None
	pass
class SpecularColorGPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : Reflect = None
	pass
class SpecularColorRPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : Reflect = None
	pass
class SpecularColorPlug(Plug):
	specularColorB_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	sb_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	specularColorG_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	sg_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	specularColorR_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	sr_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	node : Reflect = None
	pass
class TriangleNormalCameraXPlug(Plug):
	parent : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")
	node : Reflect = None
	pass
class TriangleNormalCameraYPlug(Plug):
	parent : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")
	node : Reflect = None
	pass
class TriangleNormalCameraZPlug(Plug):
	parent : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")
	node : Reflect = None
	pass
class TriangleNormalCameraPlug(Plug):
	triangleNormalCameraX_ : TriangleNormalCameraXPlug = PlugDescriptor("triangleNormalCameraX")
	tnx_ : TriangleNormalCameraXPlug = PlugDescriptor("triangleNormalCameraX")
	triangleNormalCameraY_ : TriangleNormalCameraYPlug = PlugDescriptor("triangleNormalCameraY")
	tny_ : TriangleNormalCameraYPlug = PlugDescriptor("triangleNormalCameraY")
	triangleNormalCameraZ_ : TriangleNormalCameraZPlug = PlugDescriptor("triangleNormalCameraZ")
	tnz_ : TriangleNormalCameraZPlug = PlugDescriptor("triangleNormalCameraZ")
	node : Reflect = None
	pass
# endregion


# define node class
class Reflect(Lambert):
	reflectedColorB_ : ReflectedColorBPlug = PlugDescriptor("reflectedColorB")
	reflectedColorG_ : ReflectedColorGPlug = PlugDescriptor("reflectedColorG")
	reflectedColorR_ : ReflectedColorRPlug = PlugDescriptor("reflectedColorR")
	reflectedColor_ : ReflectedColorPlug = PlugDescriptor("reflectedColor")
	reflectionLimit_ : ReflectionLimitPlug = PlugDescriptor("reflectionLimit")
	reflectionSpecularity_ : ReflectionSpecularityPlug = PlugDescriptor("reflectionSpecularity")
	reflectivity_ : ReflectivityPlug = PlugDescriptor("reflectivity")
	specularColorB_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	specularColorG_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	specularColorR_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	specularColor_ : SpecularColorPlug = PlugDescriptor("specularColor")
	triangleNormalCameraX_ : TriangleNormalCameraXPlug = PlugDescriptor("triangleNormalCameraX")
	triangleNormalCameraY_ : TriangleNormalCameraYPlug = PlugDescriptor("triangleNormalCameraY")
	triangleNormalCameraZ_ : TriangleNormalCameraZPlug = PlugDescriptor("triangleNormalCameraZ")
	triangleNormalCamera_ : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")

	# node attributes

	typeName = "reflect"
	typeIdInt = 1380338755
	nodeLeafClassAttrs = ["reflectedColorB", "reflectedColorG", "reflectedColorR", "reflectedColor", "reflectionLimit", "reflectionSpecularity", "reflectivity", "specularColorB", "specularColorG", "specularColorR", "specularColor", "triangleNormalCameraX", "triangleNormalCameraY", "triangleNormalCameraZ", "triangleNormalCamera"]
	nodeLeafPlugs = ["reflectedColor", "reflectionLimit", "reflectionSpecularity", "reflectivity", "specularColor", "triangleNormalCamera"]
	pass

