

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ObjectFilter = retriever.getNodeCls("ObjectFilter")
assert ObjectFilter
if T.TYPE_CHECKING:
	from .. import ObjectFilter

# add node doc



# region plug type defs
class ExclusiveLightsPlug(Plug):
	node : ObjectRenderFilter = None
	pass
class LightSetsPlug(Plug):
	node : ObjectRenderFilter = None
	pass
class LightsPlug(Plug):
	node : ObjectRenderFilter = None
	pass
class NonExclusiveLightsPlug(Plug):
	node : ObjectRenderFilter = None
	pass
class PostProcessPlug(Plug):
	node : ObjectRenderFilter = None
	pass
class RenderableObjectSetsPlug(Plug):
	node : ObjectRenderFilter = None
	pass
class RenderingPlug(Plug):
	node : ObjectRenderFilter = None
	pass
class ShadersPlug(Plug):
	node : ObjectRenderFilter = None
	pass
class TexturesPlug(Plug):
	node : ObjectRenderFilter = None
	pass
class Textures2DPlug(Plug):
	node : ObjectRenderFilter = None
	pass
class Textures3DPlug(Plug):
	node : ObjectRenderFilter = None
	pass
class UtilityPlug(Plug):
	node : ObjectRenderFilter = None
	pass
# endregion


# define node class
class ObjectRenderFilter(ObjectFilter):
	exclusiveLights_ : ExclusiveLightsPlug = PlugDescriptor("exclusiveLights")
	lightSets_ : LightSetsPlug = PlugDescriptor("lightSets")
	lights_ : LightsPlug = PlugDescriptor("lights")
	nonExclusiveLights_ : NonExclusiveLightsPlug = PlugDescriptor("nonExclusiveLights")
	postProcess_ : PostProcessPlug = PlugDescriptor("postProcess")
	renderableObjectSets_ : RenderableObjectSetsPlug = PlugDescriptor("renderableObjectSets")
	rendering_ : RenderingPlug = PlugDescriptor("rendering")
	shaders_ : ShadersPlug = PlugDescriptor("shaders")
	textures_ : TexturesPlug = PlugDescriptor("textures")
	textures2D_ : Textures2DPlug = PlugDescriptor("textures2D")
	textures3D_ : Textures3DPlug = PlugDescriptor("textures3D")
	utility_ : UtilityPlug = PlugDescriptor("utility")

	# node attributes

	typeName = "objectRenderFilter"
	apiTypeInt = 681
	apiTypeStr = "kObjectRenderFilter"
	typeIdInt = 1330792012
	MFnCls = om.MFnDependencyNode
	pass

