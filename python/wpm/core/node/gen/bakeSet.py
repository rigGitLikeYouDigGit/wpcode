

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ObjectSet = retriever.getNodeCls("ObjectSet")
assert ObjectSet
if T.TYPE_CHECKING:
	from .. import ObjectSet

# add node doc



# region plug type defs
class AlphaModePlug(Plug):
	node : BakeSet = None
	pass
class BakeAlphaPlug(Plug):
	node : BakeSet = None
	pass
class ColorModePlug(Plug):
	node : BakeSet = None
	pass
class CustomShaderPlug(Plug):
	node : BakeSet = None
	pass
class NormalDirectionPlug(Plug):
	node : BakeSet = None
	pass
class OcclusionFalloffPlug(Plug):
	node : BakeSet = None
	pass
class OcclusionRaysPlug(Plug):
	node : BakeSet = None
	pass
class OrthogonalReflectionPlug(Plug):
	node : BakeSet = None
	pass
# endregion


# define node class
class BakeSet(ObjectSet):
	alphaMode_ : AlphaModePlug = PlugDescriptor("alphaMode")
	bakeAlpha_ : BakeAlphaPlug = PlugDescriptor("bakeAlpha")
	colorMode_ : ColorModePlug = PlugDescriptor("colorMode")
	customShader_ : CustomShaderPlug = PlugDescriptor("customShader")
	normalDirection_ : NormalDirectionPlug = PlugDescriptor("normalDirection")
	occlusionFalloff_ : OcclusionFalloffPlug = PlugDescriptor("occlusionFalloff")
	occlusionRays_ : OcclusionRaysPlug = PlugDescriptor("occlusionRays")
	orthogonalReflection_ : OrthogonalReflectionPlug = PlugDescriptor("orthogonalReflection")

	# node attributes

	typeName = "bakeSet"
	typeIdInt = 1111575365
	pass

