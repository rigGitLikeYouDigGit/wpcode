

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
SurfaceShape = retriever.getNodeCls("SurfaceShape")
assert SurfaceShape
if T.TYPE_CHECKING:
	from .. import SurfaceShape

# add node doc



# region plug type defs
class CacheNeedsRebuildingPlug(Plug):
	node : HeightField = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : HeightField = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : HeightField = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : HeightField = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	cb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	cg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	cr_ : ColorRPlug = PlugDescriptor("colorR")
	node : HeightField = None
	pass
class DisplacementPlug(Plug):
	node : HeightField = None
	pass
class HeightScalePlug(Plug):
	node : HeightField = None
	pass
class ResolutionPlug(Plug):
	node : HeightField = None
	pass
# endregion


# define node class
class HeightField(SurfaceShape):
	cacheNeedsRebuilding_ : CacheNeedsRebuildingPlug = PlugDescriptor("cacheNeedsRebuilding")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	displacement_ : DisplacementPlug = PlugDescriptor("displacement")
	heightScale_ : HeightScalePlug = PlugDescriptor("heightScale")
	resolution_ : ResolutionPlug = PlugDescriptor("resolution")

	# node attributes

	typeName = "heightField"
	apiTypeInt = 921
	apiTypeStr = "kHeightField"
	typeIdInt = 1329811536
	MFnCls = om.MFnDagNode
	pass

