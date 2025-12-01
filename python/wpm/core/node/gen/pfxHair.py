

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PfxGeometry = retriever.getNodeCls("PfxGeometry")
assert PfxGeometry
if T.TYPE_CHECKING:
	from .. import PfxGeometry

# add node doc



# region plug type defs
class ReceiveShadowsPlug(Plug):
	node : PfxHair = None
	pass
class RenderHairsPlug(Plug):
	node : PfxHair = None
	pass
class VisibleInReflectionsPlug(Plug):
	node : PfxHair = None
	pass
class VisibleInRefractionsPlug(Plug):
	node : PfxHair = None
	pass
# endregion


# define node class
class PfxHair(PfxGeometry):
	receiveShadows_ : ReceiveShadowsPlug = PlugDescriptor("receiveShadows")
	renderHairs_ : RenderHairsPlug = PlugDescriptor("renderHairs")
	visibleInReflections_ : VisibleInReflectionsPlug = PlugDescriptor("visibleInReflections")
	visibleInRefractions_ : VisibleInRefractionsPlug = PlugDescriptor("visibleInRefractions")

	# node attributes

	typeName = "pfxHair"
	apiTypeInt = 946
	apiTypeStr = "kPfxHair"
	typeIdInt = 1346783297
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["receiveShadows", "renderHairs", "visibleInReflections", "visibleInRefractions"]
	nodeLeafPlugs = ["receiveShadows", "renderHairs", "visibleInReflections", "visibleInRefractions"]
	pass

