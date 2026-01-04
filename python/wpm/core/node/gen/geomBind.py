

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : GeomBind = None
	pass
class BindPosePlug(Plug):
	node : GeomBind = None
	pass
class FalloffPlug(Plug):
	node : GeomBind = None
	pass
class GvPostVoxelCheckPlug(Plug):
	node : GeomBind = None
	pass
class GvResolutionPlug(Plug):
	node : GeomBind = None
	pass
class MaxInfluencesPlug(Plug):
	node : GeomBind = None
	pass
class SkinClustersPlug(Plug):
	node : GeomBind = None
	pass
# endregion


# define node class
class GeomBind(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	bindPose_ : BindPosePlug = PlugDescriptor("bindPose")
	falloff_ : FalloffPlug = PlugDescriptor("falloff")
	gvPostVoxelCheck_ : GvPostVoxelCheckPlug = PlugDescriptor("gvPostVoxelCheck")
	gvResolution_ : GvResolutionPlug = PlugDescriptor("gvResolution")
	maxInfluences_ : MaxInfluencesPlug = PlugDescriptor("maxInfluences")
	skinClusters_ : SkinClustersPlug = PlugDescriptor("skinClusters")

	# node attributes

	typeName = "geomBind"
	apiTypeInt = 1100
	apiTypeStr = "kGeomBind"
	typeIdInt = 1195526478
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "bindPose", "falloff", "gvPostVoxelCheck", "gvResolution", "maxInfluences", "skinClusters"]
	nodeLeafPlugs = ["binMembership", "bindPose", "falloff", "gvPostVoxelCheck", "gvResolution", "maxInfluences", "skinClusters"]
	pass

