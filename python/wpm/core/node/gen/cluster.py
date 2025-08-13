

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
WeightGeometryFilter = retriever.getNodeCls("WeightGeometryFilter")
assert WeightGeometryFilter
if T.TYPE_CHECKING:
	from .. import WeightGeometryFilter

# add node doc



# region plug type defs
class AngleInterpolationPlug(Plug):
	node : Cluster = None
	pass
class BindPreMatrixPlug(Plug):
	node : Cluster = None
	pass
class BindStatePlug(Plug):
	node : Cluster = None
	pass
class CacheSetupPlug(Plug):
	node : Cluster = None
	pass
class PostMatrixPlug(Plug):
	parent : ClusterXformsPlug = PlugDescriptor("clusterXforms")
	node : Cluster = None
	pass
class PreMatrixPlug(Plug):
	parent : ClusterXformsPlug = PlugDescriptor("clusterXforms")
	node : Cluster = None
	pass
class WeightedMatrixPlug(Plug):
	parent : ClusterXformsPlug = PlugDescriptor("clusterXforms")
	node : Cluster = None
	pass
class ClusterXformsPlug(Plug):
	postMatrix_ : PostMatrixPlug = PlugDescriptor("postMatrix")
	post_ : PostMatrixPlug = PlugDescriptor("postMatrix")
	preMatrix_ : PreMatrixPlug = PlugDescriptor("preMatrix")
	pre_ : PreMatrixPlug = PlugDescriptor("preMatrix")
	weightedMatrix_ : WeightedMatrixPlug = PlugDescriptor("weightedMatrix")
	wt_ : WeightedMatrixPlug = PlugDescriptor("weightedMatrix")
	node : Cluster = None
	pass
class GeomMatrixPlug(Plug):
	node : Cluster = None
	pass
class MatrixPlug(Plug):
	node : Cluster = None
	pass
class PostCompensationMatrixPlug(Plug):
	node : Cluster = None
	pass
class PreCompensationMatrixPlug(Plug):
	node : Cluster = None
	pass
class RelativePlug(Plug):
	node : Cluster = None
	pass
class WeightedCompensationMatrixPlug(Plug):
	node : Cluster = None
	pass
# endregion


# define node class
class Cluster(WeightGeometryFilter):
	angleInterpolation_ : AngleInterpolationPlug = PlugDescriptor("angleInterpolation")
	bindPreMatrix_ : BindPreMatrixPlug = PlugDescriptor("bindPreMatrix")
	bindState_ : BindStatePlug = PlugDescriptor("bindState")
	cacheSetup_ : CacheSetupPlug = PlugDescriptor("cacheSetup")
	postMatrix_ : PostMatrixPlug = PlugDescriptor("postMatrix")
	preMatrix_ : PreMatrixPlug = PlugDescriptor("preMatrix")
	weightedMatrix_ : WeightedMatrixPlug = PlugDescriptor("weightedMatrix")
	clusterXforms_ : ClusterXformsPlug = PlugDescriptor("clusterXforms")
	geomMatrix_ : GeomMatrixPlug = PlugDescriptor("geomMatrix")
	matrix_ : MatrixPlug = PlugDescriptor("matrix")
	postCompensationMatrix_ : PostCompensationMatrixPlug = PlugDescriptor("postCompensationMatrix")
	preCompensationMatrix_ : PreCompensationMatrixPlug = PlugDescriptor("preCompensationMatrix")
	relative_ : RelativePlug = PlugDescriptor("relative")
	weightedCompensationMatrix_ : WeightedCompensationMatrixPlug = PlugDescriptor("weightedCompensationMatrix")

	# node attributes

	typeName = "cluster"
	apiTypeInt = 251
	apiTypeStr = "kCluster"
	typeIdInt = 1178815571
	MFnCls = om.MFnDagNode
	pass

