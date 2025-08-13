

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
class BaseLatticeMatrixPlug(Plug):
	parent : BaseLatticePlug = PlugDescriptor("baseLattice")
	node : Ffd = None
	pass
class BaseLatticePointsPlug(Plug):
	parent : BaseLatticePlug = PlugDescriptor("baseLattice")
	node : Ffd = None
	pass
class BaseLatticePlug(Plug):
	baseLatticeMatrix_ : BaseLatticeMatrixPlug = PlugDescriptor("baseLatticeMatrix")
	blm_ : BaseLatticeMatrixPlug = PlugDescriptor("baseLatticeMatrix")
	baseLatticePoints_ : BaseLatticePointsPlug = PlugDescriptor("baseLatticePoints")
	blp_ : BaseLatticePointsPlug = PlugDescriptor("baseLatticePoints")
	node : Ffd = None
	pass
class BindToOriginalGeometryPlug(Plug):
	node : Ffd = None
	pass
class CacheSetupPlug(Plug):
	node : Ffd = None
	pass
class DeformedLatticeMatrixPlug(Plug):
	parent : DeformedLatticePlug = PlugDescriptor("deformedLattice")
	node : Ffd = None
	pass
class DeformedLatticePointsPlug(Plug):
	parent : DeformedLatticePlug = PlugDescriptor("deformedLattice")
	node : Ffd = None
	pass
class DeformedLatticePlug(Plug):
	deformedLatticeMatrix_ : DeformedLatticeMatrixPlug = PlugDescriptor("deformedLatticeMatrix")
	dlm_ : DeformedLatticeMatrixPlug = PlugDescriptor("deformedLatticeMatrix")
	deformedLatticePoints_ : DeformedLatticePointsPlug = PlugDescriptor("deformedLatticePoints")
	dlp_ : DeformedLatticePointsPlug = PlugDescriptor("deformedLatticePoints")
	node : Ffd = None
	pass
class FreezeGeometryPlug(Plug):
	node : Ffd = None
	pass
class LocalPlug(Plug):
	node : Ffd = None
	pass
class LocalInfluenceSPlug(Plug):
	node : Ffd = None
	pass
class LocalInfluenceTPlug(Plug):
	node : Ffd = None
	pass
class LocalInfluenceUPlug(Plug):
	node : Ffd = None
	pass
class OutsideFalloffDistPlug(Plug):
	node : Ffd = None
	pass
class OutsideLatticePlug(Plug):
	node : Ffd = None
	pass
class PartialResolutionPlug(Plug):
	node : Ffd = None
	pass
class StuCachePlug(Plug):
	parent : StuCacheListPlug = PlugDescriptor("stuCacheList")
	node : Ffd = None
	pass
class StuCacheListPlug(Plug):
	stuCache_ : StuCachePlug = PlugDescriptor("stuCache")
	stu_ : StuCachePlug = PlugDescriptor("stuCache")
	node : Ffd = None
	pass
class UsePartialResolutionPlug(Plug):
	node : Ffd = None
	pass
# endregion


# define node class
class Ffd(WeightGeometryFilter):
	baseLatticeMatrix_ : BaseLatticeMatrixPlug = PlugDescriptor("baseLatticeMatrix")
	baseLatticePoints_ : BaseLatticePointsPlug = PlugDescriptor("baseLatticePoints")
	baseLattice_ : BaseLatticePlug = PlugDescriptor("baseLattice")
	bindToOriginalGeometry_ : BindToOriginalGeometryPlug = PlugDescriptor("bindToOriginalGeometry")
	cacheSetup_ : CacheSetupPlug = PlugDescriptor("cacheSetup")
	deformedLatticeMatrix_ : DeformedLatticeMatrixPlug = PlugDescriptor("deformedLatticeMatrix")
	deformedLatticePoints_ : DeformedLatticePointsPlug = PlugDescriptor("deformedLatticePoints")
	deformedLattice_ : DeformedLatticePlug = PlugDescriptor("deformedLattice")
	freezeGeometry_ : FreezeGeometryPlug = PlugDescriptor("freezeGeometry")
	local_ : LocalPlug = PlugDescriptor("local")
	localInfluenceS_ : LocalInfluenceSPlug = PlugDescriptor("localInfluenceS")
	localInfluenceT_ : LocalInfluenceTPlug = PlugDescriptor("localInfluenceT")
	localInfluenceU_ : LocalInfluenceUPlug = PlugDescriptor("localInfluenceU")
	outsideFalloffDist_ : OutsideFalloffDistPlug = PlugDescriptor("outsideFalloffDist")
	outsideLattice_ : OutsideLatticePlug = PlugDescriptor("outsideLattice")
	partialResolution_ : PartialResolutionPlug = PlugDescriptor("partialResolution")
	stuCache_ : StuCachePlug = PlugDescriptor("stuCache")
	stuCacheList_ : StuCacheListPlug = PlugDescriptor("stuCacheList")
	usePartialResolution_ : UsePartialResolutionPlug = PlugDescriptor("usePartialResolution")

	# node attributes

	typeName = "ffd"
	typeIdInt = 1179010628
	pass

