

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AbstractBaseCreate = Catalogue.AbstractBaseCreate
else:
	from .. import retriever
	AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
	assert AbstractBaseCreate

# add node doc



# region plug type defs
class ConnectedEdgesPlug(Plug):
	node : GlobalStitch = None
	pass
class InputSurfacePlug(Plug):
	node : GlobalStitch = None
	pass
class LockSurfacePlug(Plug):
	node : GlobalStitch = None
	pass
class MaxSeparationPlug(Plug):
	node : GlobalStitch = None
	pass
class ModificationResistancePlug(Plug):
	node : GlobalStitch = None
	pass
class OutputSurfacePlug(Plug):
	node : GlobalStitch = None
	pass
class SamplingPlug(Plug):
	node : GlobalStitch = None
	pass
class ShouldBeLastPlug(Plug):
	node : GlobalStitch = None
	pass
class StitchCornersPlug(Plug):
	node : GlobalStitch = None
	pass
class StitchEdgesPlug(Plug):
	node : GlobalStitch = None
	pass
class StitchPartialEdgesPlug(Plug):
	node : GlobalStitch = None
	pass
class StitchSmoothnessPlug(Plug):
	node : GlobalStitch = None
	pass
class TopologyPlug(Plug):
	node : GlobalStitch = None
	pass
class UnconnectedEdgesPlug(Plug):
	node : GlobalStitch = None
	pass
class UpdateSamplingPlug(Plug):
	node : GlobalStitch = None
	pass
# endregion


# define node class
class GlobalStitch(AbstractBaseCreate):
	connectedEdges_ : ConnectedEdgesPlug = PlugDescriptor("connectedEdges")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	lockSurface_ : LockSurfacePlug = PlugDescriptor("lockSurface")
	maxSeparation_ : MaxSeparationPlug = PlugDescriptor("maxSeparation")
	modificationResistance_ : ModificationResistancePlug = PlugDescriptor("modificationResistance")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	sampling_ : SamplingPlug = PlugDescriptor("sampling")
	shouldBeLast_ : ShouldBeLastPlug = PlugDescriptor("shouldBeLast")
	stitchCorners_ : StitchCornersPlug = PlugDescriptor("stitchCorners")
	stitchEdges_ : StitchEdgesPlug = PlugDescriptor("stitchEdges")
	stitchPartialEdges_ : StitchPartialEdgesPlug = PlugDescriptor("stitchPartialEdges")
	stitchSmoothness_ : StitchSmoothnessPlug = PlugDescriptor("stitchSmoothness")
	topology_ : TopologyPlug = PlugDescriptor("topology")
	unconnectedEdges_ : UnconnectedEdgesPlug = PlugDescriptor("unconnectedEdges")
	updateSampling_ : UpdateSamplingPlug = PlugDescriptor("updateSampling")

	# node attributes

	typeName = "globalStitch"
	apiTypeInt = 701
	apiTypeStr = "kGlobalStitch"
	typeIdInt = 1313297236
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["connectedEdges", "inputSurface", "lockSurface", "maxSeparation", "modificationResistance", "outputSurface", "sampling", "shouldBeLast", "stitchCorners", "stitchEdges", "stitchPartialEdges", "stitchSmoothness", "topology", "unconnectedEdges", "updateSampling"]
	nodeLeafPlugs = ["connectedEdges", "inputSurface", "lockSurface", "maxSeparation", "modificationResistance", "outputSurface", "sampling", "shouldBeLast", "stitchCorners", "stitchEdges", "stitchPartialEdges", "stitchSmoothness", "topology", "unconnectedEdges", "updateSampling"]
	pass

