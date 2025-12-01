

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Entity = retriever.getNodeCls("Entity")
assert Entity
if T.TYPE_CHECKING:
	from .. import Entity

# add node doc



# region plug type defs
class AnnotationPlug(Plug):
	node : Partition = None
	pass
class EnvironmentPlug(Plug):
	node : Partition = None
	pass
class PartitionTypePlug(Plug):
	node : Partition = None
	pass
class SetsPlug(Plug):
	node : Partition = None
	pass
# endregion


# define node class
class Partition(Entity):
	annotation_ : AnnotationPlug = PlugDescriptor("annotation")
	environment_ : EnvironmentPlug = PlugDescriptor("environment")
	partitionType_ : PartitionTypePlug = PlugDescriptor("partitionType")
	sets_ : SetsPlug = PlugDescriptor("sets")

	# node attributes

	typeName = "partition"
	apiTypeInt = 456
	apiTypeStr = "kPartition"
	typeIdInt = 1347572814
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["annotation", "environment", "partitionType", "sets"]
	nodeLeafPlugs = ["annotation", "environment", "partitionType", "sets"]
	pass

