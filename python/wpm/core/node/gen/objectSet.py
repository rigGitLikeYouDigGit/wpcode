

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
	node : ObjectSet = None
	pass
class DagSetMembersPlug(Plug):
	node : ObjectSet = None
	pass
class DnSetMembersPlug(Plug):
	node : ObjectSet = None
	pass
class EdgesOnlySetPlug(Plug):
	node : ObjectSet = None
	pass
class EditPointsOnlySetPlug(Plug):
	node : ObjectSet = None
	pass
class FacetsOnlySetPlug(Plug):
	node : ObjectSet = None
	pass
class GroupNodesPlug(Plug):
	node : ObjectSet = None
	pass
class HiddenInOutlinerPlug(Plug):
	node : ObjectSet = None
	pass
class IsLayerPlug(Plug):
	node : ObjectSet = None
	pass
class MemberWireframeColorPlug(Plug):
	node : ObjectSet = None
	pass
class PartitionPlug(Plug):
	node : ObjectSet = None
	pass
class RenderableOnlySetPlug(Plug):
	node : ObjectSet = None
	pass
class UsedByPlug(Plug):
	node : ObjectSet = None
	pass
class VerticesOnlySetPlug(Plug):
	node : ObjectSet = None
	pass
# endregion


# define node class
class ObjectSet(Entity):
	annotation_ : AnnotationPlug = PlugDescriptor("annotation")
	dagSetMembers_ : DagSetMembersPlug = PlugDescriptor("dagSetMembers")
	dnSetMembers_ : DnSetMembersPlug = PlugDescriptor("dnSetMembers")
	edgesOnlySet_ : EdgesOnlySetPlug = PlugDescriptor("edgesOnlySet")
	editPointsOnlySet_ : EditPointsOnlySetPlug = PlugDescriptor("editPointsOnlySet")
	facetsOnlySet_ : FacetsOnlySetPlug = PlugDescriptor("facetsOnlySet")
	groupNodes_ : GroupNodesPlug = PlugDescriptor("groupNodes")
	hiddenInOutliner_ : HiddenInOutlinerPlug = PlugDescriptor("hiddenInOutliner")
	isLayer_ : IsLayerPlug = PlugDescriptor("isLayer")
	memberWireframeColor_ : MemberWireframeColorPlug = PlugDescriptor("memberWireframeColor")
	partition_ : PartitionPlug = PlugDescriptor("partition")
	renderableOnlySet_ : RenderableOnlySetPlug = PlugDescriptor("renderableOnlySet")
	usedBy_ : UsedByPlug = PlugDescriptor("usedBy")
	verticesOnlySet_ : VerticesOnlySetPlug = PlugDescriptor("verticesOnlySet")

	# node attributes

	typeName = "objectSet"
	typeIdInt = 1329746772
	pass

