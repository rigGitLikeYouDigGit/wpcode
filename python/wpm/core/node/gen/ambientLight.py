

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
RenderLight = retriever.getNodeCls("RenderLight")
assert RenderLight
if T.TYPE_CHECKING:
	from .. import RenderLight

# add node doc



# region plug type defs
class AmbientShadePlug(Plug):
	node : AmbientLight = None
	pass
class CastSoftShadowsPlug(Plug):
	node : AmbientLight = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : AmbientLight = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : AmbientLight = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : AmbientLight = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : AmbientLight = None
	pass
class ObjectTypePlug(Plug):
	node : AmbientLight = None
	pass
class ReceiveShadowsPlug(Plug):
	node : AmbientLight = None
	pass
class ShadowRadiusPlug(Plug):
	node : AmbientLight = None
	pass
# endregion


# define node class
class AmbientLight(RenderLight):
	ambientShade_ : AmbientShadePlug = PlugDescriptor("ambientShade")
	castSoftShadows_ : CastSoftShadowsPlug = PlugDescriptor("castSoftShadows")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	objectType_ : ObjectTypePlug = PlugDescriptor("objectType")
	receiveShadows_ : ReceiveShadowsPlug = PlugDescriptor("receiveShadows")
	shadowRadius_ : ShadowRadiusPlug = PlugDescriptor("shadowRadius")

	# node attributes

	typeName = "ambientLight"
	apiTypeInt = 303
	apiTypeStr = "kAmbientLight"
	typeIdInt = 1095582284
	MFnCls = om.MFnDagNode
	pass

