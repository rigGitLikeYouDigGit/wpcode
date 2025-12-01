

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class FaceIdPlug(Plug):
	parent : NewUVPlug = PlugDescriptor("newUV")
	node : PolyTweakUV = None
	pass
class NewUValuePlug(Plug):
	parent : NewUVPlug = PlugDescriptor("newUV")
	node : PolyTweakUV = None
	pass
class NewVValuePlug(Plug):
	parent : NewUVPlug = PlugDescriptor("newUV")
	node : PolyTweakUV = None
	pass
class VertexIdPlug(Plug):
	parent : NewUVPlug = PlugDescriptor("newUV")
	node : PolyTweakUV = None
	pass
class NewUVPlug(Plug):
	faceId_ : FaceIdPlug = PlugDescriptor("faceId")
	fid_ : FaceIdPlug = PlugDescriptor("faceId")
	newUValue_ : NewUValuePlug = PlugDescriptor("newUValue")
	nu_ : NewUValuePlug = PlugDescriptor("newUValue")
	newVValue_ : NewVValuePlug = PlugDescriptor("newVValue")
	nv_ : NewVValuePlug = PlugDescriptor("newVValue")
	vertexId_ : VertexIdPlug = PlugDescriptor("vertexId")
	vid_ : VertexIdPlug = PlugDescriptor("vertexId")
	node : PolyTweakUV = None
	pass
class UvSetNamePlug(Plug):
	node : PolyTweakUV = None
	pass
class UvTweakUPlug(Plug):
	parent : UvTweakPlug = PlugDescriptor("uvTweak")
	node : PolyTweakUV = None
	pass
class UvTweakVPlug(Plug):
	parent : UvTweakPlug = PlugDescriptor("uvTweak")
	node : PolyTweakUV = None
	pass
class UvTweakPlug(Plug):
	uvTweakU_ : UvTweakUPlug = PlugDescriptor("uvTweakU")
	tu_ : UvTweakUPlug = PlugDescriptor("uvTweakU")
	uvTweakV_ : UvTweakVPlug = PlugDescriptor("uvTweakV")
	tv_ : UvTweakVPlug = PlugDescriptor("uvTweakV")
	node : PolyTweakUV = None
	pass
# endregion


# define node class
class PolyTweakUV(PolyModifier):
	faceId_ : FaceIdPlug = PlugDescriptor("faceId")
	newUValue_ : NewUValuePlug = PlugDescriptor("newUValue")
	newVValue_ : NewVValuePlug = PlugDescriptor("newVValue")
	vertexId_ : VertexIdPlug = PlugDescriptor("vertexId")
	newUV_ : NewUVPlug = PlugDescriptor("newUV")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")
	uvTweakU_ : UvTweakUPlug = PlugDescriptor("uvTweakU")
	uvTweakV_ : UvTweakVPlug = PlugDescriptor("uvTweakV")
	uvTweak_ : UvTweakPlug = PlugDescriptor("uvTweak")

	# node attributes

	typeName = "polyTweakUV"
	apiTypeInt = 709
	apiTypeStr = "kPolyTweakUV"
	typeIdInt = 1347704150
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["faceId", "newUValue", "newVValue", "vertexId", "newUV", "uvSetName", "uvTweakU", "uvTweakV", "uvTweak"]
	nodeLeafPlugs = ["newUV", "uvSetName", "uvTweak"]
	pass

