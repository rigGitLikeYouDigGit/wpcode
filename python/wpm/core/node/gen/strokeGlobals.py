

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
_BASE_ = retriever.getNodeCls("_BASE_")
assert _BASE_
if T.TYPE_CHECKING:
	from .. import _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : StrokeGlobals = None
	pass
class CanvasScalePlug(Plug):
	node : StrokeGlobals = None
	pass
class ForceDepthPlug(Plug):
	node : StrokeGlobals = None
	pass
class ForceRealLightsPlug(Plug):
	node : StrokeGlobals = None
	pass
class ForceTubeDirAlongPathPlug(Plug):
	node : StrokeGlobals = None
	pass
class LightDirectionXPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : StrokeGlobals = None
	pass
class LightDirectionYPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : StrokeGlobals = None
	pass
class LightDirectionZPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : StrokeGlobals = None
	pass
class LightDirectionPlug(Plug):
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	ldx_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	ldy_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	ldz_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	node : StrokeGlobals = None
	pass
class SceneScalePlug(Plug):
	node : StrokeGlobals = None
	pass
class SceneWrapHPlug(Plug):
	node : StrokeGlobals = None
	pass
class SceneWrapVPlug(Plug):
	node : StrokeGlobals = None
	pass
class UseCanvasLightPlug(Plug):
	node : StrokeGlobals = None
	pass
class WrapHPlug(Plug):
	node : StrokeGlobals = None
	pass
class WrapVPlug(Plug):
	node : StrokeGlobals = None
	pass
# endregion


# define node class
class StrokeGlobals(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	canvasScale_ : CanvasScalePlug = PlugDescriptor("canvasScale")
	forceDepth_ : ForceDepthPlug = PlugDescriptor("forceDepth")
	forceRealLights_ : ForceRealLightsPlug = PlugDescriptor("forceRealLights")
	forceTubeDirAlongPath_ : ForceTubeDirAlongPathPlug = PlugDescriptor("forceTubeDirAlongPath")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	lightDirection_ : LightDirectionPlug = PlugDescriptor("lightDirection")
	sceneScale_ : SceneScalePlug = PlugDescriptor("sceneScale")
	sceneWrapH_ : SceneWrapHPlug = PlugDescriptor("sceneWrapH")
	sceneWrapV_ : SceneWrapVPlug = PlugDescriptor("sceneWrapV")
	useCanvasLight_ : UseCanvasLightPlug = PlugDescriptor("useCanvasLight")
	wrapH_ : WrapHPlug = PlugDescriptor("wrapH")
	wrapV_ : WrapVPlug = PlugDescriptor("wrapV")

	# node attributes

	typeName = "strokeGlobals"
	apiTypeInt = 766
	apiTypeStr = "kStrokeGlobals"
	typeIdInt = 1398033223
	MFnCls = om.MFnDependencyNode
	pass

