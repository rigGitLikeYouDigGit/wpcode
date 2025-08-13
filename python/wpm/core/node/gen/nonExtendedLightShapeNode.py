

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
NonAmbientLightShapeNode = retriever.getNodeCls("NonAmbientLightShapeNode")
assert NonAmbientLightShapeNode
if T.TYPE_CHECKING:
	from .. import NonAmbientLightShapeNode

# add node doc



# region plug type defs
class CastSoftShadowsPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class DmapBiasPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class DmapFarClipPlanePlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class DmapFilterSizePlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class DmapFocusPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class DmapFrameExtPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class DmapLightNamePlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class DmapNamePlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class DmapNearClipPlanePlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class DmapResolutionPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class DmapSceneNamePlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class DmapUseMacroPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class DmapWidthFocusPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class FogShadowIntensityPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class LastWrittenDmapAnimExtNamePlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class LightRadiusPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class ReceiveShadowsPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class ReuseDmapPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class UseDepthMapShadowsPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class UseDmapAutoClippingPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class UseDmapAutoFocusPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class UseMidDistDmapPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class UseOnlySingleDmapPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class UseXMinusDmapPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class UseXPlusDmapPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class UseYMinusDmapPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class UseYPlusDmapPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class UseZMinusDmapPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class UseZPlusDmapPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class VolumeShadowSamplesPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
class WriteDmapPlug(Plug):
	node : NonExtendedLightShapeNode = None
	pass
# endregion


# define node class
class NonExtendedLightShapeNode(NonAmbientLightShapeNode):
	castSoftShadows_ : CastSoftShadowsPlug = PlugDescriptor("castSoftShadows")
	dmapBias_ : DmapBiasPlug = PlugDescriptor("dmapBias")
	dmapFarClipPlane_ : DmapFarClipPlanePlug = PlugDescriptor("dmapFarClipPlane")
	dmapFilterSize_ : DmapFilterSizePlug = PlugDescriptor("dmapFilterSize")
	dmapFocus_ : DmapFocusPlug = PlugDescriptor("dmapFocus")
	dmapFrameExt_ : DmapFrameExtPlug = PlugDescriptor("dmapFrameExt")
	dmapLightName_ : DmapLightNamePlug = PlugDescriptor("dmapLightName")
	dmapName_ : DmapNamePlug = PlugDescriptor("dmapName")
	dmapNearClipPlane_ : DmapNearClipPlanePlug = PlugDescriptor("dmapNearClipPlane")
	dmapResolution_ : DmapResolutionPlug = PlugDescriptor("dmapResolution")
	dmapSceneName_ : DmapSceneNamePlug = PlugDescriptor("dmapSceneName")
	dmapUseMacro_ : DmapUseMacroPlug = PlugDescriptor("dmapUseMacro")
	dmapWidthFocus_ : DmapWidthFocusPlug = PlugDescriptor("dmapWidthFocus")
	fogShadowIntensity_ : FogShadowIntensityPlug = PlugDescriptor("fogShadowIntensity")
	lastWrittenDmapAnimExtName_ : LastWrittenDmapAnimExtNamePlug = PlugDescriptor("lastWrittenDmapAnimExtName")
	lightRadius_ : LightRadiusPlug = PlugDescriptor("lightRadius")
	receiveShadows_ : ReceiveShadowsPlug = PlugDescriptor("receiveShadows")
	reuseDmap_ : ReuseDmapPlug = PlugDescriptor("reuseDmap")
	useDepthMapShadows_ : UseDepthMapShadowsPlug = PlugDescriptor("useDepthMapShadows")
	useDmapAutoClipping_ : UseDmapAutoClippingPlug = PlugDescriptor("useDmapAutoClipping")
	useDmapAutoFocus_ : UseDmapAutoFocusPlug = PlugDescriptor("useDmapAutoFocus")
	useMidDistDmap_ : UseMidDistDmapPlug = PlugDescriptor("useMidDistDmap")
	useOnlySingleDmap_ : UseOnlySingleDmapPlug = PlugDescriptor("useOnlySingleDmap")
	useXMinusDmap_ : UseXMinusDmapPlug = PlugDescriptor("useXMinusDmap")
	useXPlusDmap_ : UseXPlusDmapPlug = PlugDescriptor("useXPlusDmap")
	useYMinusDmap_ : UseYMinusDmapPlug = PlugDescriptor("useYMinusDmap")
	useYPlusDmap_ : UseYPlusDmapPlug = PlugDescriptor("useYPlusDmap")
	useZMinusDmap_ : UseZMinusDmapPlug = PlugDescriptor("useZMinusDmap")
	useZPlusDmap_ : UseZPlusDmapPlug = PlugDescriptor("useZPlusDmap")
	volumeShadowSamples_ : VolumeShadowSamplesPlug = PlugDescriptor("volumeShadowSamples")
	writeDmap_ : WriteDmapPlug = PlugDescriptor("writeDmap")

	# node attributes

	typeName = "nonExtendedLightShapeNode"
	typeIdInt = 1313167444
	pass

