

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
	node : RenderQuality = None
	pass
class BlueThresholdPlug(Plug):
	node : RenderQuality = None
	pass
class CoverageThresholdPlug(Plug):
	node : RenderQuality = None
	pass
class EdgeAntiAliasingPlug(Plug):
	node : RenderQuality = None
	pass
class EnableRaytracingPlug(Plug):
	node : RenderQuality = None
	pass
class GreenThresholdPlug(Plug):
	node : RenderQuality = None
	pass
class MaxShadingSamplesPlug(Plug):
	node : RenderQuality = None
	pass
class MaxVisibilitySamplesPlug(Plug):
	node : RenderQuality = None
	pass
class ParticleSamplesPlug(Plug):
	node : RenderQuality = None
	pass
class PixelFilterTypePlug(Plug):
	node : RenderQuality = None
	pass
class PixelFilterWidthXPlug(Plug):
	node : RenderQuality = None
	pass
class PixelFilterWidthYPlug(Plug):
	node : RenderQuality = None
	pass
class PlugInFilterWeightPlug(Plug):
	node : RenderQuality = None
	pass
class RayTraceBiasPlug(Plug):
	node : RenderQuality = None
	pass
class RedThresholdPlug(Plug):
	node : RenderQuality = None
	pass
class ReflectionsPlug(Plug):
	node : RenderQuality = None
	pass
class RefractionsPlug(Plug):
	node : RenderQuality = None
	pass
class RenderSamplePlug(Plug):
	node : RenderQuality = None
	pass
class ShadingSamplesPlug(Plug):
	node : RenderQuality = None
	pass
class ShadowsPlug(Plug):
	node : RenderQuality = None
	pass
class UseMultiPixelFilterPlug(Plug):
	node : RenderQuality = None
	pass
class VisibilitySamplesPlug(Plug):
	node : RenderQuality = None
	pass
class VolumeSamplesPlug(Plug):
	node : RenderQuality = None
	pass
# endregion


# define node class
class RenderQuality(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	blueThreshold_ : BlueThresholdPlug = PlugDescriptor("blueThreshold")
	coverageThreshold_ : CoverageThresholdPlug = PlugDescriptor("coverageThreshold")
	edgeAntiAliasing_ : EdgeAntiAliasingPlug = PlugDescriptor("edgeAntiAliasing")
	enableRaytracing_ : EnableRaytracingPlug = PlugDescriptor("enableRaytracing")
	greenThreshold_ : GreenThresholdPlug = PlugDescriptor("greenThreshold")
	maxShadingSamples_ : MaxShadingSamplesPlug = PlugDescriptor("maxShadingSamples")
	maxVisibilitySamples_ : MaxVisibilitySamplesPlug = PlugDescriptor("maxVisibilitySamples")
	particleSamples_ : ParticleSamplesPlug = PlugDescriptor("particleSamples")
	pixelFilterType_ : PixelFilterTypePlug = PlugDescriptor("pixelFilterType")
	pixelFilterWidthX_ : PixelFilterWidthXPlug = PlugDescriptor("pixelFilterWidthX")
	pixelFilterWidthY_ : PixelFilterWidthYPlug = PlugDescriptor("pixelFilterWidthY")
	plugInFilterWeight_ : PlugInFilterWeightPlug = PlugDescriptor("plugInFilterWeight")
	rayTraceBias_ : RayTraceBiasPlug = PlugDescriptor("rayTraceBias")
	redThreshold_ : RedThresholdPlug = PlugDescriptor("redThreshold")
	reflections_ : ReflectionsPlug = PlugDescriptor("reflections")
	refractions_ : RefractionsPlug = PlugDescriptor("refractions")
	renderSample_ : RenderSamplePlug = PlugDescriptor("renderSample")
	shadingSamples_ : ShadingSamplesPlug = PlugDescriptor("shadingSamples")
	shadows_ : ShadowsPlug = PlugDescriptor("shadows")
	useMultiPixelFilter_ : UseMultiPixelFilterPlug = PlugDescriptor("useMultiPixelFilter")
	visibilitySamples_ : VisibilitySamplesPlug = PlugDescriptor("visibilitySamples")
	volumeSamples_ : VolumeSamplesPlug = PlugDescriptor("volumeSamples")

	# node attributes

	typeName = "renderQuality"
	apiTypeInt = 525
	apiTypeStr = "kRenderQuality"
	typeIdInt = 1381061953
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "blueThreshold", "coverageThreshold", "edgeAntiAliasing", "enableRaytracing", "greenThreshold", "maxShadingSamples", "maxVisibilitySamples", "particleSamples", "pixelFilterType", "pixelFilterWidthX", "pixelFilterWidthY", "plugInFilterWeight", "rayTraceBias", "redThreshold", "reflections", "refractions", "renderSample", "shadingSamples", "shadows", "useMultiPixelFilter", "visibilitySamples", "volumeSamples"]
	nodeLeafPlugs = ["binMembership", "blueThreshold", "coverageThreshold", "edgeAntiAliasing", "enableRaytracing", "greenThreshold", "maxShadingSamples", "maxVisibilitySamples", "particleSamples", "pixelFilterType", "pixelFilterWidthX", "pixelFilterWidthY", "plugInFilterWeight", "rayTraceBias", "redThreshold", "reflections", "refractions", "renderSample", "shadingSamples", "shadows", "useMultiPixelFilter", "visibilitySamples", "volumeSamples"]
	pass

