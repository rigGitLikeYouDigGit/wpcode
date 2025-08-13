

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ImageSource = retriever.getNodeCls("ImageSource")
assert ImageSource
if T.TYPE_CHECKING:
	from .. import ImageSource

# add node doc



# region plug type defs
class PlugPlug(Plug):
	parent : AdjustmentsPlug = PlugDescriptor("adjustments")
	node : RenderLayer = None
	pass
class ValuePlug(Plug):
	parent : AdjustmentsPlug = PlugDescriptor("adjustments")
	node : RenderLayer = None
	pass
class AdjustmentsPlug(Plug):
	plug_ : PlugPlug = PlugDescriptor("plug")
	plg_ : PlugPlug = PlugDescriptor("plug")
	value_ : ValuePlug = PlugDescriptor("value")
	val_ : ValuePlug = PlugDescriptor("value")
	node : RenderLayer = None
	pass
class AttributeOverrideScriptPlug(Plug):
	node : RenderLayer = None
	pass
class DeferredOutAdjustmentsPlug(Plug):
	node : RenderLayer = None
	pass
class DeferredUndoOutAdjustmentsPlug(Plug):
	node : RenderLayer = None
	pass
class DisplayOrderPlug(Plug):
	node : RenderLayer = None
	pass
class GlobalPlug(Plug):
	node : RenderLayer = None
	pass
class ImageNamePlug(Plug):
	node : RenderLayer = None
	pass
class ImageRenderedPlug(Plug):
	node : RenderLayer = None
	pass
class IsDefaultPrecompTemplateOverridePlug(Plug):
	node : RenderLayer = None
	pass
class LayerChildrenPlug(Plug):
	node : RenderLayer = None
	pass
class LayerParentPlug(Plug):
	node : RenderLayer = None
	pass
class OutIdPlug(Plug):
	parent : OutAdjustmentsPlug = PlugDescriptor("outAdjustments")
	node : RenderLayer = None
	pass
class OutPlugPlug(Plug):
	parent : OutAdjustmentsPlug = PlugDescriptor("outAdjustments")
	node : RenderLayer = None
	pass
class OutValuePlug(Plug):
	parent : OutAdjustmentsPlug = PlugDescriptor("outAdjustments")
	node : RenderLayer = None
	pass
class OutAdjustmentsPlug(Plug):
	outId_ : OutIdPlug = PlugDescriptor("outId")
	oaid_ : OutIdPlug = PlugDescriptor("outId")
	outPlug_ : OutPlugPlug = PlugDescriptor("outPlug")
	opg_ : OutPlugPlug = PlugDescriptor("outPlug")
	outValue_ : OutValuePlug = PlugDescriptor("outValue")
	ovl_ : OutValuePlug = PlugDescriptor("outValue")
	node : RenderLayer = None
	pass
class PassContributionMapPlug(Plug):
	node : RenderLayer = None
	pass
class PrecompTemplatePlug(Plug):
	node : RenderLayer = None
	pass
class PsdAlphaChannelPlug(Plug):
	node : RenderLayer = None
	pass
class PsdBlendModePlug(Plug):
	node : RenderLayer = None
	pass
class RecycleImagePlug(Plug):
	node : RenderLayer = None
	pass
class DrawColorPlug(Plug):
	parent : RenderInfoPlug = PlugDescriptor("renderInfo")
	node : RenderLayer = None
	pass
class IdentificationPlug(Plug):
	parent : RenderInfoPlug = PlugDescriptor("renderInfo")
	node : RenderLayer = None
	pass
class RenderablePlug(Plug):
	parent : RenderInfoPlug = PlugDescriptor("renderInfo")
	node : RenderLayer = None
	pass
class RenderInfoPlug(Plug):
	drawColor_ : DrawColorPlug = PlugDescriptor("drawColor")
	c_ : DrawColorPlug = PlugDescriptor("drawColor")
	identification_ : IdentificationPlug = PlugDescriptor("identification")
	rlid_ : IdentificationPlug = PlugDescriptor("identification")
	renderable_ : RenderablePlug = PlugDescriptor("renderable")
	rndr_ : RenderablePlug = PlugDescriptor("renderable")
	node : RenderLayer = None
	pass
class RenderPassPlug(Plug):
	node : RenderLayer = None
	pass
class AmbientPlug(Plug):
	parent : RenderPassInfoPlug = PlugDescriptor("renderPassInfo")
	node : RenderLayer = None
	pass
class BeautyPlug(Plug):
	parent : RenderPassInfoPlug = PlugDescriptor("renderPassInfo")
	node : RenderLayer = None
	pass
class ColorPlug(Plug):
	parent : RenderPassInfoPlug = PlugDescriptor("renderPassInfo")
	node : RenderLayer = None
	pass
class DiffusePlug(Plug):
	parent : RenderPassInfoPlug = PlugDescriptor("renderPassInfo")
	node : RenderLayer = None
	pass
class ShadowPlug(Plug):
	parent : RenderPassInfoPlug = PlugDescriptor("renderPassInfo")
	node : RenderLayer = None
	pass
class SpecularPlug(Plug):
	parent : RenderPassInfoPlug = PlugDescriptor("renderPassInfo")
	node : RenderLayer = None
	pass
class RenderPassInfoPlug(Plug):
	ambient_ : AmbientPlug = PlugDescriptor("ambient")
	am_ : AmbientPlug = PlugDescriptor("ambient")
	beauty_ : BeautyPlug = PlugDescriptor("beauty")
	b_ : BeautyPlug = PlugDescriptor("beauty")
	color_ : ColorPlug = PlugDescriptor("color")
	cp_ : ColorPlug = PlugDescriptor("color")
	diffuse_ : DiffusePlug = PlugDescriptor("diffuse")
	di_ : DiffusePlug = PlugDescriptor("diffuse")
	shadow_ : ShadowPlug = PlugDescriptor("shadow")
	s_ : ShadowPlug = PlugDescriptor("shadow")
	specular_ : SpecularPlug = PlugDescriptor("specular")
	sp_ : SpecularPlug = PlugDescriptor("specular")
	node : RenderLayer = None
	pass
class ShadingGroupOverridePlug(Plug):
	node : RenderLayer = None
	pass
# endregion


# define node class
class RenderLayer(ImageSource):
	plug_ : PlugPlug = PlugDescriptor("plug")
	value_ : ValuePlug = PlugDescriptor("value")
	adjustments_ : AdjustmentsPlug = PlugDescriptor("adjustments")
	attributeOverrideScript_ : AttributeOverrideScriptPlug = PlugDescriptor("attributeOverrideScript")
	deferredOutAdjustments_ : DeferredOutAdjustmentsPlug = PlugDescriptor("deferredOutAdjustments")
	deferredUndoOutAdjustments_ : DeferredUndoOutAdjustmentsPlug = PlugDescriptor("deferredUndoOutAdjustments")
	displayOrder_ : DisplayOrderPlug = PlugDescriptor("displayOrder")
	global_ : GlobalPlug = PlugDescriptor("global")
	imageName_ : ImageNamePlug = PlugDescriptor("imageName")
	imageRendered_ : ImageRenderedPlug = PlugDescriptor("imageRendered")
	isDefaultPrecompTemplateOverride_ : IsDefaultPrecompTemplateOverridePlug = PlugDescriptor("isDefaultPrecompTemplateOverride")
	layerChildren_ : LayerChildrenPlug = PlugDescriptor("layerChildren")
	layerParent_ : LayerParentPlug = PlugDescriptor("layerParent")
	outId_ : OutIdPlug = PlugDescriptor("outId")
	outPlug_ : OutPlugPlug = PlugDescriptor("outPlug")
	outValue_ : OutValuePlug = PlugDescriptor("outValue")
	outAdjustments_ : OutAdjustmentsPlug = PlugDescriptor("outAdjustments")
	passContributionMap_ : PassContributionMapPlug = PlugDescriptor("passContributionMap")
	precompTemplate_ : PrecompTemplatePlug = PlugDescriptor("precompTemplate")
	psdAlphaChannel_ : PsdAlphaChannelPlug = PlugDescriptor("psdAlphaChannel")
	psdBlendMode_ : PsdBlendModePlug = PlugDescriptor("psdBlendMode")
	recycleImage_ : RecycleImagePlug = PlugDescriptor("recycleImage")
	drawColor_ : DrawColorPlug = PlugDescriptor("drawColor")
	identification_ : IdentificationPlug = PlugDescriptor("identification")
	renderable_ : RenderablePlug = PlugDescriptor("renderable")
	renderInfo_ : RenderInfoPlug = PlugDescriptor("renderInfo")
	renderPass_ : RenderPassPlug = PlugDescriptor("renderPass")
	ambient_ : AmbientPlug = PlugDescriptor("ambient")
	beauty_ : BeautyPlug = PlugDescriptor("beauty")
	color_ : ColorPlug = PlugDescriptor("color")
	diffuse_ : DiffusePlug = PlugDescriptor("diffuse")
	shadow_ : ShadowPlug = PlugDescriptor("shadow")
	specular_ : SpecularPlug = PlugDescriptor("specular")
	renderPassInfo_ : RenderPassInfoPlug = PlugDescriptor("renderPassInfo")
	shadingGroupOverride_ : ShadingGroupOverridePlug = PlugDescriptor("shadingGroupOverride")

	# node attributes

	typeName = "renderLayer"
	apiTypeInt = 785
	apiTypeStr = "kRenderLayer"
	typeIdInt = 1380861004
	MFnCls = om.MFnDependencyNode
	pass

