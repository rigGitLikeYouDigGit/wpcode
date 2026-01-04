

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	WeightGeometryFilter = Catalogue.WeightGeometryFilter
else:
	from .. import retriever
	WeightGeometryFilter = retriever.getNodeCls("WeightGeometryFilter")
	assert WeightGeometryFilter

# add node doc



# region plug type defs
class AttachmentModePlug(Plug):
	node : Solidify = None
	pass
class BorderFalloffBlurPlug(Plug):
	node : Solidify = None
	pass
class CacheSetupPlug(Plug):
	node : Solidify = None
	pass
class IslandsPlug(Plug):
	node : Solidify = None
	pass
class NormalScalePlug(Plug):
	node : Solidify = None
	pass
class ScaleEnvelopePlug(Plug):
	node : Solidify = None
	pass
class ScaleModePlug(Plug):
	node : Solidify = None
	pass
class StabilizationLevelPlug(Plug):
	node : Solidify = None
	pass
class TangentPlaneScalePlug(Plug):
	node : Solidify = None
	pass
class UseBorderFalloffPlug(Plug):
	node : Solidify = None
	pass
# endregion


# define node class
class Solidify(WeightGeometryFilter):
	attachmentMode_ : AttachmentModePlug = PlugDescriptor("attachmentMode")
	borderFalloffBlur_ : BorderFalloffBlurPlug = PlugDescriptor("borderFalloffBlur")
	cacheSetup_ : CacheSetupPlug = PlugDescriptor("cacheSetup")
	islands_ : IslandsPlug = PlugDescriptor("islands")
	normalScale_ : NormalScalePlug = PlugDescriptor("normalScale")
	scaleEnvelope_ : ScaleEnvelopePlug = PlugDescriptor("scaleEnvelope")
	scaleMode_ : ScaleModePlug = PlugDescriptor("scaleMode")
	stabilizationLevel_ : StabilizationLevelPlug = PlugDescriptor("stabilizationLevel")
	tangentPlaneScale_ : TangentPlaneScalePlug = PlugDescriptor("tangentPlaneScale")
	useBorderFalloff_ : UseBorderFalloffPlug = PlugDescriptor("useBorderFalloff")

	# node attributes

	typeName = "solidify"
	apiTypeInt = 353
	apiTypeStr = "kSolidify"
	typeIdInt = 1397705817
	MFnCls = om.MFnGeometryFilter
	nodeLeafClassAttrs = ["attachmentMode", "borderFalloffBlur", "cacheSetup", "islands", "normalScale", "scaleEnvelope", "scaleMode", "stabilizationLevel", "tangentPlaneScale", "useBorderFalloff"]
	nodeLeafPlugs = ["attachmentMode", "borderFalloffBlur", "cacheSetup", "islands", "normalScale", "scaleEnvelope", "scaleMode", "stabilizationLevel", "tangentPlaneScale", "useBorderFalloff"]
	pass

