

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
class AlphaSourcePlug(Plug):
	node : HwRenderGlobals = None
	pass
class AntiAliasPolygonsPlug(Plug):
	node : HwRenderGlobals = None
	pass
class BackgroundColorBPlug(Plug):
	parent : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	node : HwRenderGlobals = None
	pass
class BackgroundColorGPlug(Plug):
	parent : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	node : HwRenderGlobals = None
	pass
class BackgroundColorRPlug(Plug):
	parent : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	node : HwRenderGlobals = None
	pass
class BackgroundColorPlug(Plug):
	backgroundColorB_ : BackgroundColorBPlug = PlugDescriptor("backgroundColorB")
	bcb_ : BackgroundColorBPlug = PlugDescriptor("backgroundColorB")
	backgroundColorG_ : BackgroundColorGPlug = PlugDescriptor("backgroundColorG")
	bcg_ : BackgroundColorGPlug = PlugDescriptor("backgroundColorG")
	backgroundColorR_ : BackgroundColorRPlug = PlugDescriptor("backgroundColorR")
	bcr_ : BackgroundColorRPlug = PlugDescriptor("backgroundColorR")
	node : HwRenderGlobals = None
	pass
class BinMembershipPlug(Plug):
	node : HwRenderGlobals = None
	pass
class ByFramePlug(Plug):
	node : HwRenderGlobals = None
	pass
class CameraIconsPlug(Plug):
	node : HwRenderGlobals = None
	pass
class CollisionIconsPlug(Plug):
	node : HwRenderGlobals = None
	pass
class DisplayShadowsPlug(Plug):
	node : HwRenderGlobals = None
	pass
class DrawStylePlug(Plug):
	node : HwRenderGlobals = None
	pass
class EdgeSmoothingPlug(Plug):
	node : HwRenderGlobals = None
	pass
class EmitterIconsPlug(Plug):
	node : HwRenderGlobals = None
	pass
class EndFramePlug(Plug):
	node : HwRenderGlobals = None
	pass
class ExtensionPlug(Plug):
	node : HwRenderGlobals = None
	pass
class FieldIconsPlug(Plug):
	node : HwRenderGlobals = None
	pass
class FilenamePlug(Plug):
	node : HwRenderGlobals = None
	pass
class FullImageResolutionPlug(Plug):
	node : HwRenderGlobals = None
	pass
class GeometryMaskPlug(Plug):
	node : HwRenderGlobals = None
	pass
class GridPlug(Plug):
	node : HwRenderGlobals = None
	pass
class ImageFormatPlug(Plug):
	node : HwRenderGlobals = None
	pass
class ImfPluginKeyPlug(Plug):
	node : HwRenderGlobals = None
	pass
class ImfPluginKeyExtPlug(Plug):
	node : HwRenderGlobals = None
	pass
class LightIconsPlug(Plug):
	node : HwRenderGlobals = None
	pass
class LightingModePlug(Plug):
	node : HwRenderGlobals = None
	pass
class LineSmoothingPlug(Plug):
	node : HwRenderGlobals = None
	pass
class MotionBlurPlug(Plug):
	node : HwRenderGlobals = None
	pass
class MultiPassRenderingPlug(Plug):
	node : HwRenderGlobals = None
	pass
class RenderPassesPlug(Plug):
	node : HwRenderGlobals = None
	pass
class ResolutionPlug(Plug):
	node : HwRenderGlobals = None
	pass
class StartFramePlug(Plug):
	node : HwRenderGlobals = None
	pass
class TexturingPlug(Plug):
	node : HwRenderGlobals = None
	pass
class TransformIconsPlug(Plug):
	node : HwRenderGlobals = None
	pass
class WriteZDepthPlug(Plug):
	node : HwRenderGlobals = None
	pass
# endregion


# define node class
class HwRenderGlobals(_BASE_):
	alphaSource_ : AlphaSourcePlug = PlugDescriptor("alphaSource")
	antiAliasPolygons_ : AntiAliasPolygonsPlug = PlugDescriptor("antiAliasPolygons")
	backgroundColorB_ : BackgroundColorBPlug = PlugDescriptor("backgroundColorB")
	backgroundColorG_ : BackgroundColorGPlug = PlugDescriptor("backgroundColorG")
	backgroundColorR_ : BackgroundColorRPlug = PlugDescriptor("backgroundColorR")
	backgroundColor_ : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	byFrame_ : ByFramePlug = PlugDescriptor("byFrame")
	cameraIcons_ : CameraIconsPlug = PlugDescriptor("cameraIcons")
	collisionIcons_ : CollisionIconsPlug = PlugDescriptor("collisionIcons")
	displayShadows_ : DisplayShadowsPlug = PlugDescriptor("displayShadows")
	drawStyle_ : DrawStylePlug = PlugDescriptor("drawStyle")
	edgeSmoothing_ : EdgeSmoothingPlug = PlugDescriptor("edgeSmoothing")
	emitterIcons_ : EmitterIconsPlug = PlugDescriptor("emitterIcons")
	endFrame_ : EndFramePlug = PlugDescriptor("endFrame")
	extension_ : ExtensionPlug = PlugDescriptor("extension")
	fieldIcons_ : FieldIconsPlug = PlugDescriptor("fieldIcons")
	filename_ : FilenamePlug = PlugDescriptor("filename")
	fullImageResolution_ : FullImageResolutionPlug = PlugDescriptor("fullImageResolution")
	geometryMask_ : GeometryMaskPlug = PlugDescriptor("geometryMask")
	grid_ : GridPlug = PlugDescriptor("grid")
	imageFormat_ : ImageFormatPlug = PlugDescriptor("imageFormat")
	imfPluginKey_ : ImfPluginKeyPlug = PlugDescriptor("imfPluginKey")
	imfPluginKeyExt_ : ImfPluginKeyExtPlug = PlugDescriptor("imfPluginKeyExt")
	lightIcons_ : LightIconsPlug = PlugDescriptor("lightIcons")
	lightingMode_ : LightingModePlug = PlugDescriptor("lightingMode")
	lineSmoothing_ : LineSmoothingPlug = PlugDescriptor("lineSmoothing")
	motionBlur_ : MotionBlurPlug = PlugDescriptor("motionBlur")
	multiPassRendering_ : MultiPassRenderingPlug = PlugDescriptor("multiPassRendering")
	renderPasses_ : RenderPassesPlug = PlugDescriptor("renderPasses")
	resolution_ : ResolutionPlug = PlugDescriptor("resolution")
	startFrame_ : StartFramePlug = PlugDescriptor("startFrame")
	texturing_ : TexturingPlug = PlugDescriptor("texturing")
	transformIcons_ : TransformIconsPlug = PlugDescriptor("transformIcons")
	writeZDepth_ : WriteZDepthPlug = PlugDescriptor("writeZDepth")

	# node attributes

	typeName = "hwRenderGlobals"
	typeIdInt = 1497911876
	nodeLeafClassAttrs = ["alphaSource", "antiAliasPolygons", "backgroundColorB", "backgroundColorG", "backgroundColorR", "backgroundColor", "binMembership", "byFrame", "cameraIcons", "collisionIcons", "displayShadows", "drawStyle", "edgeSmoothing", "emitterIcons", "endFrame", "extension", "fieldIcons", "filename", "fullImageResolution", "geometryMask", "grid", "imageFormat", "imfPluginKey", "imfPluginKeyExt", "lightIcons", "lightingMode", "lineSmoothing", "motionBlur", "multiPassRendering", "renderPasses", "resolution", "startFrame", "texturing", "transformIcons", "writeZDepth"]
	nodeLeafPlugs = ["alphaSource", "antiAliasPolygons", "backgroundColor", "binMembership", "byFrame", "cameraIcons", "collisionIcons", "displayShadows", "drawStyle", "edgeSmoothing", "emitterIcons", "endFrame", "extension", "fieldIcons", "filename", "fullImageResolution", "geometryMask", "grid", "imageFormat", "imfPluginKey", "imfPluginKeyExt", "lightIcons", "lightingMode", "lineSmoothing", "motionBlur", "multiPassRendering", "renderPasses", "resolution", "startFrame", "texturing", "transformIcons", "writeZDepth"]
	pass

