

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
	node : CameraView = None
	pass
class CenterOfInterestXPlug(Plug):
	parent : CenterOfInterestPlug = PlugDescriptor("centerOfInterest")
	node : CameraView = None
	pass
class CenterOfInterestYPlug(Plug):
	parent : CenterOfInterestPlug = PlugDescriptor("centerOfInterest")
	node : CameraView = None
	pass
class CenterOfInterestZPlug(Plug):
	parent : CenterOfInterestPlug = PlugDescriptor("centerOfInterest")
	node : CameraView = None
	pass
class CenterOfInterestPlug(Plug):
	centerOfInterestX_ : CenterOfInterestXPlug = PlugDescriptor("centerOfInterestX")
	cx_ : CenterOfInterestXPlug = PlugDescriptor("centerOfInterestX")
	centerOfInterestY_ : CenterOfInterestYPlug = PlugDescriptor("centerOfInterestY")
	cy_ : CenterOfInterestYPlug = PlugDescriptor("centerOfInterestY")
	centerOfInterestZ_ : CenterOfInterestZPlug = PlugDescriptor("centerOfInterestZ")
	cz_ : CenterOfInterestZPlug = PlugDescriptor("centerOfInterestZ")
	node : CameraView = None
	pass
class DescriptionPlug(Plug):
	node : CameraView = None
	pass
class EyeXPlug(Plug):
	parent : EyePlug = PlugDescriptor("eye")
	node : CameraView = None
	pass
class EyeYPlug(Plug):
	parent : EyePlug = PlugDescriptor("eye")
	node : CameraView = None
	pass
class EyeZPlug(Plug):
	parent : EyePlug = PlugDescriptor("eye")
	node : CameraView = None
	pass
class EyePlug(Plug):
	eyeX_ : EyeXPlug = PlugDescriptor("eyeX")
	ex_ : EyeXPlug = PlugDescriptor("eyeX")
	eyeY_ : EyeYPlug = PlugDescriptor("eyeY")
	ey_ : EyeYPlug = PlugDescriptor("eyeY")
	eyeZ_ : EyeZPlug = PlugDescriptor("eyeZ")
	ez_ : EyeZPlug = PlugDescriptor("eyeZ")
	node : CameraView = None
	pass
class FocalLengthPlug(Plug):
	node : CameraView = None
	pass
class HorizontalAperturePlug(Plug):
	node : CameraView = None
	pass
class HorizontalPanPlug(Plug):
	node : CameraView = None
	pass
class OrthographicPlug(Plug):
	node : CameraView = None
	pass
class OrthographicWidthPlug(Plug):
	node : CameraView = None
	pass
class PanZoomEnabledPlug(Plug):
	node : CameraView = None
	pass
class RenderPanZoomPlug(Plug):
	node : CameraView = None
	pass
class TumblePivotXPlug(Plug):
	parent : TumblePivotPlug = PlugDescriptor("tumblePivot")
	node : CameraView = None
	pass
class TumblePivotYPlug(Plug):
	parent : TumblePivotPlug = PlugDescriptor("tumblePivot")
	node : CameraView = None
	pass
class TumblePivotZPlug(Plug):
	parent : TumblePivotPlug = PlugDescriptor("tumblePivot")
	node : CameraView = None
	pass
class TumblePivotPlug(Plug):
	tumblePivotX_ : TumblePivotXPlug = PlugDescriptor("tumblePivotX")
	tpx_ : TumblePivotXPlug = PlugDescriptor("tumblePivotX")
	tumblePivotY_ : TumblePivotYPlug = PlugDescriptor("tumblePivotY")
	tpy_ : TumblePivotYPlug = PlugDescriptor("tumblePivotY")
	tumblePivotZ_ : TumblePivotZPlug = PlugDescriptor("tumblePivotZ")
	tpz_ : TumblePivotZPlug = PlugDescriptor("tumblePivotZ")
	node : CameraView = None
	pass
class UpXPlug(Plug):
	parent : UpPlug = PlugDescriptor("up")
	node : CameraView = None
	pass
class UpYPlug(Plug):
	parent : UpPlug = PlugDescriptor("up")
	node : CameraView = None
	pass
class UpZPlug(Plug):
	parent : UpPlug = PlugDescriptor("up")
	node : CameraView = None
	pass
class UpPlug(Plug):
	upX_ : UpXPlug = PlugDescriptor("upX")
	ux_ : UpXPlug = PlugDescriptor("upX")
	upY_ : UpYPlug = PlugDescriptor("upY")
	uy_ : UpYPlug = PlugDescriptor("upY")
	upZ_ : UpZPlug = PlugDescriptor("upZ")
	uz_ : UpZPlug = PlugDescriptor("upZ")
	node : CameraView = None
	pass
class VerticalAperturePlug(Plug):
	node : CameraView = None
	pass
class VerticalPanPlug(Plug):
	node : CameraView = None
	pass
class ViewTypePlug(Plug):
	node : CameraView = None
	pass
class ZoomPlug(Plug):
	node : CameraView = None
	pass
# endregion


# define node class
class CameraView(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	centerOfInterestX_ : CenterOfInterestXPlug = PlugDescriptor("centerOfInterestX")
	centerOfInterestY_ : CenterOfInterestYPlug = PlugDescriptor("centerOfInterestY")
	centerOfInterestZ_ : CenterOfInterestZPlug = PlugDescriptor("centerOfInterestZ")
	centerOfInterest_ : CenterOfInterestPlug = PlugDescriptor("centerOfInterest")
	description_ : DescriptionPlug = PlugDescriptor("description")
	eyeX_ : EyeXPlug = PlugDescriptor("eyeX")
	eyeY_ : EyeYPlug = PlugDescriptor("eyeY")
	eyeZ_ : EyeZPlug = PlugDescriptor("eyeZ")
	eye_ : EyePlug = PlugDescriptor("eye")
	focalLength_ : FocalLengthPlug = PlugDescriptor("focalLength")
	horizontalAperture_ : HorizontalAperturePlug = PlugDescriptor("horizontalAperture")
	horizontalPan_ : HorizontalPanPlug = PlugDescriptor("horizontalPan")
	orthographic_ : OrthographicPlug = PlugDescriptor("orthographic")
	orthographicWidth_ : OrthographicWidthPlug = PlugDescriptor("orthographicWidth")
	panZoomEnabled_ : PanZoomEnabledPlug = PlugDescriptor("panZoomEnabled")
	renderPanZoom_ : RenderPanZoomPlug = PlugDescriptor("renderPanZoom")
	tumblePivotX_ : TumblePivotXPlug = PlugDescriptor("tumblePivotX")
	tumblePivotY_ : TumblePivotYPlug = PlugDescriptor("tumblePivotY")
	tumblePivotZ_ : TumblePivotZPlug = PlugDescriptor("tumblePivotZ")
	tumblePivot_ : TumblePivotPlug = PlugDescriptor("tumblePivot")
	upX_ : UpXPlug = PlugDescriptor("upX")
	upY_ : UpYPlug = PlugDescriptor("upY")
	upZ_ : UpZPlug = PlugDescriptor("upZ")
	up_ : UpPlug = PlugDescriptor("up")
	verticalAperture_ : VerticalAperturePlug = PlugDescriptor("verticalAperture")
	verticalPan_ : VerticalPanPlug = PlugDescriptor("verticalPan")
	viewType_ : ViewTypePlug = PlugDescriptor("viewType")
	zoom_ : ZoomPlug = PlugDescriptor("zoom")

	# node attributes

	typeName = "cameraView"
	apiTypeInt = 34
	apiTypeStr = "kCameraView"
	typeIdInt = 1145258326
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "centerOfInterestX", "centerOfInterestY", "centerOfInterestZ", "centerOfInterest", "description", "eyeX", "eyeY", "eyeZ", "eye", "focalLength", "horizontalAperture", "horizontalPan", "orthographic", "orthographicWidth", "panZoomEnabled", "renderPanZoom", "tumblePivotX", "tumblePivotY", "tumblePivotZ", "tumblePivot", "upX", "upY", "upZ", "up", "verticalAperture", "verticalPan", "viewType", "zoom"]
	nodeLeafPlugs = ["binMembership", "centerOfInterest", "description", "eye", "focalLength", "horizontalAperture", "horizontalPan", "orthographic", "orthographicWidth", "panZoomEnabled", "renderPanZoom", "tumblePivot", "up", "verticalAperture", "verticalPan", "viewType", "zoom"]
	pass

