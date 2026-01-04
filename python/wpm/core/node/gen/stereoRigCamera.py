

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Camera = Catalogue.Camera
else:
	from .. import retriever
	Camera = retriever.getNodeCls("Camera")
	assert Camera

# add node doc



# region plug type defs
class DisplayFarClipPlug(Plug):
	node : StereoRigCamera = None
	pass
class DisplayFrustumPlug(Plug):
	node : StereoRigCamera = None
	pass
class DisplayNearClipPlug(Plug):
	node : StereoRigCamera = None
	pass
class FilmOffsetLeftCamPlug(Plug):
	node : StereoRigCamera = None
	pass
class FilmOffsetRightCamPlug(Plug):
	node : StereoRigCamera = None
	pass
class InteraxialSeparationPlug(Plug):
	node : StereoRigCamera = None
	pass
class SafeStereoPlug(Plug):
	node : StereoRigCamera = None
	pass
class SafeViewingVolumePlug(Plug):
	node : StereoRigCamera = None
	pass
class SafeVolumeColorBluePlug(Plug):
	parent : SafeVolumeColorPlug = PlugDescriptor("safeVolumeColor")
	node : StereoRigCamera = None
	pass
class SafeVolumeColorGreenPlug(Plug):
	parent : SafeVolumeColorPlug = PlugDescriptor("safeVolumeColor")
	node : StereoRigCamera = None
	pass
class SafeVolumeColorRedPlug(Plug):
	parent : SafeVolumeColorPlug = PlugDescriptor("safeVolumeColor")
	node : StereoRigCamera = None
	pass
class SafeVolumeColorPlug(Plug):
	safeVolumeColorBlue_ : SafeVolumeColorBluePlug = PlugDescriptor("safeVolumeColorBlue")
	svcb_ : SafeVolumeColorBluePlug = PlugDescriptor("safeVolumeColorBlue")
	safeVolumeColorGreen_ : SafeVolumeColorGreenPlug = PlugDescriptor("safeVolumeColorGreen")
	svcg_ : SafeVolumeColorGreenPlug = PlugDescriptor("safeVolumeColorGreen")
	safeVolumeColorRed_ : SafeVolumeColorRedPlug = PlugDescriptor("safeVolumeColorRed")
	svcr_ : SafeVolumeColorRedPlug = PlugDescriptor("safeVolumeColorRed")
	node : StereoRigCamera = None
	pass
class SafeVolumeTransparencyPlug(Plug):
	node : StereoRigCamera = None
	pass
class StereoPlug(Plug):
	node : StereoRigCamera = None
	pass
class ToeInAdjustPlug(Plug):
	node : StereoRigCamera = None
	pass
class ZeroParallaxPlug(Plug):
	node : StereoRigCamera = None
	pass
class ZeroParallaxColorBluePlug(Plug):
	parent : ZeroParallaxColorPlug = PlugDescriptor("zeroParallaxColor")
	node : StereoRigCamera = None
	pass
class ZeroParallaxColorGreenPlug(Plug):
	parent : ZeroParallaxColorPlug = PlugDescriptor("zeroParallaxColor")
	node : StereoRigCamera = None
	pass
class ZeroParallaxColorRedPlug(Plug):
	parent : ZeroParallaxColorPlug = PlugDescriptor("zeroParallaxColor")
	node : StereoRigCamera = None
	pass
class ZeroParallaxColorPlug(Plug):
	zeroParallaxColorBlue_ : ZeroParallaxColorBluePlug = PlugDescriptor("zeroParallaxColorBlue")
	zpcb_ : ZeroParallaxColorBluePlug = PlugDescriptor("zeroParallaxColorBlue")
	zeroParallaxColorGreen_ : ZeroParallaxColorGreenPlug = PlugDescriptor("zeroParallaxColorGreen")
	zpcg_ : ZeroParallaxColorGreenPlug = PlugDescriptor("zeroParallaxColorGreen")
	zeroParallaxColorRed_ : ZeroParallaxColorRedPlug = PlugDescriptor("zeroParallaxColorRed")
	zpcr_ : ZeroParallaxColorRedPlug = PlugDescriptor("zeroParallaxColorRed")
	node : StereoRigCamera = None
	pass
class ZeroParallaxPlanePlug(Plug):
	node : StereoRigCamera = None
	pass
class ZeroParallaxTransparencyPlug(Plug):
	node : StereoRigCamera = None
	pass
# endregion


# define node class
class StereoRigCamera(Camera):
	displayFarClip_ : DisplayFarClipPlug = PlugDescriptor("displayFarClip")
	displayFrustum_ : DisplayFrustumPlug = PlugDescriptor("displayFrustum")
	displayNearClip_ : DisplayNearClipPlug = PlugDescriptor("displayNearClip")
	filmOffsetLeftCam_ : FilmOffsetLeftCamPlug = PlugDescriptor("filmOffsetLeftCam")
	filmOffsetRightCam_ : FilmOffsetRightCamPlug = PlugDescriptor("filmOffsetRightCam")
	interaxialSeparation_ : InteraxialSeparationPlug = PlugDescriptor("interaxialSeparation")
	safeStereo_ : SafeStereoPlug = PlugDescriptor("safeStereo")
	safeViewingVolume_ : SafeViewingVolumePlug = PlugDescriptor("safeViewingVolume")
	safeVolumeColorBlue_ : SafeVolumeColorBluePlug = PlugDescriptor("safeVolumeColorBlue")
	safeVolumeColorGreen_ : SafeVolumeColorGreenPlug = PlugDescriptor("safeVolumeColorGreen")
	safeVolumeColorRed_ : SafeVolumeColorRedPlug = PlugDescriptor("safeVolumeColorRed")
	safeVolumeColor_ : SafeVolumeColorPlug = PlugDescriptor("safeVolumeColor")
	safeVolumeTransparency_ : SafeVolumeTransparencyPlug = PlugDescriptor("safeVolumeTransparency")
	stereo_ : StereoPlug = PlugDescriptor("stereo")
	toeInAdjust_ : ToeInAdjustPlug = PlugDescriptor("toeInAdjust")
	zeroParallax_ : ZeroParallaxPlug = PlugDescriptor("zeroParallax")
	zeroParallaxColorBlue_ : ZeroParallaxColorBluePlug = PlugDescriptor("zeroParallaxColorBlue")
	zeroParallaxColorGreen_ : ZeroParallaxColorGreenPlug = PlugDescriptor("zeroParallaxColorGreen")
	zeroParallaxColorRed_ : ZeroParallaxColorRedPlug = PlugDescriptor("zeroParallaxColorRed")
	zeroParallaxColor_ : ZeroParallaxColorPlug = PlugDescriptor("zeroParallaxColor")
	zeroParallaxPlane_ : ZeroParallaxPlanePlug = PlugDescriptor("zeroParallaxPlane")
	zeroParallaxTransparency_ : ZeroParallaxTransparencyPlug = PlugDescriptor("zeroParallaxTransparency")

	# node attributes

	typeName = "stereoRigCamera"
	typeIdInt = 1397900097
	nodeLeafClassAttrs = ["displayFarClip", "displayFrustum", "displayNearClip", "filmOffsetLeftCam", "filmOffsetRightCam", "interaxialSeparation", "safeStereo", "safeViewingVolume", "safeVolumeColorBlue", "safeVolumeColorGreen", "safeVolumeColorRed", "safeVolumeColor", "safeVolumeTransparency", "stereo", "toeInAdjust", "zeroParallax", "zeroParallaxColorBlue", "zeroParallaxColorGreen", "zeroParallaxColorRed", "zeroParallaxColor", "zeroParallaxPlane", "zeroParallaxTransparency"]
	nodeLeafPlugs = ["displayFarClip", "displayFrustum", "displayNearClip", "filmOffsetLeftCam", "filmOffsetRightCam", "interaxialSeparation", "safeStereo", "safeViewingVolume", "safeVolumeColor", "safeVolumeTransparency", "stereo", "toeInAdjust", "zeroParallax", "zeroParallaxColor", "zeroParallaxPlane", "zeroParallaxTransparency"]
	pass

