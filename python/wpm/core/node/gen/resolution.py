

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
class AspectLockPlug(Plug):
	node : Resolution = None
	pass
class BinMembershipPlug(Plug):
	node : Resolution = None
	pass
class DeviceAspectRatioPlug(Plug):
	node : Resolution = None
	pass
class DotsPerInchPlug(Plug):
	node : Resolution = None
	pass
class FieldsPlug(Plug):
	node : Resolution = None
	pass
class HeightPlug(Plug):
	node : Resolution = None
	pass
class ImageSizeUnitsPlug(Plug):
	node : Resolution = None
	pass
class LockDeviceAspectRatioPlug(Plug):
	node : Resolution = None
	pass
class OddFieldFirstPlug(Plug):
	node : Resolution = None
	pass
class PixelAspectPlug(Plug):
	node : Resolution = None
	pass
class PixelDensityUnitsPlug(Plug):
	node : Resolution = None
	pass
class WidthPlug(Plug):
	node : Resolution = None
	pass
class ZerothScanlinePlug(Plug):
	node : Resolution = None
	pass
# endregion


# define node class
class Resolution(_BASE_):
	aspectLock_ : AspectLockPlug = PlugDescriptor("aspectLock")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	deviceAspectRatio_ : DeviceAspectRatioPlug = PlugDescriptor("deviceAspectRatio")
	dotsPerInch_ : DotsPerInchPlug = PlugDescriptor("dotsPerInch")
	fields_ : FieldsPlug = PlugDescriptor("fields")
	height_ : HeightPlug = PlugDescriptor("height")
	imageSizeUnits_ : ImageSizeUnitsPlug = PlugDescriptor("imageSizeUnits")
	lockDeviceAspectRatio_ : LockDeviceAspectRatioPlug = PlugDescriptor("lockDeviceAspectRatio")
	oddFieldFirst_ : OddFieldFirstPlug = PlugDescriptor("oddFieldFirst")
	pixelAspect_ : PixelAspectPlug = PlugDescriptor("pixelAspect")
	pixelDensityUnits_ : PixelDensityUnitsPlug = PlugDescriptor("pixelDensityUnits")
	width_ : WidthPlug = PlugDescriptor("width")
	zerothScanline_ : ZerothScanlinePlug = PlugDescriptor("zerothScanline")

	# node attributes

	typeName = "resolution"
	apiTypeInt = 526
	apiTypeStr = "kResolution"
	typeIdInt = 1380734030
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["aspectLock", "binMembership", "deviceAspectRatio", "dotsPerInch", "fields", "height", "imageSizeUnits", "lockDeviceAspectRatio", "oddFieldFirst", "pixelAspect", "pixelDensityUnits", "width", "zerothScanline"]
	nodeLeafPlugs = ["aspectLock", "binMembership", "deviceAspectRatio", "dotsPerInch", "fields", "height", "imageSizeUnits", "lockDeviceAspectRatio", "oddFieldFirst", "pixelAspect", "pixelDensityUnits", "width", "zerothScanline"]
	pass

