

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Manip3D = Catalogue.Manip3D
else:
	from .. import retriever
	Manip3D = retriever.getNodeCls("Manip3D")
	assert Manip3D

# add node doc



# region plug type defs
class PivotRotateXPlug(Plug):
	parent : PivotRotatePlug = PlugDescriptor("pivotRotate")
	node : VolumeBindManip = None
	pass
class PivotRotateYPlug(Plug):
	parent : PivotRotatePlug = PlugDescriptor("pivotRotate")
	node : VolumeBindManip = None
	pass
class PivotRotateZPlug(Plug):
	parent : PivotRotatePlug = PlugDescriptor("pivotRotate")
	node : VolumeBindManip = None
	pass
class PivotRotatePlug(Plug):
	pivotRotateX_ : PivotRotateXPlug = PlugDescriptor("pivotRotateX")
	prx_ : PivotRotateXPlug = PlugDescriptor("pivotRotateX")
	pivotRotateY_ : PivotRotateYPlug = PlugDescriptor("pivotRotateY")
	pry_ : PivotRotateYPlug = PlugDescriptor("pivotRotateY")
	pivotRotateZ_ : PivotRotateZPlug = PlugDescriptor("pivotRotateZ")
	prz_ : PivotRotateZPlug = PlugDescriptor("pivotRotateZ")
	node : VolumeBindManip = None
	pass
class PivotTranslateXPlug(Plug):
	parent : PivotTranslatePlug = PlugDescriptor("pivotTranslate")
	node : VolumeBindManip = None
	pass
class PivotTranslateYPlug(Plug):
	parent : PivotTranslatePlug = PlugDescriptor("pivotTranslate")
	node : VolumeBindManip = None
	pass
class PivotTranslateZPlug(Plug):
	parent : PivotTranslatePlug = PlugDescriptor("pivotTranslate")
	node : VolumeBindManip = None
	pass
class PivotTranslatePlug(Plug):
	pivotTranslateX_ : PivotTranslateXPlug = PlugDescriptor("pivotTranslateX")
	ptx_ : PivotTranslateXPlug = PlugDescriptor("pivotTranslateX")
	pivotTranslateY_ : PivotTranslateYPlug = PlugDescriptor("pivotTranslateY")
	pty_ : PivotTranslateYPlug = PlugDescriptor("pivotTranslateY")
	pivotTranslateZ_ : PivotTranslateZPlug = PlugDescriptor("pivotTranslateZ")
	ptz_ : PivotTranslateZPlug = PlugDescriptor("pivotTranslateZ")
	node : VolumeBindManip = None
	pass
# endregion


# define node class
class VolumeBindManip(Manip3D):
	pivotRotateX_ : PivotRotateXPlug = PlugDescriptor("pivotRotateX")
	pivotRotateY_ : PivotRotateYPlug = PlugDescriptor("pivotRotateY")
	pivotRotateZ_ : PivotRotateZPlug = PlugDescriptor("pivotRotateZ")
	pivotRotate_ : PivotRotatePlug = PlugDescriptor("pivotRotate")
	pivotTranslateX_ : PivotTranslateXPlug = PlugDescriptor("pivotTranslateX")
	pivotTranslateY_ : PivotTranslateYPlug = PlugDescriptor("pivotTranslateY")
	pivotTranslateZ_ : PivotTranslateZPlug = PlugDescriptor("pivotTranslateZ")
	pivotTranslate_ : PivotTranslatePlug = PlugDescriptor("pivotTranslate")

	# node attributes

	typeName = "volumeBindManip"
	apiTypeInt = 1063
	apiTypeStr = "kVolumeBindManip"
	typeIdInt = 1447185986
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = ["pivotRotateX", "pivotRotateY", "pivotRotateZ", "pivotRotate", "pivotTranslateX", "pivotTranslateY", "pivotTranslateZ", "pivotTranslate"]
	nodeLeafPlugs = ["pivotRotate", "pivotTranslate"]
	pass

