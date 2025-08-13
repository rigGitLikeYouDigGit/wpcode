

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Texture3d = retriever.getNodeCls("Texture3d")
assert Texture3d
if T.TYPE_CHECKING:
	from .. import Texture3d

# add node doc



# region plug type defs
class IncrementPlug(Plug):
	node : Brownian = None
	pass
class LacunarityPlug(Plug):
	node : Brownian = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Brownian = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Brownian = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Brownian = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : Brownian = None
	pass
class OctavesPlug(Plug):
	node : Brownian = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Brownian = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Brownian = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Brownian = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : Brownian = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Brownian = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Brownian = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Brownian = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	roy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	roz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : Brownian = None
	pass
class Weight3dXPlug(Plug):
	parent : Weight3dPlug = PlugDescriptor("weight3d")
	node : Brownian = None
	pass
class Weight3dYPlug(Plug):
	parent : Weight3dPlug = PlugDescriptor("weight3d")
	node : Brownian = None
	pass
class Weight3dZPlug(Plug):
	parent : Weight3dPlug = PlugDescriptor("weight3d")
	node : Brownian = None
	pass
class Weight3dPlug(Plug):
	weight3dX_ : Weight3dXPlug = PlugDescriptor("weight3dX")
	w3x_ : Weight3dXPlug = PlugDescriptor("weight3dX")
	weight3dY_ : Weight3dYPlug = PlugDescriptor("weight3dY")
	w3y_ : Weight3dYPlug = PlugDescriptor("weight3dY")
	weight3dZ_ : Weight3dZPlug = PlugDescriptor("weight3dZ")
	w3z_ : Weight3dZPlug = PlugDescriptor("weight3dZ")
	node : Brownian = None
	pass
class XPixelAnglePlug(Plug):
	node : Brownian = None
	pass
# endregion


# define node class
class Brownian(Texture3d):
	increment_ : IncrementPlug = PlugDescriptor("increment")
	lacunarity_ : LacunarityPlug = PlugDescriptor("lacunarity")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	octaves_ : OctavesPlug = PlugDescriptor("octaves")
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	refPointCamera_ : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	refPointObj_ : RefPointObjPlug = PlugDescriptor("refPointObj")
	weight3dX_ : Weight3dXPlug = PlugDescriptor("weight3dX")
	weight3dY_ : Weight3dYPlug = PlugDescriptor("weight3dY")
	weight3dZ_ : Weight3dZPlug = PlugDescriptor("weight3dZ")
	weight3d_ : Weight3dPlug = PlugDescriptor("weight3d")
	xPixelAngle_ : XPixelAnglePlug = PlugDescriptor("xPixelAngle")

	# node attributes

	typeName = "brownian"
	apiTypeInt = 508
	apiTypeStr = "kBrownian"
	typeIdInt = 1380336205
	MFnCls = om.MFnDependencyNode
	pass

