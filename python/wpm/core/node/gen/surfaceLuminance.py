

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
class BinMembershipPlug(Plug):
	node : SurfaceLuminance = None
	pass
class LightAmbientPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : SurfaceLuminance = None
	pass
class LightBlindDataPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : SurfaceLuminance = None
	pass
class LightDiffusePlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : SurfaceLuminance = None
	pass
class LightDirectionXPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : SurfaceLuminance = None
	pass
class LightDirectionYPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : SurfaceLuminance = None
	pass
class LightDirectionZPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : SurfaceLuminance = None
	pass
class LightDirectionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	ldx_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	ldy_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	ldz_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	node : SurfaceLuminance = None
	pass
class LightIntensityBPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : SurfaceLuminance = None
	pass
class LightIntensityGPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : SurfaceLuminance = None
	pass
class LightIntensityRPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : SurfaceLuminance = None
	pass
class LightIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightIntensityB_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lib_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lightIntensityG_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lig_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lightIntensityR_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	lir_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	node : SurfaceLuminance = None
	pass
class LightShadowFractionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : SurfaceLuminance = None
	pass
class LightSpecularPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : SurfaceLuminance = None
	pass
class PreShadowIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : SurfaceLuminance = None
	pass
class LightDataArrayPlug(Plug):
	lightAmbient_ : LightAmbientPlug = PlugDescriptor("lightAmbient")
	la_ : LightAmbientPlug = PlugDescriptor("lightAmbient")
	lightBlindData_ : LightBlindDataPlug = PlugDescriptor("lightBlindData")
	lbd_ : LightBlindDataPlug = PlugDescriptor("lightBlindData")
	lightDiffuse_ : LightDiffusePlug = PlugDescriptor("lightDiffuse")
	ldf_ : LightDiffusePlug = PlugDescriptor("lightDiffuse")
	lightDirection_ : LightDirectionPlug = PlugDescriptor("lightDirection")
	ld_ : LightDirectionPlug = PlugDescriptor("lightDirection")
	lightIntensity_ : LightIntensityPlug = PlugDescriptor("lightIntensity")
	li_ : LightIntensityPlug = PlugDescriptor("lightIntensity")
	lightShadowFraction_ : LightShadowFractionPlug = PlugDescriptor("lightShadowFraction")
	lsf_ : LightShadowFractionPlug = PlugDescriptor("lightShadowFraction")
	lightSpecular_ : LightSpecularPlug = PlugDescriptor("lightSpecular")
	ls_ : LightSpecularPlug = PlugDescriptor("lightSpecular")
	preShadowIntensity_ : PreShadowIntensityPlug = PlugDescriptor("preShadowIntensity")
	psi_ : PreShadowIntensityPlug = PlugDescriptor("preShadowIntensity")
	node : SurfaceLuminance = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : SurfaceLuminance = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : SurfaceLuminance = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : SurfaceLuminance = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : SurfaceLuminance = None
	pass
class OutValuePlug(Plug):
	node : SurfaceLuminance = None
	pass
# endregion


# define node class
class SurfaceLuminance(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	lightAmbient_ : LightAmbientPlug = PlugDescriptor("lightAmbient")
	lightBlindData_ : LightBlindDataPlug = PlugDescriptor("lightBlindData")
	lightDiffuse_ : LightDiffusePlug = PlugDescriptor("lightDiffuse")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	lightDirection_ : LightDirectionPlug = PlugDescriptor("lightDirection")
	lightIntensityB_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lightIntensityG_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lightIntensityR_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	lightIntensity_ : LightIntensityPlug = PlugDescriptor("lightIntensity")
	lightShadowFraction_ : LightShadowFractionPlug = PlugDescriptor("lightShadowFraction")
	lightSpecular_ : LightSpecularPlug = PlugDescriptor("lightSpecular")
	preShadowIntensity_ : PreShadowIntensityPlug = PlugDescriptor("preShadowIntensity")
	lightDataArray_ : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	outValue_ : OutValuePlug = PlugDescriptor("outValue")

	# node attributes

	typeName = "surfaceLuminance"
	apiTypeInt = 487
	apiTypeStr = "kSurfaceLuminance"
	typeIdInt = 1381190741
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "lightAmbient", "lightBlindData", "lightDiffuse", "lightDirectionX", "lightDirectionY", "lightDirectionZ", "lightDirection", "lightIntensityB", "lightIntensityG", "lightIntensityR", "lightIntensity", "lightShadowFraction", "lightSpecular", "preShadowIntensity", "lightDataArray", "normalCameraX", "normalCameraY", "normalCameraZ", "normalCamera", "outValue"]
	nodeLeafPlugs = ["binMembership", "lightDataArray", "normalCamera", "outValue"]
	pass

