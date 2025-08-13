

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
	node : DefaultLightList = None
	pass
class LightAmbientOutPlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	node : DefaultLightList = None
	pass
class LightBlindDataOutPlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	node : DefaultLightList = None
	pass
class LightDiffuseOutPlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	node : DefaultLightList = None
	pass
class LightDirectionOutXPlug(Plug):
	parent : LightDirectionOutPlug = PlugDescriptor("lightDirectionOut")
	node : DefaultLightList = None
	pass
class LightDirectionOutYPlug(Plug):
	parent : LightDirectionOutPlug = PlugDescriptor("lightDirectionOut")
	node : DefaultLightList = None
	pass
class LightDirectionOutZPlug(Plug):
	parent : LightDirectionOutPlug = PlugDescriptor("lightDirectionOut")
	node : DefaultLightList = None
	pass
class LightDirectionOutPlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	lightDirectionOutX_ : LightDirectionOutXPlug = PlugDescriptor("lightDirectionOutX")
	lqx_ : LightDirectionOutXPlug = PlugDescriptor("lightDirectionOutX")
	lightDirectionOutY_ : LightDirectionOutYPlug = PlugDescriptor("lightDirectionOutY")
	lqy_ : LightDirectionOutYPlug = PlugDescriptor("lightDirectionOutY")
	lightDirectionOutZ_ : LightDirectionOutZPlug = PlugDescriptor("lightDirectionOutZ")
	lqz_ : LightDirectionOutZPlug = PlugDescriptor("lightDirectionOutZ")
	node : DefaultLightList = None
	pass
class LightIntensityOutBPlug(Plug):
	parent : LightIntensityOutPlug = PlugDescriptor("lightIntensityOut")
	node : DefaultLightList = None
	pass
class LightIntensityOutGPlug(Plug):
	parent : LightIntensityOutPlug = PlugDescriptor("lightIntensityOut")
	node : DefaultLightList = None
	pass
class LightIntensityOutRPlug(Plug):
	parent : LightIntensityOutPlug = PlugDescriptor("lightIntensityOut")
	node : DefaultLightList = None
	pass
class LightIntensityOutPlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	lightIntensityOutB_ : LightIntensityOutBPlug = PlugDescriptor("lightIntensityOutB")
	lwb_ : LightIntensityOutBPlug = PlugDescriptor("lightIntensityOutB")
	lightIntensityOutG_ : LightIntensityOutGPlug = PlugDescriptor("lightIntensityOutG")
	lwg_ : LightIntensityOutGPlug = PlugDescriptor("lightIntensityOutG")
	lightIntensityOutR_ : LightIntensityOutRPlug = PlugDescriptor("lightIntensityOutR")
	lwr_ : LightIntensityOutRPlug = PlugDescriptor("lightIntensityOutR")
	node : DefaultLightList = None
	pass
class LightShadowFractionOutPlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	node : DefaultLightList = None
	pass
class LightSpecularOutPlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	node : DefaultLightList = None
	pass
class PreShadowIntensityOutPlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	node : DefaultLightList = None
	pass
class LightDataPlug(Plug):
	lightAmbientOut_ : LightAmbientOutPlug = PlugDescriptor("lightAmbientOut")
	lya_ : LightAmbientOutPlug = PlugDescriptor("lightAmbientOut")
	lightBlindDataOut_ : LightBlindDataOutPlug = PlugDescriptor("lightBlindDataOut")
	lbdo_ : LightBlindDataOutPlug = PlugDescriptor("lightBlindDataOut")
	lightDiffuseOut_ : LightDiffuseOutPlug = PlugDescriptor("lightDiffuseOut")
	lyf_ : LightDiffuseOutPlug = PlugDescriptor("lightDiffuseOut")
	lightDirectionOut_ : LightDirectionOutPlug = PlugDescriptor("lightDirectionOut")
	ldo_ : LightDirectionOutPlug = PlugDescriptor("lightDirectionOut")
	lightIntensityOut_ : LightIntensityOutPlug = PlugDescriptor("lightIntensityOut")
	lw_ : LightIntensityOutPlug = PlugDescriptor("lightIntensityOut")
	lightShadowFractionOut_ : LightShadowFractionOutPlug = PlugDescriptor("lightShadowFractionOut")
	sfo_ : LightShadowFractionOutPlug = PlugDescriptor("lightShadowFractionOut")
	lightSpecularOut_ : LightSpecularOutPlug = PlugDescriptor("lightSpecularOut")
	lys_ : LightSpecularOutPlug = PlugDescriptor("lightSpecularOut")
	preShadowIntensityOut_ : PreShadowIntensityOutPlug = PlugDescriptor("preShadowIntensityOut")
	psio_ : PreShadowIntensityOutPlug = PlugDescriptor("preShadowIntensityOut")
	node : DefaultLightList = None
	pass
class LightAmbientPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : DefaultLightList = None
	pass
class LightBlindDataPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : DefaultLightList = None
	pass
class LightDiffusePlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : DefaultLightList = None
	pass
class LightDirectionXPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : DefaultLightList = None
	pass
class LightDirectionYPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : DefaultLightList = None
	pass
class LightDirectionZPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : DefaultLightList = None
	pass
class LightDirectionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	ldx_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	ldy_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	ldz_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	node : DefaultLightList = None
	pass
class LightIntensityBPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : DefaultLightList = None
	pass
class LightIntensityGPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : DefaultLightList = None
	pass
class LightIntensityRPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : DefaultLightList = None
	pass
class LightIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightIntensityB_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lib_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lightIntensityG_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lig_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lightIntensityR_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	lir_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	node : DefaultLightList = None
	pass
class LightShadowFractionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : DefaultLightList = None
	pass
class LightSpecularPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : DefaultLightList = None
	pass
class PreShadowIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : DefaultLightList = None
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
	node : DefaultLightList = None
	pass
# endregion


# define node class
class DefaultLightList(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	lightAmbientOut_ : LightAmbientOutPlug = PlugDescriptor("lightAmbientOut")
	lightBlindDataOut_ : LightBlindDataOutPlug = PlugDescriptor("lightBlindDataOut")
	lightDiffuseOut_ : LightDiffuseOutPlug = PlugDescriptor("lightDiffuseOut")
	lightDirectionOutX_ : LightDirectionOutXPlug = PlugDescriptor("lightDirectionOutX")
	lightDirectionOutY_ : LightDirectionOutYPlug = PlugDescriptor("lightDirectionOutY")
	lightDirectionOutZ_ : LightDirectionOutZPlug = PlugDescriptor("lightDirectionOutZ")
	lightDirectionOut_ : LightDirectionOutPlug = PlugDescriptor("lightDirectionOut")
	lightIntensityOutB_ : LightIntensityOutBPlug = PlugDescriptor("lightIntensityOutB")
	lightIntensityOutG_ : LightIntensityOutGPlug = PlugDescriptor("lightIntensityOutG")
	lightIntensityOutR_ : LightIntensityOutRPlug = PlugDescriptor("lightIntensityOutR")
	lightIntensityOut_ : LightIntensityOutPlug = PlugDescriptor("lightIntensityOut")
	lightShadowFractionOut_ : LightShadowFractionOutPlug = PlugDescriptor("lightShadowFractionOut")
	lightSpecularOut_ : LightSpecularOutPlug = PlugDescriptor("lightSpecularOut")
	preShadowIntensityOut_ : PreShadowIntensityOutPlug = PlugDescriptor("preShadowIntensityOut")
	lightData_ : LightDataPlug = PlugDescriptor("lightData")
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

	# node attributes

	typeName = "defaultLightList"
	apiTypeInt = 317
	apiTypeStr = "kDefaultLightList"
	typeIdInt = 1145390668
	MFnCls = om.MFnDependencyNode
	pass

