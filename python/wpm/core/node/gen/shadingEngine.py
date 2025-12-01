

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ObjectSet = retriever.getNodeCls("ObjectSet")
assert ObjectSet
if T.TYPE_CHECKING:
	from .. import ObjectSet

# add node doc



# region plug type defs
class BogusAmbientPlug(Plug):
	parent : BogusAttributePlug = PlugDescriptor("bogusAttribute")
	node : ShadingEngine = None
	pass
class BogusBlindDataPlug(Plug):
	parent : BogusAttributePlug = PlugDescriptor("bogusAttribute")
	node : ShadingEngine = None
	pass
class BogusDiffusePlug(Plug):
	parent : BogusAttributePlug = PlugDescriptor("bogusAttribute")
	node : ShadingEngine = None
	pass
class BogusDirectionXPlug(Plug):
	parent : BogusDirectionPlug = PlugDescriptor("bogusDirection")
	node : ShadingEngine = None
	pass
class BogusDirectionYPlug(Plug):
	parent : BogusDirectionPlug = PlugDescriptor("bogusDirection")
	node : ShadingEngine = None
	pass
class BogusDirectionZPlug(Plug):
	parent : BogusDirectionPlug = PlugDescriptor("bogusDirection")
	node : ShadingEngine = None
	pass
class BogusDirectionPlug(Plug):
	parent : BogusAttributePlug = PlugDescriptor("bogusAttribute")
	bogusDirectionX_ : BogusDirectionXPlug = PlugDescriptor("bogusDirectionX")
	blx_ : BogusDirectionXPlug = PlugDescriptor("bogusDirectionX")
	bogusDirectionY_ : BogusDirectionYPlug = PlugDescriptor("bogusDirectionY")
	bly_ : BogusDirectionYPlug = PlugDescriptor("bogusDirectionY")
	bogusDirectionZ_ : BogusDirectionZPlug = PlugDescriptor("bogusDirectionZ")
	blz_ : BogusDirectionZPlug = PlugDescriptor("bogusDirectionZ")
	node : ShadingEngine = None
	pass
class BogusIntensityBPlug(Plug):
	parent : BogusIntensityPlug = PlugDescriptor("bogusIntensity")
	node : ShadingEngine = None
	pass
class BogusIntensityGPlug(Plug):
	parent : BogusIntensityPlug = PlugDescriptor("bogusIntensity")
	node : ShadingEngine = None
	pass
class BogusIntensityRPlug(Plug):
	parent : BogusIntensityPlug = PlugDescriptor("bogusIntensity")
	node : ShadingEngine = None
	pass
class BogusIntensityPlug(Plug):
	parent : BogusAttributePlug = PlugDescriptor("bogusAttribute")
	bogusIntensityB_ : BogusIntensityBPlug = PlugDescriptor("bogusIntensityB")
	blb_ : BogusIntensityBPlug = PlugDescriptor("bogusIntensityB")
	bogusIntensityG_ : BogusIntensityGPlug = PlugDescriptor("bogusIntensityG")
	blg_ : BogusIntensityGPlug = PlugDescriptor("bogusIntensityG")
	bogusIntensityR_ : BogusIntensityRPlug = PlugDescriptor("bogusIntensityR")
	blr_ : BogusIntensityRPlug = PlugDescriptor("bogusIntensityR")
	node : ShadingEngine = None
	pass
class BogusPreShadowIntensityPlug(Plug):
	parent : BogusAttributePlug = PlugDescriptor("bogusAttribute")
	node : ShadingEngine = None
	pass
class BogusShadowFractionPlug(Plug):
	parent : BogusAttributePlug = PlugDescriptor("bogusAttribute")
	node : ShadingEngine = None
	pass
class BogusSpecularPlug(Plug):
	parent : BogusAttributePlug = PlugDescriptor("bogusAttribute")
	node : ShadingEngine = None
	pass
class BogusAttributePlug(Plug):
	bogusAmbient_ : BogusAmbientPlug = PlugDescriptor("bogusAmbient")
	bla_ : BogusAmbientPlug = PlugDescriptor("bogusAmbient")
	bogusBlindData_ : BogusBlindDataPlug = PlugDescriptor("bogusBlindData")
	bbld_ : BogusBlindDataPlug = PlugDescriptor("bogusBlindData")
	bogusDiffuse_ : BogusDiffusePlug = PlugDescriptor("bogusDiffuse")
	blf_ : BogusDiffusePlug = PlugDescriptor("bogusDiffuse")
	bogusDirection_ : BogusDirectionPlug = PlugDescriptor("bogusDirection")
	bld_ : BogusDirectionPlug = PlugDescriptor("bogusDirection")
	bogusIntensity_ : BogusIntensityPlug = PlugDescriptor("bogusIntensity")
	bli_ : BogusIntensityPlug = PlugDescriptor("bogusIntensity")
	bogusPreShadowIntensity_ : BogusPreShadowIntensityPlug = PlugDescriptor("bogusPreShadowIntensity")
	blps_ : BogusPreShadowIntensityPlug = PlugDescriptor("bogusPreShadowIntensity")
	bogusShadowFraction_ : BogusShadowFractionPlug = PlugDescriptor("bogusShadowFraction")
	blp_ : BogusShadowFractionPlug = PlugDescriptor("bogusShadowFraction")
	bogusSpecular_ : BogusSpecularPlug = PlugDescriptor("bogusSpecular")
	bls_ : BogusSpecularPlug = PlugDescriptor("bogusSpecular")
	node : ShadingEngine = None
	pass
class DefaultLightsPlug(Plug):
	node : ShadingEngine = None
	pass
class DShadowAmbientPlug(Plug):
	parent : DefaultShadowsPlug = PlugDescriptor("defaultShadows")
	node : ShadingEngine = None
	pass
class DShadowBlindDataPlug(Plug):
	parent : DefaultShadowsPlug = PlugDescriptor("defaultShadows")
	node : ShadingEngine = None
	pass
class DShadowDiffusePlug(Plug):
	parent : DefaultShadowsPlug = PlugDescriptor("defaultShadows")
	node : ShadingEngine = None
	pass
class DShadowDirectionXPlug(Plug):
	parent : DShadowDirectionPlug = PlugDescriptor("dShadowDirection")
	node : ShadingEngine = None
	pass
class DShadowDirectionYPlug(Plug):
	parent : DShadowDirectionPlug = PlugDescriptor("dShadowDirection")
	node : ShadingEngine = None
	pass
class DShadowDirectionZPlug(Plug):
	parent : DShadowDirectionPlug = PlugDescriptor("dShadowDirection")
	node : ShadingEngine = None
	pass
class DShadowDirectionPlug(Plug):
	parent : DefaultShadowsPlug = PlugDescriptor("defaultShadows")
	dShadowDirectionX_ : DShadowDirectionXPlug = PlugDescriptor("dShadowDirectionX")
	dsx_ : DShadowDirectionXPlug = PlugDescriptor("dShadowDirectionX")
	dShadowDirectionY_ : DShadowDirectionYPlug = PlugDescriptor("dShadowDirectionY")
	dsy_ : DShadowDirectionYPlug = PlugDescriptor("dShadowDirectionY")
	dShadowDirectionZ_ : DShadowDirectionZPlug = PlugDescriptor("dShadowDirectionZ")
	dsz_ : DShadowDirectionZPlug = PlugDescriptor("dShadowDirectionZ")
	node : ShadingEngine = None
	pass
class DShadowIntensityBPlug(Plug):
	parent : DShadowIntensityPlug = PlugDescriptor("dShadowIntensity")
	node : ShadingEngine = None
	pass
class DShadowIntensityGPlug(Plug):
	parent : DShadowIntensityPlug = PlugDescriptor("dShadowIntensity")
	node : ShadingEngine = None
	pass
class DShadowIntensityRPlug(Plug):
	parent : DShadowIntensityPlug = PlugDescriptor("dShadowIntensity")
	node : ShadingEngine = None
	pass
class DShadowIntensityPlug(Plug):
	parent : DefaultShadowsPlug = PlugDescriptor("defaultShadows")
	dShadowIntensityB_ : DShadowIntensityBPlug = PlugDescriptor("dShadowIntensityB")
	dsb_ : DShadowIntensityBPlug = PlugDescriptor("dShadowIntensityB")
	dShadowIntensityG_ : DShadowIntensityGPlug = PlugDescriptor("dShadowIntensityG")
	dsg_ : DShadowIntensityGPlug = PlugDescriptor("dShadowIntensityG")
	dShadowIntensityR_ : DShadowIntensityRPlug = PlugDescriptor("dShadowIntensityR")
	dsr_ : DShadowIntensityRPlug = PlugDescriptor("dShadowIntensityR")
	node : ShadingEngine = None
	pass
class DShadowPreShadowIntensityPlug(Plug):
	parent : DefaultShadowsPlug = PlugDescriptor("defaultShadows")
	node : ShadingEngine = None
	pass
class DShadowShadowFractionPlug(Plug):
	parent : DefaultShadowsPlug = PlugDescriptor("defaultShadows")
	node : ShadingEngine = None
	pass
class DShadowSpecularPlug(Plug):
	parent : DefaultShadowsPlug = PlugDescriptor("defaultShadows")
	node : ShadingEngine = None
	pass
class DefaultShadowsPlug(Plug):
	dShadowAmbient_ : DShadowAmbientPlug = PlugDescriptor("dShadowAmbient")
	dsa_ : DShadowAmbientPlug = PlugDescriptor("dShadowAmbient")
	dShadowBlindData_ : DShadowBlindDataPlug = PlugDescriptor("dShadowBlindData")
	dbld_ : DShadowBlindDataPlug = PlugDescriptor("dShadowBlindData")
	dShadowDiffuse_ : DShadowDiffusePlug = PlugDescriptor("dShadowDiffuse")
	dsf_ : DShadowDiffusePlug = PlugDescriptor("dShadowDiffuse")
	dShadowDirection_ : DShadowDirectionPlug = PlugDescriptor("dShadowDirection")
	dsd_ : DShadowDirectionPlug = PlugDescriptor("dShadowDirection")
	dShadowIntensity_ : DShadowIntensityPlug = PlugDescriptor("dShadowIntensity")
	dsi_ : DShadowIntensityPlug = PlugDescriptor("dShadowIntensity")
	dShadowPreShadowIntensity_ : DShadowPreShadowIntensityPlug = PlugDescriptor("dShadowPreShadowIntensity")
	dsps_ : DShadowPreShadowIntensityPlug = PlugDescriptor("dShadowPreShadowIntensity")
	dShadowShadowFraction_ : DShadowShadowFractionPlug = PlugDescriptor("dShadowShadowFraction")
	dssf_ : DShadowShadowFractionPlug = PlugDescriptor("dShadowShadowFraction")
	dShadowSpecular_ : DShadowSpecularPlug = PlugDescriptor("dShadowSpecular")
	dss_ : DShadowSpecularPlug = PlugDescriptor("dShadowSpecular")
	node : ShadingEngine = None
	pass
class DisplacementShaderPlug(Plug):
	node : ShadingEngine = None
	pass
class IgnoredLightsPlug(Plug):
	node : ShadingEngine = None
	pass
class XShadowAmbientPlug(Plug):
	parent : IgnoredShadowsPlug = PlugDescriptor("ignoredShadows")
	node : ShadingEngine = None
	pass
class XShadowBlindDataPlug(Plug):
	parent : IgnoredShadowsPlug = PlugDescriptor("ignoredShadows")
	node : ShadingEngine = None
	pass
class XShadowDiffusePlug(Plug):
	parent : IgnoredShadowsPlug = PlugDescriptor("ignoredShadows")
	node : ShadingEngine = None
	pass
class XShadowDirectionXPlug(Plug):
	parent : XShadowDirectionPlug = PlugDescriptor("xShadowDirection")
	node : ShadingEngine = None
	pass
class XShadowDirectionYPlug(Plug):
	parent : XShadowDirectionPlug = PlugDescriptor("xShadowDirection")
	node : ShadingEngine = None
	pass
class XShadowDirectionZPlug(Plug):
	parent : XShadowDirectionPlug = PlugDescriptor("xShadowDirection")
	node : ShadingEngine = None
	pass
class XShadowDirectionPlug(Plug):
	parent : IgnoredShadowsPlug = PlugDescriptor("ignoredShadows")
	xShadowDirectionX_ : XShadowDirectionXPlug = PlugDescriptor("xShadowDirectionX")
	xsx_ : XShadowDirectionXPlug = PlugDescriptor("xShadowDirectionX")
	xShadowDirectionY_ : XShadowDirectionYPlug = PlugDescriptor("xShadowDirectionY")
	xsy_ : XShadowDirectionYPlug = PlugDescriptor("xShadowDirectionY")
	xShadowDirectionZ_ : XShadowDirectionZPlug = PlugDescriptor("xShadowDirectionZ")
	xsz_ : XShadowDirectionZPlug = PlugDescriptor("xShadowDirectionZ")
	node : ShadingEngine = None
	pass
class XShadowIntensityBPlug(Plug):
	parent : XShadowIntensityPlug = PlugDescriptor("xShadowIntensity")
	node : ShadingEngine = None
	pass
class XShadowIntensityGPlug(Plug):
	parent : XShadowIntensityPlug = PlugDescriptor("xShadowIntensity")
	node : ShadingEngine = None
	pass
class XShadowIntensityRPlug(Plug):
	parent : XShadowIntensityPlug = PlugDescriptor("xShadowIntensity")
	node : ShadingEngine = None
	pass
class XShadowIntensityPlug(Plug):
	parent : IgnoredShadowsPlug = PlugDescriptor("ignoredShadows")
	xShadowIntensityB_ : XShadowIntensityBPlug = PlugDescriptor("xShadowIntensityB")
	xsb_ : XShadowIntensityBPlug = PlugDescriptor("xShadowIntensityB")
	xShadowIntensityG_ : XShadowIntensityGPlug = PlugDescriptor("xShadowIntensityG")
	xsg_ : XShadowIntensityGPlug = PlugDescriptor("xShadowIntensityG")
	xShadowIntensityR_ : XShadowIntensityRPlug = PlugDescriptor("xShadowIntensityR")
	xsr_ : XShadowIntensityRPlug = PlugDescriptor("xShadowIntensityR")
	node : ShadingEngine = None
	pass
class XShadowPreShadowIntensityPlug(Plug):
	parent : IgnoredShadowsPlug = PlugDescriptor("ignoredShadows")
	node : ShadingEngine = None
	pass
class XShadowShadowFractionPlug(Plug):
	parent : IgnoredShadowsPlug = PlugDescriptor("ignoredShadows")
	node : ShadingEngine = None
	pass
class XShadowSpecularPlug(Plug):
	parent : IgnoredShadowsPlug = PlugDescriptor("ignoredShadows")
	node : ShadingEngine = None
	pass
class IgnoredShadowsPlug(Plug):
	xShadowAmbient_ : XShadowAmbientPlug = PlugDescriptor("xShadowAmbient")
	xsa_ : XShadowAmbientPlug = PlugDescriptor("xShadowAmbient")
	xShadowBlindData_ : XShadowBlindDataPlug = PlugDescriptor("xShadowBlindData")
	xbld_ : XShadowBlindDataPlug = PlugDescriptor("xShadowBlindData")
	xShadowDiffuse_ : XShadowDiffusePlug = PlugDescriptor("xShadowDiffuse")
	xsf_ : XShadowDiffusePlug = PlugDescriptor("xShadowDiffuse")
	xShadowDirection_ : XShadowDirectionPlug = PlugDescriptor("xShadowDirection")
	xsd_ : XShadowDirectionPlug = PlugDescriptor("xShadowDirection")
	xShadowIntensity_ : XShadowIntensityPlug = PlugDescriptor("xShadowIntensity")
	xsi_ : XShadowIntensityPlug = PlugDescriptor("xShadowIntensity")
	xShadowPreShadowIntensity_ : XShadowPreShadowIntensityPlug = PlugDescriptor("xShadowPreShadowIntensity")
	xsps_ : XShadowPreShadowIntensityPlug = PlugDescriptor("xShadowPreShadowIntensity")
	xShadowShadowFraction_ : XShadowShadowFractionPlug = PlugDescriptor("xShadowShadowFraction")
	xssf_ : XShadowShadowFractionPlug = PlugDescriptor("xShadowShadowFraction")
	xShadowSpecular_ : XShadowSpecularPlug = PlugDescriptor("xShadowSpecular")
	xss_ : XShadowSpecularPlug = PlugDescriptor("xShadowSpecular")
	node : ShadingEngine = None
	pass
class ImageShaderPlug(Plug):
	node : ShadingEngine = None
	pass
class LinkedLightsPlug(Plug):
	node : ShadingEngine = None
	pass
class LShadowAmbientPlug(Plug):
	parent : LinkedShadowsPlug = PlugDescriptor("linkedShadows")
	node : ShadingEngine = None
	pass
class LShadowBlindDataPlug(Plug):
	parent : LinkedShadowsPlug = PlugDescriptor("linkedShadows")
	node : ShadingEngine = None
	pass
class LShadowDiffusePlug(Plug):
	parent : LinkedShadowsPlug = PlugDescriptor("linkedShadows")
	node : ShadingEngine = None
	pass
class LShadowDirectionXPlug(Plug):
	parent : LShadowDirectionPlug = PlugDescriptor("lShadowDirection")
	node : ShadingEngine = None
	pass
class LShadowDirectionYPlug(Plug):
	parent : LShadowDirectionPlug = PlugDescriptor("lShadowDirection")
	node : ShadingEngine = None
	pass
class LShadowDirectionZPlug(Plug):
	parent : LShadowDirectionPlug = PlugDescriptor("lShadowDirection")
	node : ShadingEngine = None
	pass
class LShadowDirectionPlug(Plug):
	parent : LinkedShadowsPlug = PlugDescriptor("linkedShadows")
	lShadowDirectionX_ : LShadowDirectionXPlug = PlugDescriptor("lShadowDirectionX")
	lsx_ : LShadowDirectionXPlug = PlugDescriptor("lShadowDirectionX")
	lShadowDirectionY_ : LShadowDirectionYPlug = PlugDescriptor("lShadowDirectionY")
	lsy_ : LShadowDirectionYPlug = PlugDescriptor("lShadowDirectionY")
	lShadowDirectionZ_ : LShadowDirectionZPlug = PlugDescriptor("lShadowDirectionZ")
	lsz_ : LShadowDirectionZPlug = PlugDescriptor("lShadowDirectionZ")
	node : ShadingEngine = None
	pass
class LShadowIntensityBPlug(Plug):
	parent : LShadowIntensityPlug = PlugDescriptor("lShadowIntensity")
	node : ShadingEngine = None
	pass
class LShadowIntensityGPlug(Plug):
	parent : LShadowIntensityPlug = PlugDescriptor("lShadowIntensity")
	node : ShadingEngine = None
	pass
class LShadowIntensityRPlug(Plug):
	parent : LShadowIntensityPlug = PlugDescriptor("lShadowIntensity")
	node : ShadingEngine = None
	pass
class LShadowIntensityPlug(Plug):
	parent : LinkedShadowsPlug = PlugDescriptor("linkedShadows")
	lShadowIntensityB_ : LShadowIntensityBPlug = PlugDescriptor("lShadowIntensityB")
	lsb_ : LShadowIntensityBPlug = PlugDescriptor("lShadowIntensityB")
	lShadowIntensityG_ : LShadowIntensityGPlug = PlugDescriptor("lShadowIntensityG")
	lsg_ : LShadowIntensityGPlug = PlugDescriptor("lShadowIntensityG")
	lShadowIntensityR_ : LShadowIntensityRPlug = PlugDescriptor("lShadowIntensityR")
	lsr_ : LShadowIntensityRPlug = PlugDescriptor("lShadowIntensityR")
	node : ShadingEngine = None
	pass
class LShadowPreShadowIntensityPlug(Plug):
	parent : LinkedShadowsPlug = PlugDescriptor("linkedShadows")
	node : ShadingEngine = None
	pass
class LShadowShadowFractionPlug(Plug):
	parent : LinkedShadowsPlug = PlugDescriptor("linkedShadows")
	node : ShadingEngine = None
	pass
class LShadowSpecularPlug(Plug):
	parent : LinkedShadowsPlug = PlugDescriptor("linkedShadows")
	node : ShadingEngine = None
	pass
class LinkedShadowsPlug(Plug):
	lShadowAmbient_ : LShadowAmbientPlug = PlugDescriptor("lShadowAmbient")
	lsa_ : LShadowAmbientPlug = PlugDescriptor("lShadowAmbient")
	lShadowBlindData_ : LShadowBlindDataPlug = PlugDescriptor("lShadowBlindData")
	lbld_ : LShadowBlindDataPlug = PlugDescriptor("lShadowBlindData")
	lShadowDiffuse_ : LShadowDiffusePlug = PlugDescriptor("lShadowDiffuse")
	lsf_ : LShadowDiffusePlug = PlugDescriptor("lShadowDiffuse")
	lShadowDirection_ : LShadowDirectionPlug = PlugDescriptor("lShadowDirection")
	lsd_ : LShadowDirectionPlug = PlugDescriptor("lShadowDirection")
	lShadowIntensity_ : LShadowIntensityPlug = PlugDescriptor("lShadowIntensity")
	lsi_ : LShadowIntensityPlug = PlugDescriptor("lShadowIntensity")
	lShadowPreShadowIntensity_ : LShadowPreShadowIntensityPlug = PlugDescriptor("lShadowPreShadowIntensity")
	lsps_ : LShadowPreShadowIntensityPlug = PlugDescriptor("lShadowPreShadowIntensity")
	lShadowShadowFraction_ : LShadowShadowFractionPlug = PlugDescriptor("lShadowShadowFraction")
	lssf_ : LShadowShadowFractionPlug = PlugDescriptor("lShadowShadowFraction")
	lShadowSpecular_ : LShadowSpecularPlug = PlugDescriptor("lShadowSpecular")
	lss_ : LShadowSpecularPlug = PlugDescriptor("lShadowSpecular")
	node : ShadingEngine = None
	pass
class SurfaceShaderPlug(Plug):
	node : ShadingEngine = None
	pass
class UnsolicitedPlug(Plug):
	node : ShadingEngine = None
	pass
class VolumeShaderPlug(Plug):
	node : ShadingEngine = None
	pass
# endregion


# define node class
class ShadingEngine(ObjectSet):
	bogusAmbient_ : BogusAmbientPlug = PlugDescriptor("bogusAmbient")
	bogusBlindData_ : BogusBlindDataPlug = PlugDescriptor("bogusBlindData")
	bogusDiffuse_ : BogusDiffusePlug = PlugDescriptor("bogusDiffuse")
	bogusDirectionX_ : BogusDirectionXPlug = PlugDescriptor("bogusDirectionX")
	bogusDirectionY_ : BogusDirectionYPlug = PlugDescriptor("bogusDirectionY")
	bogusDirectionZ_ : BogusDirectionZPlug = PlugDescriptor("bogusDirectionZ")
	bogusDirection_ : BogusDirectionPlug = PlugDescriptor("bogusDirection")
	bogusIntensityB_ : BogusIntensityBPlug = PlugDescriptor("bogusIntensityB")
	bogusIntensityG_ : BogusIntensityGPlug = PlugDescriptor("bogusIntensityG")
	bogusIntensityR_ : BogusIntensityRPlug = PlugDescriptor("bogusIntensityR")
	bogusIntensity_ : BogusIntensityPlug = PlugDescriptor("bogusIntensity")
	bogusPreShadowIntensity_ : BogusPreShadowIntensityPlug = PlugDescriptor("bogusPreShadowIntensity")
	bogusShadowFraction_ : BogusShadowFractionPlug = PlugDescriptor("bogusShadowFraction")
	bogusSpecular_ : BogusSpecularPlug = PlugDescriptor("bogusSpecular")
	bogusAttribute_ : BogusAttributePlug = PlugDescriptor("bogusAttribute")
	defaultLights_ : DefaultLightsPlug = PlugDescriptor("defaultLights")
	dShadowAmbient_ : DShadowAmbientPlug = PlugDescriptor("dShadowAmbient")
	dShadowBlindData_ : DShadowBlindDataPlug = PlugDescriptor("dShadowBlindData")
	dShadowDiffuse_ : DShadowDiffusePlug = PlugDescriptor("dShadowDiffuse")
	dShadowDirectionX_ : DShadowDirectionXPlug = PlugDescriptor("dShadowDirectionX")
	dShadowDirectionY_ : DShadowDirectionYPlug = PlugDescriptor("dShadowDirectionY")
	dShadowDirectionZ_ : DShadowDirectionZPlug = PlugDescriptor("dShadowDirectionZ")
	dShadowDirection_ : DShadowDirectionPlug = PlugDescriptor("dShadowDirection")
	dShadowIntensityB_ : DShadowIntensityBPlug = PlugDescriptor("dShadowIntensityB")
	dShadowIntensityG_ : DShadowIntensityGPlug = PlugDescriptor("dShadowIntensityG")
	dShadowIntensityR_ : DShadowIntensityRPlug = PlugDescriptor("dShadowIntensityR")
	dShadowIntensity_ : DShadowIntensityPlug = PlugDescriptor("dShadowIntensity")
	dShadowPreShadowIntensity_ : DShadowPreShadowIntensityPlug = PlugDescriptor("dShadowPreShadowIntensity")
	dShadowShadowFraction_ : DShadowShadowFractionPlug = PlugDescriptor("dShadowShadowFraction")
	dShadowSpecular_ : DShadowSpecularPlug = PlugDescriptor("dShadowSpecular")
	defaultShadows_ : DefaultShadowsPlug = PlugDescriptor("defaultShadows")
	displacementShader_ : DisplacementShaderPlug = PlugDescriptor("displacementShader")
	ignoredLights_ : IgnoredLightsPlug = PlugDescriptor("ignoredLights")
	xShadowAmbient_ : XShadowAmbientPlug = PlugDescriptor("xShadowAmbient")
	xShadowBlindData_ : XShadowBlindDataPlug = PlugDescriptor("xShadowBlindData")
	xShadowDiffuse_ : XShadowDiffusePlug = PlugDescriptor("xShadowDiffuse")
	xShadowDirectionX_ : XShadowDirectionXPlug = PlugDescriptor("xShadowDirectionX")
	xShadowDirectionY_ : XShadowDirectionYPlug = PlugDescriptor("xShadowDirectionY")
	xShadowDirectionZ_ : XShadowDirectionZPlug = PlugDescriptor("xShadowDirectionZ")
	xShadowDirection_ : XShadowDirectionPlug = PlugDescriptor("xShadowDirection")
	xShadowIntensityB_ : XShadowIntensityBPlug = PlugDescriptor("xShadowIntensityB")
	xShadowIntensityG_ : XShadowIntensityGPlug = PlugDescriptor("xShadowIntensityG")
	xShadowIntensityR_ : XShadowIntensityRPlug = PlugDescriptor("xShadowIntensityR")
	xShadowIntensity_ : XShadowIntensityPlug = PlugDescriptor("xShadowIntensity")
	xShadowPreShadowIntensity_ : XShadowPreShadowIntensityPlug = PlugDescriptor("xShadowPreShadowIntensity")
	xShadowShadowFraction_ : XShadowShadowFractionPlug = PlugDescriptor("xShadowShadowFraction")
	xShadowSpecular_ : XShadowSpecularPlug = PlugDescriptor("xShadowSpecular")
	ignoredShadows_ : IgnoredShadowsPlug = PlugDescriptor("ignoredShadows")
	imageShader_ : ImageShaderPlug = PlugDescriptor("imageShader")
	linkedLights_ : LinkedLightsPlug = PlugDescriptor("linkedLights")
	lShadowAmbient_ : LShadowAmbientPlug = PlugDescriptor("lShadowAmbient")
	lShadowBlindData_ : LShadowBlindDataPlug = PlugDescriptor("lShadowBlindData")
	lShadowDiffuse_ : LShadowDiffusePlug = PlugDescriptor("lShadowDiffuse")
	lShadowDirectionX_ : LShadowDirectionXPlug = PlugDescriptor("lShadowDirectionX")
	lShadowDirectionY_ : LShadowDirectionYPlug = PlugDescriptor("lShadowDirectionY")
	lShadowDirectionZ_ : LShadowDirectionZPlug = PlugDescriptor("lShadowDirectionZ")
	lShadowDirection_ : LShadowDirectionPlug = PlugDescriptor("lShadowDirection")
	lShadowIntensityB_ : LShadowIntensityBPlug = PlugDescriptor("lShadowIntensityB")
	lShadowIntensityG_ : LShadowIntensityGPlug = PlugDescriptor("lShadowIntensityG")
	lShadowIntensityR_ : LShadowIntensityRPlug = PlugDescriptor("lShadowIntensityR")
	lShadowIntensity_ : LShadowIntensityPlug = PlugDescriptor("lShadowIntensity")
	lShadowPreShadowIntensity_ : LShadowPreShadowIntensityPlug = PlugDescriptor("lShadowPreShadowIntensity")
	lShadowShadowFraction_ : LShadowShadowFractionPlug = PlugDescriptor("lShadowShadowFraction")
	lShadowSpecular_ : LShadowSpecularPlug = PlugDescriptor("lShadowSpecular")
	linkedShadows_ : LinkedShadowsPlug = PlugDescriptor("linkedShadows")
	surfaceShader_ : SurfaceShaderPlug = PlugDescriptor("surfaceShader")
	unsolicited_ : UnsolicitedPlug = PlugDescriptor("unsolicited")
	volumeShader_ : VolumeShaderPlug = PlugDescriptor("volumeShader")

	# node attributes

	typeName = "shadingEngine"
	apiTypeInt = 320
	apiTypeStr = "kShadingEngine"
	typeIdInt = 1397244228
	MFnCls = om.MFnSet
	nodeLeafClassAttrs = ["bogusAmbient", "bogusBlindData", "bogusDiffuse", "bogusDirectionX", "bogusDirectionY", "bogusDirectionZ", "bogusDirection", "bogusIntensityB", "bogusIntensityG", "bogusIntensityR", "bogusIntensity", "bogusPreShadowIntensity", "bogusShadowFraction", "bogusSpecular", "bogusAttribute", "defaultLights", "dShadowAmbient", "dShadowBlindData", "dShadowDiffuse", "dShadowDirectionX", "dShadowDirectionY", "dShadowDirectionZ", "dShadowDirection", "dShadowIntensityB", "dShadowIntensityG", "dShadowIntensityR", "dShadowIntensity", "dShadowPreShadowIntensity", "dShadowShadowFraction", "dShadowSpecular", "defaultShadows", "displacementShader", "ignoredLights", "xShadowAmbient", "xShadowBlindData", "xShadowDiffuse", "xShadowDirectionX", "xShadowDirectionY", "xShadowDirectionZ", "xShadowDirection", "xShadowIntensityB", "xShadowIntensityG", "xShadowIntensityR", "xShadowIntensity", "xShadowPreShadowIntensity", "xShadowShadowFraction", "xShadowSpecular", "ignoredShadows", "imageShader", "linkedLights", "lShadowAmbient", "lShadowBlindData", "lShadowDiffuse", "lShadowDirectionX", "lShadowDirectionY", "lShadowDirectionZ", "lShadowDirection", "lShadowIntensityB", "lShadowIntensityG", "lShadowIntensityR", "lShadowIntensity", "lShadowPreShadowIntensity", "lShadowShadowFraction", "lShadowSpecular", "linkedShadows", "surfaceShader", "unsolicited", "volumeShader"]
	nodeLeafPlugs = ["bogusAttribute", "defaultLights", "defaultShadows", "displacementShader", "ignoredLights", "ignoredShadows", "imageShader", "linkedLights", "linkedShadows", "surfaceShader", "unsolicited", "volumeShader"]
	pass

