

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Shape = retriever.getNodeCls("Shape")
assert Shape
if T.TYPE_CHECKING:
	from .. import Shape

# add node doc



# region plug type defs
class AttractRadiusOffsetPlug(Plug):
	node : LineModifier = None
	pass
class AttractRadiusScalePlug(Plug):
	node : LineModifier = None
	pass
class BranchDropoutPlug(Plug):
	node : LineModifier = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : LineModifier = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : LineModifier = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : LineModifier = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	crb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	crg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	crr_ : ColorRPlug = PlugDescriptor("colorR")
	node : LineModifier = None
	pass
class DirectionalDisplacementPlug(Plug):
	node : LineModifier = None
	pass
class DirectionalForcePlug(Plug):
	node : LineModifier = None
	pass
class DisplacementPlug(Plug):
	node : LineModifier = None
	pass
class Dropoff_FloatValuePlug(Plug):
	parent : DropoffPlug = PlugDescriptor("dropoff")
	node : LineModifier = None
	pass
class Dropoff_InterpPlug(Plug):
	parent : DropoffPlug = PlugDescriptor("dropoff")
	node : LineModifier = None
	pass
class Dropoff_PositionPlug(Plug):
	parent : DropoffPlug = PlugDescriptor("dropoff")
	node : LineModifier = None
	pass
class DropoffPlug(Plug):
	dropoff_FloatValue_ : Dropoff_FloatValuePlug = PlugDescriptor("dropoff_FloatValue")
	drpfv_ : Dropoff_FloatValuePlug = PlugDescriptor("dropoff_FloatValue")
	dropoff_Interp_ : Dropoff_InterpPlug = PlugDescriptor("dropoff_Interp")
	drpi_ : Dropoff_InterpPlug = PlugDescriptor("dropoff_Interp")
	dropoff_Position_ : Dropoff_PositionPlug = PlugDescriptor("dropoff_Position")
	drpp_ : Dropoff_PositionPlug = PlugDescriptor("dropoff_Position")
	node : LineModifier = None
	pass
class DropoffNoisePlug(Plug):
	node : LineModifier = None
	pass
class FlowerDropoutPlug(Plug):
	node : LineModifier = None
	pass
class FlowerScalePlug(Plug):
	node : LineModifier = None
	pass
class ForcePlug(Plug):
	node : LineModifier = None
	pass
class InputMeshPlug(Plug):
	node : LineModifier = None
	pass
class LeafDropoutPlug(Plug):
	node : LineModifier = None
	pass
class LeafScalePlug(Plug):
	node : LineModifier = None
	pass
class LineExtendPlug(Plug):
	node : LineModifier = None
	pass
class ModifyColorPlug(Plug):
	node : LineModifier = None
	pass
class NoiseFrequencyPlug(Plug):
	node : LineModifier = None
	pass
class OccupyAttractionPlug(Plug):
	node : LineModifier = None
	pass
class OccupyBranchTerminationPlug(Plug):
	node : LineModifier = None
	pass
class OccupyGridResolutionPlug(Plug):
	node : LineModifier = None
	pass
class OccupyRadiusOffsetPlug(Plug):
	node : LineModifier = None
	pass
class OccupyRadiusScalePlug(Plug):
	node : LineModifier = None
	pass
class OpacityOffsetPlug(Plug):
	node : LineModifier = None
	pass
class OpacityScalePlug(Plug):
	node : LineModifier = None
	pass
class OutLineModifierPlug(Plug):
	node : LineModifier = None
	pass
class ShapePlug(Plug):
	node : LineModifier = None
	pass
class SurfaceOffsetPlug(Plug):
	node : LineModifier = None
	pass
class TubeDropoutPlug(Plug):
	node : LineModifier = None
	pass
class TubeScalePlug(Plug):
	node : LineModifier = None
	pass
class TwigDropoutPlug(Plug):
	node : LineModifier = None
	pass
class WidthOffsetPlug(Plug):
	node : LineModifier = None
	pass
class WidthScalePlug(Plug):
	node : LineModifier = None
	pass
# endregion


# define node class
class LineModifier(Shape):
	attractRadiusOffset_ : AttractRadiusOffsetPlug = PlugDescriptor("attractRadiusOffset")
	attractRadiusScale_ : AttractRadiusScalePlug = PlugDescriptor("attractRadiusScale")
	branchDropout_ : BranchDropoutPlug = PlugDescriptor("branchDropout")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	directionalDisplacement_ : DirectionalDisplacementPlug = PlugDescriptor("directionalDisplacement")
	directionalForce_ : DirectionalForcePlug = PlugDescriptor("directionalForce")
	displacement_ : DisplacementPlug = PlugDescriptor("displacement")
	dropoff_FloatValue_ : Dropoff_FloatValuePlug = PlugDescriptor("dropoff_FloatValue")
	dropoff_Interp_ : Dropoff_InterpPlug = PlugDescriptor("dropoff_Interp")
	dropoff_Position_ : Dropoff_PositionPlug = PlugDescriptor("dropoff_Position")
	dropoff_ : DropoffPlug = PlugDescriptor("dropoff")
	dropoffNoise_ : DropoffNoisePlug = PlugDescriptor("dropoffNoise")
	flowerDropout_ : FlowerDropoutPlug = PlugDescriptor("flowerDropout")
	flowerScale_ : FlowerScalePlug = PlugDescriptor("flowerScale")
	force_ : ForcePlug = PlugDescriptor("force")
	inputMesh_ : InputMeshPlug = PlugDescriptor("inputMesh")
	leafDropout_ : LeafDropoutPlug = PlugDescriptor("leafDropout")
	leafScale_ : LeafScalePlug = PlugDescriptor("leafScale")
	lineExtend_ : LineExtendPlug = PlugDescriptor("lineExtend")
	modifyColor_ : ModifyColorPlug = PlugDescriptor("modifyColor")
	noiseFrequency_ : NoiseFrequencyPlug = PlugDescriptor("noiseFrequency")
	occupyAttraction_ : OccupyAttractionPlug = PlugDescriptor("occupyAttraction")
	occupyBranchTermination_ : OccupyBranchTerminationPlug = PlugDescriptor("occupyBranchTermination")
	occupyGridResolution_ : OccupyGridResolutionPlug = PlugDescriptor("occupyGridResolution")
	occupyRadiusOffset_ : OccupyRadiusOffsetPlug = PlugDescriptor("occupyRadiusOffset")
	occupyRadiusScale_ : OccupyRadiusScalePlug = PlugDescriptor("occupyRadiusScale")
	opacityOffset_ : OpacityOffsetPlug = PlugDescriptor("opacityOffset")
	opacityScale_ : OpacityScalePlug = PlugDescriptor("opacityScale")
	outLineModifier_ : OutLineModifierPlug = PlugDescriptor("outLineModifier")
	shape_ : ShapePlug = PlugDescriptor("shape")
	surfaceOffset_ : SurfaceOffsetPlug = PlugDescriptor("surfaceOffset")
	tubeDropout_ : TubeDropoutPlug = PlugDescriptor("tubeDropout")
	tubeScale_ : TubeScalePlug = PlugDescriptor("tubeScale")
	twigDropout_ : TwigDropoutPlug = PlugDescriptor("twigDropout")
	widthOffset_ : WidthOffsetPlug = PlugDescriptor("widthOffset")
	widthScale_ : WidthScalePlug = PlugDescriptor("widthScale")

	# node attributes

	typeName = "lineModifier"
	apiTypeInt = 978
	apiTypeStr = "kLineModifier"
	typeIdInt = 1280134980
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["attractRadiusOffset", "attractRadiusScale", "branchDropout", "colorB", "colorG", "colorR", "color", "directionalDisplacement", "directionalForce", "displacement", "dropoff_FloatValue", "dropoff_Interp", "dropoff_Position", "dropoff", "dropoffNoise", "flowerDropout", "flowerScale", "force", "inputMesh", "leafDropout", "leafScale", "lineExtend", "modifyColor", "noiseFrequency", "occupyAttraction", "occupyBranchTermination", "occupyGridResolution", "occupyRadiusOffset", "occupyRadiusScale", "opacityOffset", "opacityScale", "outLineModifier", "shape", "surfaceOffset", "tubeDropout", "tubeScale", "twigDropout", "widthOffset", "widthScale"]
	nodeLeafPlugs = ["attractRadiusOffset", "attractRadiusScale", "branchDropout", "color", "directionalDisplacement", "directionalForce", "displacement", "dropoff", "dropoffNoise", "flowerDropout", "flowerScale", "force", "inputMesh", "leafDropout", "leafScale", "lineExtend", "modifyColor", "noiseFrequency", "occupyAttraction", "occupyBranchTermination", "occupyGridResolution", "occupyRadiusOffset", "occupyRadiusScale", "opacityOffset", "opacityScale", "outLineModifier", "shape", "surfaceOffset", "tubeDropout", "tubeScale", "twigDropout", "widthOffset", "widthScale"]
	pass

