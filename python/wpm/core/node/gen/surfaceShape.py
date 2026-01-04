

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ControlPoint = Catalogue.ControlPoint
else:
	from .. import retriever
	ControlPoint = retriever.getNodeCls("ControlPoint")
	assert ControlPoint

# add node doc



# region plug type defs
class BoundingBoxScaleXPlug(Plug):
	parent : BoundingBoxScalePlug = PlugDescriptor("boundingBoxScale")
	node : SurfaceShape = None
	pass
class BoundingBoxScaleYPlug(Plug):
	parent : BoundingBoxScalePlug = PlugDescriptor("boundingBoxScale")
	node : SurfaceShape = None
	pass
class BoundingBoxScaleZPlug(Plug):
	parent : BoundingBoxScalePlug = PlugDescriptor("boundingBoxScale")
	node : SurfaceShape = None
	pass
class BoundingBoxScalePlug(Plug):
	boundingBoxScaleX_ : BoundingBoxScaleXPlug = PlugDescriptor("boundingBoxScaleX")
	bscx_ : BoundingBoxScaleXPlug = PlugDescriptor("boundingBoxScaleX")
	boundingBoxScaleY_ : BoundingBoxScaleYPlug = PlugDescriptor("boundingBoxScaleY")
	bscy_ : BoundingBoxScaleYPlug = PlugDescriptor("boundingBoxScaleY")
	boundingBoxScaleZ_ : BoundingBoxScaleZPlug = PlugDescriptor("boundingBoxScaleZ")
	bscz_ : BoundingBoxScaleZPlug = PlugDescriptor("boundingBoxScaleZ")
	node : SurfaceShape = None
	pass
class CollisionDepthVelocityIncrement_FloatValuePlug(Plug):
	parent : CollisionDepthVelocityIncrementPlug = PlugDescriptor("collisionDepthVelocityIncrement")
	node : SurfaceShape = None
	pass
class CollisionDepthVelocityIncrement_InterpPlug(Plug):
	parent : CollisionDepthVelocityIncrementPlug = PlugDescriptor("collisionDepthVelocityIncrement")
	node : SurfaceShape = None
	pass
class CollisionDepthVelocityIncrement_PositionPlug(Plug):
	parent : CollisionDepthVelocityIncrementPlug = PlugDescriptor("collisionDepthVelocityIncrement")
	node : SurfaceShape = None
	pass
class CollisionDepthVelocityIncrementPlug(Plug):
	collisionDepthVelocityIncrement_FloatValue_ : CollisionDepthVelocityIncrement_FloatValuePlug = PlugDescriptor("collisionDepthVelocityIncrement_FloatValue")
	cdvifv_ : CollisionDepthVelocityIncrement_FloatValuePlug = PlugDescriptor("collisionDepthVelocityIncrement_FloatValue")
	collisionDepthVelocityIncrement_Interp_ : CollisionDepthVelocityIncrement_InterpPlug = PlugDescriptor("collisionDepthVelocityIncrement_Interp")
	cdvii_ : CollisionDepthVelocityIncrement_InterpPlug = PlugDescriptor("collisionDepthVelocityIncrement_Interp")
	collisionDepthVelocityIncrement_Position_ : CollisionDepthVelocityIncrement_PositionPlug = PlugDescriptor("collisionDepthVelocityIncrement_Position")
	cdvip_ : CollisionDepthVelocityIncrement_PositionPlug = PlugDescriptor("collisionDepthVelocityIncrement_Position")
	node : SurfaceShape = None
	pass
class CollisionDepthVelocityMultiplier_FloatValuePlug(Plug):
	parent : CollisionDepthVelocityMultiplierPlug = PlugDescriptor("collisionDepthVelocityMultiplier")
	node : SurfaceShape = None
	pass
class CollisionDepthVelocityMultiplier_InterpPlug(Plug):
	parent : CollisionDepthVelocityMultiplierPlug = PlugDescriptor("collisionDepthVelocityMultiplier")
	node : SurfaceShape = None
	pass
class CollisionDepthVelocityMultiplier_PositionPlug(Plug):
	parent : CollisionDepthVelocityMultiplierPlug = PlugDescriptor("collisionDepthVelocityMultiplier")
	node : SurfaceShape = None
	pass
class CollisionDepthVelocityMultiplierPlug(Plug):
	collisionDepthVelocityMultiplier_FloatValue_ : CollisionDepthVelocityMultiplier_FloatValuePlug = PlugDescriptor("collisionDepthVelocityMultiplier_FloatValue")
	cdvmfv_ : CollisionDepthVelocityMultiplier_FloatValuePlug = PlugDescriptor("collisionDepthVelocityMultiplier_FloatValue")
	collisionDepthVelocityMultiplier_Interp_ : CollisionDepthVelocityMultiplier_InterpPlug = PlugDescriptor("collisionDepthVelocityMultiplier_Interp")
	cdvmi_ : CollisionDepthVelocityMultiplier_InterpPlug = PlugDescriptor("collisionDepthVelocityMultiplier_Interp")
	collisionDepthVelocityMultiplier_Position_ : CollisionDepthVelocityMultiplier_PositionPlug = PlugDescriptor("collisionDepthVelocityMultiplier_Position")
	cdvmp_ : CollisionDepthVelocityMultiplier_PositionPlug = PlugDescriptor("collisionDepthVelocityMultiplier_Position")
	node : SurfaceShape = None
	pass
class CollisionOffsetVelocityIncrement_FloatValuePlug(Plug):
	parent : CollisionOffsetVelocityIncrementPlug = PlugDescriptor("collisionOffsetVelocityIncrement")
	node : SurfaceShape = None
	pass
class CollisionOffsetVelocityIncrement_InterpPlug(Plug):
	parent : CollisionOffsetVelocityIncrementPlug = PlugDescriptor("collisionOffsetVelocityIncrement")
	node : SurfaceShape = None
	pass
class CollisionOffsetVelocityIncrement_PositionPlug(Plug):
	parent : CollisionOffsetVelocityIncrementPlug = PlugDescriptor("collisionOffsetVelocityIncrement")
	node : SurfaceShape = None
	pass
class CollisionOffsetVelocityIncrementPlug(Plug):
	collisionOffsetVelocityIncrement_FloatValue_ : CollisionOffsetVelocityIncrement_FloatValuePlug = PlugDescriptor("collisionOffsetVelocityIncrement_FloatValue")
	covifv_ : CollisionOffsetVelocityIncrement_FloatValuePlug = PlugDescriptor("collisionOffsetVelocityIncrement_FloatValue")
	collisionOffsetVelocityIncrement_Interp_ : CollisionOffsetVelocityIncrement_InterpPlug = PlugDescriptor("collisionOffsetVelocityIncrement_Interp")
	covii_ : CollisionOffsetVelocityIncrement_InterpPlug = PlugDescriptor("collisionOffsetVelocityIncrement_Interp")
	collisionOffsetVelocityIncrement_Position_ : CollisionOffsetVelocityIncrement_PositionPlug = PlugDescriptor("collisionOffsetVelocityIncrement_Position")
	covip_ : CollisionOffsetVelocityIncrement_PositionPlug = PlugDescriptor("collisionOffsetVelocityIncrement_Position")
	node : SurfaceShape = None
	pass
class CollisionOffsetVelocityMultiplier_FloatValuePlug(Plug):
	parent : CollisionOffsetVelocityMultiplierPlug = PlugDescriptor("collisionOffsetVelocityMultiplier")
	node : SurfaceShape = None
	pass
class CollisionOffsetVelocityMultiplier_InterpPlug(Plug):
	parent : CollisionOffsetVelocityMultiplierPlug = PlugDescriptor("collisionOffsetVelocityMultiplier")
	node : SurfaceShape = None
	pass
class CollisionOffsetVelocityMultiplier_PositionPlug(Plug):
	parent : CollisionOffsetVelocityMultiplierPlug = PlugDescriptor("collisionOffsetVelocityMultiplier")
	node : SurfaceShape = None
	pass
class CollisionOffsetVelocityMultiplierPlug(Plug):
	collisionOffsetVelocityMultiplier_FloatValue_ : CollisionOffsetVelocityMultiplier_FloatValuePlug = PlugDescriptor("collisionOffsetVelocityMultiplier_FloatValue")
	covmfv_ : CollisionOffsetVelocityMultiplier_FloatValuePlug = PlugDescriptor("collisionOffsetVelocityMultiplier_FloatValue")
	collisionOffsetVelocityMultiplier_Interp_ : CollisionOffsetVelocityMultiplier_InterpPlug = PlugDescriptor("collisionOffsetVelocityMultiplier_Interp")
	covmi_ : CollisionOffsetVelocityMultiplier_InterpPlug = PlugDescriptor("collisionOffsetVelocityMultiplier_Interp")
	collisionOffsetVelocityMultiplier_Position_ : CollisionOffsetVelocityMultiplier_PositionPlug = PlugDescriptor("collisionOffsetVelocityMultiplier_Position")
	covmp_ : CollisionOffsetVelocityMultiplier_PositionPlug = PlugDescriptor("collisionOffsetVelocityMultiplier_Position")
	node : SurfaceShape = None
	pass
class DisplayHWEnvironmentPlug(Plug):
	node : SurfaceShape = None
	pass
class DoubleSidedPlug(Plug):
	node : SurfaceShape = None
	pass
class ExtraSampleRatePlug(Plug):
	node : SurfaceShape = None
	pass
class FeatureDisplacementPlug(Plug):
	node : SurfaceShape = None
	pass
class HoldOutPlug(Plug):
	node : SurfaceShape = None
	pass
class IgnoreHwShaderPlug(Plug):
	node : SurfaceShape = None
	pass
class InitialSampleRatePlug(Plug):
	node : SurfaceShape = None
	pass
class NormalThresholdPlug(Plug):
	node : SurfaceShape = None
	pass
class OppositePlug(Plug):
	node : SurfaceShape = None
	pass
class SmoothShadingPlug(Plug):
	node : SurfaceShape = None
	pass
class TextureThresholdPlug(Plug):
	node : SurfaceShape = None
	pass
# endregion


# define node class
class SurfaceShape(ControlPoint):
	boundingBoxScaleX_ : BoundingBoxScaleXPlug = PlugDescriptor("boundingBoxScaleX")
	boundingBoxScaleY_ : BoundingBoxScaleYPlug = PlugDescriptor("boundingBoxScaleY")
	boundingBoxScaleZ_ : BoundingBoxScaleZPlug = PlugDescriptor("boundingBoxScaleZ")
	boundingBoxScale_ : BoundingBoxScalePlug = PlugDescriptor("boundingBoxScale")
	collisionDepthVelocityIncrement_FloatValue_ : CollisionDepthVelocityIncrement_FloatValuePlug = PlugDescriptor("collisionDepthVelocityIncrement_FloatValue")
	collisionDepthVelocityIncrement_Interp_ : CollisionDepthVelocityIncrement_InterpPlug = PlugDescriptor("collisionDepthVelocityIncrement_Interp")
	collisionDepthVelocityIncrement_Position_ : CollisionDepthVelocityIncrement_PositionPlug = PlugDescriptor("collisionDepthVelocityIncrement_Position")
	collisionDepthVelocityIncrement_ : CollisionDepthVelocityIncrementPlug = PlugDescriptor("collisionDepthVelocityIncrement")
	collisionDepthVelocityMultiplier_FloatValue_ : CollisionDepthVelocityMultiplier_FloatValuePlug = PlugDescriptor("collisionDepthVelocityMultiplier_FloatValue")
	collisionDepthVelocityMultiplier_Interp_ : CollisionDepthVelocityMultiplier_InterpPlug = PlugDescriptor("collisionDepthVelocityMultiplier_Interp")
	collisionDepthVelocityMultiplier_Position_ : CollisionDepthVelocityMultiplier_PositionPlug = PlugDescriptor("collisionDepthVelocityMultiplier_Position")
	collisionDepthVelocityMultiplier_ : CollisionDepthVelocityMultiplierPlug = PlugDescriptor("collisionDepthVelocityMultiplier")
	collisionOffsetVelocityIncrement_FloatValue_ : CollisionOffsetVelocityIncrement_FloatValuePlug = PlugDescriptor("collisionOffsetVelocityIncrement_FloatValue")
	collisionOffsetVelocityIncrement_Interp_ : CollisionOffsetVelocityIncrement_InterpPlug = PlugDescriptor("collisionOffsetVelocityIncrement_Interp")
	collisionOffsetVelocityIncrement_Position_ : CollisionOffsetVelocityIncrement_PositionPlug = PlugDescriptor("collisionOffsetVelocityIncrement_Position")
	collisionOffsetVelocityIncrement_ : CollisionOffsetVelocityIncrementPlug = PlugDescriptor("collisionOffsetVelocityIncrement")
	collisionOffsetVelocityMultiplier_FloatValue_ : CollisionOffsetVelocityMultiplier_FloatValuePlug = PlugDescriptor("collisionOffsetVelocityMultiplier_FloatValue")
	collisionOffsetVelocityMultiplier_Interp_ : CollisionOffsetVelocityMultiplier_InterpPlug = PlugDescriptor("collisionOffsetVelocityMultiplier_Interp")
	collisionOffsetVelocityMultiplier_Position_ : CollisionOffsetVelocityMultiplier_PositionPlug = PlugDescriptor("collisionOffsetVelocityMultiplier_Position")
	collisionOffsetVelocityMultiplier_ : CollisionOffsetVelocityMultiplierPlug = PlugDescriptor("collisionOffsetVelocityMultiplier")
	displayHWEnvironment_ : DisplayHWEnvironmentPlug = PlugDescriptor("displayHWEnvironment")
	doubleSided_ : DoubleSidedPlug = PlugDescriptor("doubleSided")
	extraSampleRate_ : ExtraSampleRatePlug = PlugDescriptor("extraSampleRate")
	featureDisplacement_ : FeatureDisplacementPlug = PlugDescriptor("featureDisplacement")
	holdOut_ : HoldOutPlug = PlugDescriptor("holdOut")
	ignoreHwShader_ : IgnoreHwShaderPlug = PlugDescriptor("ignoreHwShader")
	initialSampleRate_ : InitialSampleRatePlug = PlugDescriptor("initialSampleRate")
	normalThreshold_ : NormalThresholdPlug = PlugDescriptor("normalThreshold")
	opposite_ : OppositePlug = PlugDescriptor("opposite")
	smoothShading_ : SmoothShadingPlug = PlugDescriptor("smoothShading")
	textureThreshold_ : TextureThresholdPlug = PlugDescriptor("textureThreshold")

	# node attributes

	typeName = "surfaceShape"
	typeIdInt = 1129534291
	nodeLeafClassAttrs = ["boundingBoxScaleX", "boundingBoxScaleY", "boundingBoxScaleZ", "boundingBoxScale", "collisionDepthVelocityIncrement_FloatValue", "collisionDepthVelocityIncrement_Interp", "collisionDepthVelocityIncrement_Position", "collisionDepthVelocityIncrement", "collisionDepthVelocityMultiplier_FloatValue", "collisionDepthVelocityMultiplier_Interp", "collisionDepthVelocityMultiplier_Position", "collisionDepthVelocityMultiplier", "collisionOffsetVelocityIncrement_FloatValue", "collisionOffsetVelocityIncrement_Interp", "collisionOffsetVelocityIncrement_Position", "collisionOffsetVelocityIncrement", "collisionOffsetVelocityMultiplier_FloatValue", "collisionOffsetVelocityMultiplier_Interp", "collisionOffsetVelocityMultiplier_Position", "collisionOffsetVelocityMultiplier", "displayHWEnvironment", "doubleSided", "extraSampleRate", "featureDisplacement", "holdOut", "ignoreHwShader", "initialSampleRate", "normalThreshold", "opposite", "smoothShading", "textureThreshold"]
	nodeLeafPlugs = ["boundingBoxScale", "collisionDepthVelocityIncrement", "collisionDepthVelocityMultiplier", "collisionOffsetVelocityIncrement", "collisionOffsetVelocityMultiplier", "displayHWEnvironment", "doubleSided", "extraSampleRate", "featureDisplacement", "holdOut", "ignoreHwShader", "initialSampleRate", "normalThreshold", "opposite", "smoothShading", "textureThreshold"]
	pass

