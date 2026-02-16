from __future__ import annotations
import types, typing as T
import pprint
import math
from wplib import log

from wpm import om, omr, omui
from wpm.lib.plugin import PluginNodeTemplate, PluginNodeIdData, PluginDrawOverrideTemplate
from wpm.lib.plugin.template import PluginMPxData
from wpsim import force
from wpsim.kine.builder import BuilderForceField

def maya_useNewAPI():
    pass
class WpSimForceFieldMPxData(PluginMPxData):
	"""data container for force field info"""
	clsName = "wpSimForceFieldMPxData"
	dataClsT = BuilderForceField
	kTypeId = om.MTypeId(0x00112234)


def readForceFieldParams(node: om.MObject) -> tuple[int, float, float, tuple[float, float, float]]:
	shapeType = om.MPlug(node, WpSimForceFieldNode.aShapeType).asShort()
	radius = om.MPlug(node, WpSimForceFieldNode.aRadius).asDouble()
	height = om.MPlug(node, WpSimForceFieldNode.aHeight).asDouble()
	halfPlug = om.MPlug(node, WpSimForceFieldNode.aHalfExtents)
	halfExtents = (
		halfPlug.child(0).asDouble(),
		halfPlug.child(1).asDouble(),
		halfPlug.child(2).asDouble(),
	)
	return shapeType, radius, height, halfExtents


def buildForceFieldBounds(
		shapeType: int,
		radius: float,
		height: float,
		halfExtents: tuple[float, float, float]
) -> om.MBoundingBox:
	if shapeType == force.shapeCapsule:
		halfHeight = 0.5 * height
		extents = om.MVector(
			radius,
			halfHeight + radius,
			radius
		)
	elif shapeType == force.shapeCube:
		extents = om.MVector(*halfExtents)
	else:
		extents = om.MVector(radius, radius, radius)

	minPoint = om.MPoint(-extents.x, -extents.y, -extents.z)
	maxPoint = om.MPoint(extents.x, extents.y, extents.z)
	return om.MBoundingBox(minPoint, maxPoint)

class WpSimForceFieldNode(PluginNodeTemplate, omui.MPxLocatorNode):
	"""force field shape node with simple debug drawing"""
	kDrawClassification = "drawdb/geometry/wpSimForceField"

	@classmethod
	def pluginNodeIdData(cls) -> PluginNodeIdData:
		return PluginNodeIdData(
			"wpSimForceField",
			om.MPxNode.kLocatorNode
		)

	@classmethod
	def initialiseNode(cls):
		nFn = om.MFnNumericAttribute()
		eFn = om.MFnEnumAttribute()
		tFn = om.MFnTypedAttribute()

		cls.aName = tFn.create(
			"name", "name", om.MFnData.kString
		)

		cls.aShapeType = eFn.create("shapeType", "shapeType", force.shapeSphere)
		eFn.addField("sphere", force.shapeSphere)
		eFn.addField("capsule", force.shapeCapsule)
		eFn.addField("cube", force.shapeCube)

		cls.aRadius = nFn.create(
			"radius", "radius", om.MFnNumericData.kDouble, 1.0
		)
		nFn.setMin(0.0)

		cls.aHeight = nFn.create(
			"height", "height", om.MFnNumericData.kDouble, 1.0
		)
		nFn.setMin(0.0)

		cls.aHalfExtents = nFn.create(
			"halfExtents", "halfExtents", om.MFnNumericData.k3Double
		)
		nFn.setMin((0, 0, 0))
		nFn.default = (0.5, 0.5, 0.5)

		cls.aFieldData = tFn.create(
			"fieldData", "fieldData", WpSimForceFieldMPxData.kTypeId
		)
		tFn.writable = False

		cls.addAttribute(cls.aName)
		cls.addAttribute(cls.aShapeType)
		cls.addAttribute(cls.aRadius)
		cls.addAttribute(cls.aHeight)
		cls.addAttribute(cls.aHalfExtents)

		cls.addAttribute(cls.aFieldData)
		cls.setAttributesAffect([
			cls.aName, cls.aShapeType, cls.aRadius, cls.aHeight,
			cls.aHalfExtents
		],
		[cls.aFieldData])

	def compute(self, pPlug:om.MPlug, pData:om.MDataBlock):
		if pData.isClean(pPlug):
			return
		name = pData.inputValue(self.aName).asString() or self.name()
		shapeType = pData.inputValue(self.aShapeType).asShort()
		radius = pData.inputValue(self.aRadius).asDouble()
		height = pData.inputValue(self.aHeight).asDouble()
		halfExtents = pData.inputValue(self.aHalfExtents).asFloat3()

		forceField = BuilderForceField(
			name=name,
			shapeType=shapeType,
			radius=radius,
			height=height,
			halfExtents=halfExtents
		 )
		fieldDataHandle = pData.outputValue(self.aFieldData)
		mpxData = WpSimForceFieldMPxData()
		mpxData.setDataCls(forceField)
		fieldDataHandle.setMPxData(mpxData)
		pData.setClean(self.aFieldData)
		pData.setClean(pPlug)

	def isBounded(self) -> bool:
		return True

	def boundingBox(self) -> om.MBoundingBox:
		shapeType, radius, height, halfExtents = readForceFieldParams(
			self.thisMObject()
		)
		return buildForceFieldBounds(shapeType, radius, height, halfExtents)


class WpSimForceFieldDrawData(om.MUserData):
	def __init__(self):
		om.MUserData.__init__(self, False)
		self.shapeType = force.shapeSphere
		self.radius = 1.0
		self.height = 1.0
		self.halfExtents = (0.5, 0.5, 0.5)


class WpSimForceFieldDrawOverride(omr.MPxDrawOverride, PluginDrawOverrideTemplate):
	def __init__(self, obj):
		omr.MPxDrawOverride.__init__(self, obj, self.drawCallback, True)

	@staticmethod
	def drawCallback(*args, **kwargs):
		return

	def isBounded(self, objPath, cameraPath):
		return True

	def boundingBox(self, objPath, cameraPath):
		shapeType, radius, height, halfExtents = readForceFieldParams(
			objPath.node()
		)
		return buildForceFieldBounds(shapeType, radius, height, halfExtents)

	def supportedDrawAPIs(self):
		return omr.MRenderer.kOpenGLCoreProfile

	def hasUIDrawables(self):
		return True

	def addUIDrawables(self, objPath, drawManager, frameContext,
	                   data: WpSimForceFieldDrawData):
		if data is None:
			return
		drawManager.beginDrawable(
			selectability=omr.MUIDrawManager.kNonSelectable,
			selectionName=0
		)
		drawManager.setColor(om.MColor((0.7, 0.7, 0.7, 1.0)))

		if data.shapeType == force.shapeCapsule:
			self.drawCapsule(drawManager, data.radius, data.height)
		elif data.shapeType == force.shapeCube:
			self.drawBox(drawManager, data.halfExtents)
		else:
			self.drawSphere(drawManager, data.radius)

		drawManager.endDrawable()

	def prepareForDraw(self, objPath, cameraPath, frameContext, oldData):
		if not isinstance(oldData, WpSimForceFieldDrawData):
			oldData = WpSimForceFieldDrawData()

		shapeType, radius, height, halfExtents = readForceFieldParams(
			objPath.node()
		)
		oldData.shapeType = shapeType
		oldData.radius = radius
		oldData.height = height
		oldData.halfExtents = halfExtents
		return oldData

	def drawCircle(self, drawManager, axis: str, radius: float,
	               yOffset: float = 0.0, segments: int = 24):
		for i in range(segments):
			angle0 = (2.0 * math.pi) * (i / float(segments))
			angle1 = (2.0 * math.pi) * ((i + 1) / float(segments))
			point0 = self.circlePoint(axis, radius, angle0, yOffset)
			point1 = self.circlePoint(axis, radius, angle1, yOffset)
			drawManager.line(point0, point1)

	def circlePoint(self, axis: str, radius: float, angle: float,
	                yOffset: float) -> om.MPoint:
		cosT = math.cos(angle)
		sinT = math.sin(angle)
		if axis == "xy":
			return om.MPoint(radius * cosT, radius * sinT + yOffset, 0.0)
		if axis == "xz":
			return om.MPoint(radius * cosT, yOffset, radius * sinT)
		return om.MPoint(0.0, radius * cosT + yOffset, radius * sinT)

	def drawSphere(self, drawManager, radius: float):
		self.drawCircle(drawManager, "xy", radius)
		self.drawCircle(drawManager, "xz", radius)
		self.drawCircle(drawManager, "yz", radius)

	def drawCapsule(self, drawManager, radius: float, height: float):
		halfHeight = 0.5 * height
		self.drawCircle(drawManager, "xz", radius, yOffset=halfHeight)
		self.drawCircle(drawManager, "xz", radius, yOffset=-halfHeight)
		self.drawCircle(drawManager, "xy", radius)
		self.drawCircle(drawManager, "yz", radius)

		self.drawLine(drawManager, (radius, halfHeight, 0.0),
		              (radius, -halfHeight, 0.0))
		self.drawLine(drawManager, (-radius, halfHeight, 0.0),
		              (-radius, -halfHeight, 0.0))
		self.drawLine(drawManager, (0.0, halfHeight, radius),
		              (0.0, -halfHeight, radius))
		self.drawLine(drawManager, (0.0, halfHeight, -radius),
		              (0.0, -halfHeight, -radius))

	def drawBox(self, drawManager, halfExtents: tuple[float, float, float]):
		hx, hy, hz = halfExtents
		corners = [
			(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz),
			(-hx, hy, -hz), (hx, hy, -hz), (hx, hy, hz), (-hx, hy, hz),
		]
		edges = [
			(0, 1), (1, 2), (2, 3), (3, 0),
			(4, 5), (5, 6), (6, 7), (7, 4),
			(0, 4), (1, 5), (2, 6), (3, 7),
		]
		for start, end in edges:
			self.drawLine(drawManager, corners[start], corners[end])

	def drawLine(self, drawManager, start: tuple[float, float, float],
	             end: tuple[float, float, float]):
		drawManager.line(om.MPoint(*start), om.MPoint(*end))
