from __future__ import annotations
import types, typing as T
import pprint
from wplib import log



from PySide2 import QtCore, QtWidgets, QtGui
import copy
import typing as T
import math, pprint
import numpy as np # for angles
from collections import namedtuple
from dataclasses import dataclass

"""tests for drawing various shaped circle segments, for
potential use in marking menus"""

SegmentData = namedtuple("SegmentData",
                         ("startAngle", "endAngle",
		"innerRadiusFract", "outerRadiusFract",
                     "maxRadius",
                     "marginAngle"))

def cubicTangentLengthForAngle(theta):
	"""
	taken from paper by Christian Apprich, Anne Dietrich,
	Klaus Höllig, Esfandiar Nava-Yazdani at Stuttgart university
	"""
	return (
			(1 - math.cos(theta)) /
			(3 * math.sin(theta) * (3 + 2 * math.cos(theta)))
			* (2 * (4 + 3 * math.cos(theta)) - math.sqrt(2 * (5 + 3 * math.cos(theta)))))


def linearAnglePoints(theta, n)->list[QtCore.QPointF]:
	"""return n points on arc of theta"""
	return [QtCore.QPointF(math.sin(theta / n * i),
	                       math.cos(theta / n * i)
	                       ) for i in range(n)]


def cubicPointsForAngleInterval(startAngle, endAngle):
	"""return uniform QPointFs giving the approximate cubic solution
	to the given circular angle
	"""
	rot = QtGui.QTransform()
	rot.rotate(90)
	invRot = QtGui.QTransform()
	invRot.rotate(-90)
	startAngle = math.radians(startAngle)
	endAngle = math.radians(endAngle)
	tanLength = cubicTangentLengthForAngle(endAngle - startAngle)
	# place point at start of arc, then tangent to it
	lesserDirVec = QtCore.QPointF(math.sin((startAngle)),
	                              math.cos((startAngle)))
	lesserTanVec = invRot.map(lesserDirVec) * tanLength

	greaterDirVec = QtCore.QPointF(math.sin((endAngle)),
	                               math.cos((endAngle)))
	greaterTanVec = rot.map(greaterDirVec) * tanLength
	return (lesserDirVec,
	        lesserDirVec + lesserTanVec,
	        greaterDirVec + greaterTanVec,
	        greaterDirVec)

def pointsForSegment(segData:SegmentData)->list[QtCore.QPoint]:
	"""some extra maths to give straight margin edges"""

	def innerOuterPoints(baseAngle, finalMarginAngle, innerRad, outerRad):
		radDiff = outerRad - innerRad
		radRatio = outerRad / innerRad

		baseAngle = math.radians(baseAngle)
		startAngleArr = np.array([math.sin(baseAngle), math.cos(baseAngle)])
		finalMarginAngle = math.radians(finalMarginAngle)
		startAngleMarginArr = np.array([math.sin(finalMarginAngle), math.cos(finalMarginAngle)])

		innerPoint = startAngleMarginArr * innerRad
		outerPoint = innerPoint + startAngleArr * (math.cos(finalMarginAngle - baseAngle)) * radDiff
		#outerPoint = innerPoint + startAngleArr * radDiff
		return innerPoint, outerPoint



	innerRad = segData.innerRadiusFract * segData.maxRadius
	outerRad = segData.outerRadiusFract * segData.maxRadius


	startDirInnerPoint, startDirOuterPoint = innerOuterPoints(segData.startAngle, segData.startAngle + segData.marginAngle, innerRad, outerRad)
	endDirInnerPoint, endDirOuterPoint = innerOuterPoints(segData.endAngle, segData.endAngle , innerRad, outerRad)

	points = [startDirInnerPoint, startDirOuterPoint, endDirOuterPoint, endDirInnerPoint]
	return [QtCore.QPointF(*i).toPoint() for i in points]


def pathThroughPoints(points:list[QtCore.QPoint],
                      pathToExtend:QtGui.QPainterPath=None)->QtGui.QPainterPath:
	path = pathToExtend or QtGui.QPainterPath(points[0])
	if pathToExtend:
		path.lineTo(points[0])
	for i in range(1, len(points)):
		path.lineTo(points[i])
	return path

def pathForArc(startAngle, endAngle, radius=1.0):
	points = []
	theta = endAngle - startAngle
	subRes = math.floor(theta / 90) + 1
	span = theta / subRes
	prevAngle = startAngle

	for i in range(subRes):
		points.extend(cubicPointsForAngleInterval(prevAngle, prevAngle + span))
		prevAngle += span
	points = [i * radius for i in points]
	path = QtGui.QPainterPath(points[0])

	for i in range(subRes):
		path.cubicTo(points[i * 4 + 1], points[i * 4 + 2], points[i * 4 + 3])
	return path, points

def pathForSegment(segData:SegmentData,
                   )->tuple[QtGui.QPainterPath, list[QtCore.QPoint]]:
	allPoints = []
	theta = segData.endAngle - segData.startAngle
	subRes = math.floor(theta / 90) + 1 # number of internal segments to draw curves over

	innerRad = segData.innerRadiusFract * segData.maxRadius
	outerRad = segData.outerRadiusFract * segData.maxRadius
	points = pointsForSegment(segData)

	outerArcPoints = []
	innerArcPoints = []
	prevAngle = segData.startAngle
	span = theta / subRes

	for i in range(subRes):
		arcPoints = cubicPointsForAngleInterval(
			prevAngle, prevAngle + span)
		prevAngle = prevAngle + span

		outerArcPoints.extend(i * outerRad for i in arcPoints)
		innerArcPoints.extend(i * innerRad for i in arcPoints)
		allPoints.extend(allPoints)

	path = QtGui.QPainterPath()
	path.moveTo(points[0])
	path.lineTo(points[1])

	for i in range(subRes):
		path.cubicTo(outerArcPoints[i * 4 + 1], outerArcPoints[i * 4 + 2], outerArcPoints[i * 4 + 3])

	path.lineTo(points[-1])
	innerArcPoints = list(reversed(innerArcPoints))
	for i in range(subRes):
		path.cubicTo(innerArcPoints[i * 4 + 1], innerArcPoints[i * 4 + 2], innerArcPoints[i * 4 + 3])
	# if clockwise: # just flip on x lol
	# 	path.
	return path, allPoints


class RingSegment(QtWidgets.QGraphicsItem):

	def __init__(self, segData:SegmentData,
	             parent=None):
		super(RingSegment, self).__init__(parent)
		self.segData = segData
		assert self.segData.startAngle <= self.segData.endAngle, "Start angle must be less than end angle"

		self.subRes = math.floor(self.theta() / 90) + 1 # number of internal segments to draw curves over

		# list of points on segments, recached when recalculated
		self._pathPoints = []
		self._path : QtGui.QPainterPath = None

	def globalise(self, a):
		return self.mapToItem(self, a)

	def theta(self):
		return self.segData.endAngle - self.segData.startAngle

	def thetaRad(self):
		return math.radians(self.theta())

	def normal(self):
		return QtCore.QPointF(
			math.sin(math.radians(self.segData.startAngle) + self.thetaRad() / 2),
			math.cos(math.radians(self.segData.startAngle) + self.thetaRad() / 2))

	def mainPoints(self):
		"""return (inner-lesser, outer-lesser,
		outer-greater, inner-greater) points for this
		segment of a ring"""
		return list(map(self.globalise, pointsForSegment(self.segData
		)))

	def centrePath(self)->QtGui.QPainterPath:
		radius = (self.segData.innerRadiusFract + self.segData.outerRadiusFract) / 2 * self.segData.maxRadius
		return pathForArc(self.segData.startAngle, self.segData.endAngle, radius)[0]

	def boundingRect(self) -> QtCore.QRectF:
		#return self.boundingRegion().boundingRect()
		poly = QtGui.QPolygon(self.mainPoints())
		return poly.boundingRect()

	def boundingRegion(self, itemToDeviceTransform:QtGui.QTransform) -> QtGui.QRegion:
		return (QtGui.QRegion(QtGui.QPolygon(self._pathPoints)))


	def outlinePath(self)->QtGui.QPainterPath:
		"""approximate circular arc as multiple cubic paths"""
		self._pathPoints.clear()
		path, points = pathForSegment(self.segData)
		path = self.globalise(path)
		points = [self.globalise(i) for i in points]
		self._path = path
		self._pathPoints.extend(points)
		return path

	# def paint(self, painter:QtGui.QPainter, option:QtWidgets.QStyleOptionGraphicsItem, widget:T.Optional[QtWidgets.QWidget]=...) -> None:
	# 	path = self.outlinePath()
	# 	painter.drawPath(path)
	# 	arcPoints = self.cubicPointsForAngleInterval(self.startAngle, self.endAngle)
	# 	arcPoints = [i * self.maxRadius for i in arcPoints]
	# 	painter.drawEllipse(arcPoints[0], 2, 2)
	# 	painter.drawEllipse(arcPoints[1], 4, 4)
	# 	painter.drawEllipse(arcPoints[2], 6, 6)
	# 	painter.drawEllipse(arcPoints[3], 8, 8)





class RingGroup(QtWidgets.QGraphicsItemGroup):
	"""holds a group of ring segments together"""

	def __init__(self, nSegments=8, spacing=4,
	             totalAngle=360,
	             parent=None):
		super(RingGroup, self).__init__(parent)
		self.segments = []
		self.spacing = spacing
		startAngle = 0
		for i in range(nSegments):
			theta = totalAngle / (nSegments )
			seg = RingSegment(
				startAngle, startAngle + theta,
				parent=self
			)
			self.segments.append(seg)
			offsetPos = seg.normal() * self.spacing
			seg.moveBy(offsetPos.x(), offsetPos.y())

			startAngle += theta


class RingWidget(QtWidgets.QGraphicsView):
	pass

def test():

	import sys
	app = QtWidgets.QApplication(sys.argv)
	win = QtWidgets.QMainWindow()

	group = RingGroup(nSegments=12)

	widg = RingWidget()
	s = QtWidgets.QGraphicsScene()
	widg.setScene(s)
	widg.scene().addItem(group)


	win.setCentralWidget(widg)
	win.show()
	sys.exit(app.exec_())

if __name__ == '__main__':
	test()



