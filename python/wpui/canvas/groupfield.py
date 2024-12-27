from __future__ import annotations
import types, typing as T
import pprint

import scipy.spatial

from wplib import log

try:
	import numpy as np
	from scipy.spatial import ConvexHull
	from scipy.spatial import Voronoi
	from scipy.optimize import minimize_scalar, minimize
	SCIPY_ON = True
except ImportError:
	print("scipy is not available, switching to simple graph hull drawing")
	SCIPY_ON = False

from PySide2 import QtCore, QtWidgets, QtGui

from wplib.maths import shape, arr, arrT
from wpdex import *
from wpui.keystate import KeyState

"""class for drawing translucent shapes around / behind items
in a qgraphicsscene, usd for groups of nodes

extend for user interaction, dragging points, edges etc

"""

base = object
if T.TYPE_CHECKING:
	base = QtWidgets.QGraphicsItem
	from .view import WpCanvasView
	from .scene import WpCanvasScene
	from .element import WpCanvasElement


def roundedPathFromPolygon(poly:QtGui.QPolygon, roundedRadius:10):
	"""generate a rounded path from polygon vertices
	this is actually really hard, skip for now"""
	poly.toList()

def distanceFromQLine(line:QtCore.QLine,
                      pos:QtCore.QPoint):
	"""TODO: get proper support for 2d primitives with some kind of

	"""
	return



def closestParamOnQPainterPath(path:QtGui.QPainterPath,
                               pos:QtCore.QPointF,
                               iterations=(10, 5))->QtCore.QPointF:
	"""do something quite basic - brute force dividing path
	into the given sections

	TODO: maybe later do something to cache path positions into a numpy array,
		but we still have to sample the path very densely
	TODO: maybe we could even make this async?
		do we have a function on these called something like "refine"? where we
		just keep working on prettifying graphics if the scene is static?
	"""
	pos = arr(pos)
	def _distanceOnPath(x:float)->float:
		return np.linalg.norm(arr(path.pointAtPercent(x)) - pos)

	minT = minimize_scalar(_distanceOnPath,
	                       method="bounded",
	                       bounds=(0.0, 1.0),
	                       options={"maxiter" : 20}
	                       )
	return minT["x"]

def mutualClosestParamsOnQPainterPaths(paths:list[QtGui.QPainterPath],
                                       tol=1e-2,
                                       maxiter=20)->list[float]:
	"""test for multivariate scipy optimisation"""
	def _toMinimise(x:list[float]):
		"""sum of all distances between each point on each path"""
		return sum(
			sum(np.linalg.norm(
				arr(path.pointAtPercent(x[i])) -
				arr(otherPath.pointAtPercent(x[otherI]))
				) for otherI, otherPath in enumerate(paths)
			) for i, path in enumerate(paths)
		) # ship it
	initCoords = [0.5] * len(paths)
	result = minimize(_toMinimise,
	                  arr(initCoords),
	                  bounds=[(0.0, 1.0)] * len(paths),
	                  method="BGFS",
	                  tol=tol,
	                  options={"maxiter" : maxiter}
	                  )
	return result["x"]


class Tetherable:
	"""how we could show what element of a group node is driving
	the group node"""

	def getTetherSource(self, key="main")->(QtCore.QRectF, QtWidgets.QGraphicsItem,
	                            QtCore.QPointF):
		"""if rect, draw nearest point on rect edges
		if point, draw line straight from it,
		if it's an item, get its path or rect"""
		raise NotImplementedError

	@classmethod
	def drawTether(cls, a:Tetherable, b:Tetherable, key="main"):
		raise NotImplementedError



class UserPolygon(QtWidgets.QGraphicsPolygonItem):
	"""let user drag out their own shape -
	highlight edges and vertices on mouse over

	maybe this is better done with child items
	"""
	
	def __init__(self, parent=None,
	             allowSelectEdges=True,
	             allowSelectPoints=True,
	             pointRadius=10,
	             colour=(0.5, 0.5, 0.5),
	             lineThickness=2):
		super().__init__(parent=parent)
		self.ks = KeyState()
		self.allowSelectEdges = rx(allowSelectEdges)
		self.allowSelectPoints = rx(allowSelectPoints)
		self.pointRadius = rx(pointRadius)
		self.lineThickness = rx(lineThickness)
		self.colour = rx(colour).rx.pipe(lambda a : QtGui.QColor.fromRgbF(*a))
		for i in (self.pointRadius, lineThickness, colour):
			i.rx.watch(self._updateDrawing)

		self.elementsUnderMouse = [[], []] # points, edges

		self.childPoints :list[QtWidgets.QGraphicsEllipseItem] = []
		self.childLines : list[QtWidgets.QGraphicsLineItem] = []
		#self.isDragging = [[]]

		self.updateChildItems()

	def rebuildChildItems(self):
		pointArr = arr(self.polygon())
		rad = EVAL(self.pointRadius)
		thick = EVAL(self.lineThickness)
		for i in range(len(pointArr)):
			end = (i + 1) % len(pointArr)
			rect = QtCore.QRectF(0, 0, rad, rad)
			rect.moveCenter(fromArr(pointArr[i], QtCore.QPointF))
			self.childPoints.append(
				QtWidgets.QGraphicsEllipseItem(rect, self))
			line = fromArr(arr([pointArr[i], pointArr[end]]), QtCore.QLineF)
			self.childLines.append(QtWidgets.QGraphicsLineItem(
				line, self))

	def updateChildItems(self):
		"""create child graphics items for each edge and vertex of the polygon -
		makes selection and manipulation way easier, costs slightly more complicated
		events"""
		pointArr = arr(self.polygon())
		rad = EVAL(self.pointRadius)
		thick = EVAL(self.lineThickness)
		col = EVAL(self.colour)
		for i in range(len(pointArr)):
			end = (i + 1) % len(pointArr)
			rect = QtCore.QRectF(0, 0, rad, rad)
			rect.moveCenter(fromArr(pointArr[i], QtCore.QPointF))
			self.childPoints[i].setRect(rect)
			self.childPoints[i].setPen(QtGui.QPen(col))
			self.childPoints[i].setBrush(QtGui.QBrush(col))

			line = fromArr(arr([pointArr[i], pointArr[end]]), QtCore.QLineF)
			self.childLines[i].setLine(line)
			self.childLines[i].setPen(QtGui.QPen(col))
			self.childLines[i].setBrush(QtGui.QBrush(col))


	def _updateDrawing(self, *a):
		self.setPen(EVAL(self.colour))
		backCol = QtGui.QColor(EVAL(self.colour))
		backCol.setAlphaF(0.5)
		self.setBrush(EVAL(backCol))

	def asArr(self)->arrT:
		return arr(self.polygon())
	def setArr(self, a:arrT):
		self.setPolygon(fromArr(a, QtGui.QPolygonF))

	def mouseOverControlColour(self)->QtGui.QColor:
		return QtGui.QColor.fromRgbF(0.6, 0.6, 0.6, 0.6)
	def mouseDownControlColour(self)->QtGui.QColor:
		return QtGui.QColor.fromRgbF(1.0, 1.0, 1.0, 0.8)

	def points(self)->list[QtCore.QPointF]:
		return self.polygon().toList()

	def edgeLines(self)->list[QtCore.QLineF]:
		points = self.points()
		nPoints = len(self.points())
		lines = []
		for i in range(nPoints):
			start = i
			end = (i + 1) % nPoints
			lines.append(QtCore.QLineF(points[start], points[end]))
		return lines

	def edgeArrays(self):
		return np.array([arr(i) for i in self.edgeLines()])

	def distanceFromElements(self, pos,
	                   ):
		"""return distance of all points and edges to the given position. unsorted

		return [ (pointArr, distance, index) ]
			[ (edgeArr, distance, index) ]
		"""
		points = []
		lines = []
		pos = arr(pos)
		pointsArr = self.asArr()
		for i in range(len(self.points())):
			points.append((pointsArr[i], np.linalg.norm(pointsArr[i] - pos), i))
			nextI = (i + 1) % len(pointsArr)
			seg = shape.segmentFromPoints(pointsArr[i], pointsArr[nextI])
			lines.append((seg,
			               np.linalg.norm(pos - shape.closestPosOnSegment(seg, pos)),
			               i))
		return points, lines

	def mousePressEvent(self, event):
		self.ks.mousePressed(event)

	def mouseMoveEvent(self, event):
		"""highlight points and edges available for dragging"""
		self.ks.mouseMoved(event)

	def mouseReleaseEvent(self, event):
		self.ks.mouseReleased(event)

	def _getElementsUnderPos(self, pos, pointDistances, lineDistances):
		underPointIds = []
		underLineIds = []
		for pointArr, distance, index in pointDistances:
			if distance < self.pointRadius * 1.5 :
				underPointIds.append(index)
		for lineArr, distance, index in lineDistances:
			if distance < self.lineThickness * 3.0 :
				underLineIds.append(index)
		return underPointIds, underLineIds


	def hoverMoveEvent(self, event):
		self.ks.mouseMoved(event)
		points, lines = self.distanceFromElements(event.pos())
		underPointIds, underLineIds = self._getElementsUnderPos(
			event.pos(), points, lines
		)
		self.elementsUnderMouse[0] = underPointIds
		self.elementsUnderMouse[1] = underLineIds


	def hoverLeaveEvent(self, event):
		self.ks.reset()

	def paint(self, painter, option, widget=...):
		painter.setPen(self.pen())
		painter.setBrush(self.brush())
		painter.drawPolygon(self.polygon() )

		# draw highlights for all elements under pointer

	def boundingRect(self):
		rad = EVAL(self.pointRadius)
		return super().boundingRect().marginsAdded(
			QtCore.QMarginsF(rad, rad, rad, rad)
		)



class GroupPolygon(QtWidgets.QGraphicsPolygonItem):
	"""draw convex field around items
	later maybe add system for rounding corners

	TODO: voronoi

	TODO: define named LAYERS in canvas for drawing things like
		voronoi groups, that all have to be drawn at once
		Also lets us start turning things on and off for different views
	"""
	def __init__(self, parent=None,
	             items:T.Iterable[QtWidgets.QGraphicsItem]=()):
		super(GroupPolygon, self).__init__(parent)
		self.itemsToContain = items

	def containedPoints(self, boundingBoxPadding=2)->list[QtCore.QPointF]:
		"""return bounding box corners of all items contained in field"""
		points = list()
		margins = QtCore.QMargins(
			boundingBoxPadding, boundingBoxPadding,
			boundingBoxPadding, boundingBoxPadding		)
		for item in EVAL(self.itemsToContain):
			r = item.sceneBoundingRect()
			r = r.marginsAdded(margins)
			points.append(r.topLeft())
			points.append(r.topRight())
			points.append(r.bottomRight())
			points.append(r.bottomLeft())
		return points

	def _polygonVerticesSimple(self, points:list[QtCore.QPointF]):
		return points

	def _polygonVerticesScipy(self, points:list[QtCore.QPointF]):
		pointArray = np.array(points)
		hull = ConvexHull(pointArray, incremental=False)
		return list(map(QtCore.QPointF, hull.vertices))

	def polygonVertices(self, padding=2)->list[QtCore.QPointF]:
		"""return convex points for polygon"""
		containPoints = self.containedPoints(boundingBoxPadding=padding)
		if not SCIPY_ON:
			return self._polygonVerticesSimple(containPoints)
		return self._polygonVerticesScipy(containPoints)

	def sync(self):
		self.setPolygon(QtGui.QPolygonF(self.polygonVertices(padding=5)))
		pass

def rectToLines(rect:QtCore.QRect)->list[QtCore.QLineF]:
	lT = QtCore.QLine if isinstance(rect, QtCore.QRect) else QtCore.QLineF
	return [lT(rect.topLeft(), rect.topRight()), lT(rect.topRight(), rect.bottomRight()),
			lT(rect.bottomRight(), rect.bottomLeft()), lT(rect.bottomLeft(), rect.topLeft())]

class GroupVoronoi(GroupPolygon):
	"""
	voronoi field drawn around contained items,
	needs to be wrangled with other fields in the same layer
	for the effect to work - manage that at scene-level

	actually seems very complicated

	to connect regions, do a bit of brute force, draw outline paths
	for each one and combine them with QT's boolean features
	"""

	def __init__(self, items=(), parent=None):
		super().__init__(parent=parent, items=items)
		self.regions : list[arrT] = []
		self.paths : list[QtGui.QPainterPath] = []
		self.polygons = []


	def fixInfiniteVertices(self, baseRegion:arrT, seedPoint:arrT):
		""" 'fix' voronoi vertices placed at infinity,
		replace with the nearest points on the scene rect
		to each one"""

		rectSegments = [arr(i) for i in rectToLines(self.sceneBoundingRect())]
		minDist = np.inf
		nearSegmentIndex = None
		nearPos = None
		for i, segment in enumerate(rectSegments):
			pos = shape.closestPosOnSegment(segment, seedPoint)
			if np.linalg.norm(pos - seedPoint) < minDist:
				nearSegmentIndex = i
				nearPos = pos
		return nearPos, nearSegmentIndex



	@classmethod
	def polygonsForVoronoiGroups(cls,
	                             groups:T.List[GroupVoronoi]
	                             ):
		"""return list of polygons for each passed in group,
		using its contained points as centres"""
		points = []
		groupBounds = []
		vtxGroupIds = []
		index = 0
		for i, grp in enumerate(groups): #type:GroupVoronoi
			grpPointsArr = arr(list(map(arr, grp.containedPoints())))
			points.extend(grpPointsArr)
			groupBounds.append(index)
			index = index + len(grpPointsArr)
			groupBounds.append(index)
			vtxGroupIds.extend([i for n in grpPointsArr])

		assert SCIPY_ON
		# build voronoi pattern from points
		points = np.array(points)
		vo = Voronoi(points)
		groupRegions = [ [] for i in groups]
		# one voronoi region per seed point
		for i, region in enumerate(vo.regions):
			# try to "fix" -1 infinity points in this region -
			region = list(region)
			region.INDEX = i # pack index on list, sue me
			groupRegions[vtxGroupIds[i]].append(region)


		# for now just set regions here, let group work out how to draw them, fix missing vertices, etc
		for i, group in enumerate(groups): #type:GroupVoronoi
			group.regions = groupRegions[i]
			group.polygons = group.buildRegionPolygons(group.regions, vo=vo)

		#TODO: we could also just put some points out near infinity in each axis,
		# but that makes it more complicated to work with group indices


	def buildRegionPolygons(self, regions:list[arrT], vo:Voronoi):
		polygons = []
		for i, region in enumerate(regions):
			seedPos = vo.points[region.INDEX]
			vertexPoints = []
			for vtx in region:
				if vtx == -1:
					nearPos, nearSegment = self.fixInfiniteVertices(
						region, seedPos
					)
				else:
					nearPos = vo.vertices[vtx]
				vertexPoints.append(fromArr(nearPos, QtCore.QPointF))
			polygons.append(QtGui.QPolygonF.fromList(vertexPoints))
		return polygons

	def paint(self, painter, option, widget=...):
		painter.setBrush(QtGui.QColor.fromRgbF(1.0, 0., 0., 0.5))
		painter.setPen(QtGui.QColor.fromRgbF(0.5, 0.5, 1.0, 1.0))
		for i in self.polygons:
			painter.drawPolygon(i)


if __name__ == '__main__':

	col = QtGui.QColor(1, 1, 1)
	colB = QtGui.QColor.fromRgbF((1.0, 1.0, 1.0))

