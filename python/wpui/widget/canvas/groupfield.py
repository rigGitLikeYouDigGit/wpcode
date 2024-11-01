from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

try:
	import numpy as np
	from scipy.spatial import ConvexHull
	SCIPY_ON = True
except ImportError:
	print("scipy is not available, switching to simple graph hull drawing")
	SCIPY_ON = False

from PySide2 import QtCore, QtWidgets, QtGui


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

class UserPolygon(QtWidgets.QGraphicsPolygonItem):
	"""let user drag out their own shape -
	highlight edges and vertices on mouse over"""
	
	def __init__(self, parent=None,
	             allowSelectEdges=True,
	             allowSelectPoints=True,
	             pointRadius=10):
		super().__init__(parent=parent)
		self.allowSelectEdges = allowSelectEdges
		self.allowSelectPoints = allowSelectPoints
		self.pointRadius = 10

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

	def nearestElements(self, pos,
	                   pointRad=10):
		"""sort all edges and points of this polygon by distance to the given pos,
		and return their distances
		(brute force here, since converting between numpy and qt is finnicky enough
		already)"""

	def mouseMoveEvent(self, event):
		"""highlight points and edges available for dragging"""
		if event.pos():pass

class GroupPolygon(QtWidgets.QGraphicsPolygonItem):
	"""draw convex field around items
	later maybe add system for rounding corners
	"""
	def __init__(self, parent=None):
		super(GroupPolygon, self).__init__(parent)
		self.itemsToContain :set[QtWidgets.QGraphicsItem] = set()


	def containedPoints(self, boundingBoxPadding=2)->list[QtCore.QPointF]:
		"""return bounding box corners of all items contained in field"""
		points = list()
		margins = QtCore.QMargins(
			boundingBoxPadding, boundingBoxPadding,
			boundingBoxPadding, boundingBoxPadding		)
		for item in self.itemsToContain:
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
