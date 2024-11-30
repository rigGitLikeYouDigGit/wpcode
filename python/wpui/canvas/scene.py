

from __future__ import annotations

import math
import typing as T
from collections import defaultdict
import numpy as np

from PySide2 import QtWidgets, QtCore, QtGui
#from param import rx
import networkx as nx
 
from wplib import log, sequence, nxlib

from wptree import Tree
from wpdex import WpDexProxy
from wplib.serial import Serialisable

from wpui.keystate import KeyState
from wpui import lib as uilib
from wplib.maths import shape, arr, fromArr, arrT

from wpui.canvas.element import WpCanvasElement

from wpui.canvas.connection import ConnectionPoint, ConnectionsPainter, ConnectionGroupDelegate

if T.TYPE_CHECKING:
	from .element import WpCanvasElement
	from .connection import ConnectionPoint, ConnectionGroupDelegate

def drawGridOverRect(rect:QtCore.QRect,
                     painter:QtGui.QPainter,
                     cellSize=(10, 10)
                     ):
	"""basic qt brush cross fill is trash, so we need to roll our own -
	get the integer coordinates enclosing the given view, and draw pen
	lines between them -
	assume the rect has already been transformed by whatever mat you
	want"""
	cellSize = [int(i) for i in cellSize]
	points = np.array((rect.topLeft().toTuple(), rect.bottomRight().toTuple()),
	                  dtype=int)#.transpose()
	#log("points", points)
	# intPoints = np.array(map(int, points))
	# log("intpoints", intPoints)
	for x in range(points[0][0], points[1][0]): # verticals
		if x % cellSize[0]: continue
		painter.drawLine(QtCore.QLine(x, points[0][1],
		                              x, points[1][1]))
	for y in range(points[0][1], points[1][1]): # horizontals
		if y % cellSize[1] : continue
		painter.drawLine(QtCore.QLine(points[0][0], y,
		                              points[1][0], y))


def iterQGraphicsItems(rootItem, includeRoot=True):
	result = [rootItem] if includeRoot else []
	toIter = [rootItem]
	while toIter:
		item : QtWidgets.QGraphicsItem = toIter.pop(-1)
		result.extend(item.childItems())
		toIter.extend(item.childItems())
	return result

class WpCanvasScene(QtWidgets.QGraphicsScene):
	"""scene for a single canvas

	- setting all the single-use property things to be dynamic/reactive would be rad
	- setting the more complicated event callbacks to be reactive may not be useful

	- framework for items connected to each other ( and specialised for path connections
		between nodes )

	TODO: ACTIONS
		for changing selection state
		moving objects
		doing anything really - THAT is another advantage of having everything as data
	"""

	def __init__(self, parent=None):
		super().__init__(parent=parent)

		self.objDelegateMap : dict[T.Any, set[QtWidgets.QGraphicsItem]] = defaultdict(set)
		#self.delegateObjMap : dict[int, T.Any] = {} #delegates hold their own object reference

		# graph for tracking general relationships between objects
		self.relationGraph = nx.MultiGraph()

		# template dashed path to preview dragged connections between ports
		self._dragPath = QtWidgets.QGraphicsPathItem()
		pen = QtGui.QPen(QtCore.Qt.yellow, 1.0,
		                 s=QtCore.Qt.PenStyle.DashDotLine)
		self._dragPath.setPen(pen)
		self.addItem(self._dragPath)
		self._dragPath.hide()
		self._dragSource : ConnectionPoint = None # if not none, dragging in progress
		self._candidateConnectPoint : ConnectionPoint = None

		self._buildBackground()

	def _buildBackground(self):
		self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 255)))
		self.setSceneRect(0, 0, 1000, 1000)
		# create brushes to use - base one for black background,
		# 2 more for the grid (eventually have this as an option for snapping
		self.baseBrush = QtGui.QBrush(QtCore.Qt.black)

		self.gridScale = 20
		self.gridPen = QtGui.QPen(QtGui.QColor.fromRgbF(0, 0, 0.5))
		self.gridPen.setCosmetic(True)
		self.coarseScale = 5.0 * self.gridScale
		self.coarsePen = QtGui.QPen(QtGui.QColor.fromRgbF(0.3, 0.4, 1.0))
		self.coarsePen.setCosmetic(True)
		self.coarsePen.setStyle(QtCore.Qt.DashLine)

	# def addConnectionPointToGroup(self,
	#                             ptA:ConnectionPoint,
	#                             connectionGroupDelegate:ConnectionGroupDelegate=None
	#                             ):
	# 	"""TODO: try and inject a user way to get a new connection for
	# 			 a given point
	# 	"""
	# 	if connectionGroupDelegate not in self.relationGraph:
	# 	self.relationGraph.add_edge(ptA, ptB, key="connectEnd")

	def _connectionDelegateForPoints(self, *points:ConnectionPoint)->ConnectionGroupDelegate:
		"""return a new connectionDelegate object initialised on the
		given points"""
		raise NotImplementedError(self, points)

	def connectPoints(self,
	                  ptA:ConnectionPoint,
	                  ptB:ConnectionPoint,
	                  connectionGroupDelegate:ConnectionGroupDelegate=None
	                  ):
		"""
		called when a valid connection is created in UI -
		OVERRIDE
		for use-specific logic

		TODO: hoooooow do we create the connection delegate?
				how does that gel with updating overall graph to model

		TODO: use proper constant names here, not just raw strings
		"""
		if connectionGroupDelegate is None: # no existing group selected
			connectionGroupDelegate = self._connectionDelegateForPoints(ptA, ptB)
		self.addItem(connectionGroupDelegate)
		self.relationGraph.add_edge(ptA, ptB, key="connectPoint")
		self.relationGraph.add_edge(ptA, connectionGroupDelegate, key="connectGroup")
		self.relationGraph.add_edge(ptB, connectionGroupDelegate, key="connectGroup")
		return connectionGroupDelegate

	def connectionPoints(self):
		return (i for i in self.relationGraph if isinstance(i, ConnectionPoint))

	def connectedItems(self,
	                   seedItem,
	                   key="connectPoint")->T.Iterable[WpCanvasElement]:
		return nxlib.multiGraphAdjacentNodesForKey(
			self.relationGraph, seedItem, key=key
		)

	def addItem(self, item):
		#from .element import WpCanvasElement
		log("scene addItem", item, type(item))
		super().addItem(item)

		# check through all children in case we need to connect up elements known "globally" to scene
		for i in iterQGraphicsItems(item, includeRoot=True):
			if not isinstance(i, WpCanvasElement):
				continue # only process proper canvasElements
			self.relationGraph.add_node(i)
			i.elementChanged.connect(self._onItemChanged)
			self.objDelegateMap[i.obj].add(i)
			log("after addItem", item, item.obj)
		return item

	def removeItem(self, item):
		"""
		we check if item was a connectionPoint, and if so, remove
		any connectionGroup items it was connected to.
		might not be the right place for it
		"""
		#from .element import WpCanvasElement
		#if getattr(item, "elementChanged", None):
		for child in iterQGraphicsItems(item, includeRoot=True):
			if not isinstance(child, WpCanvasElement):
				continue
			self.relationGraph.remove_node(child)
			if isinstance(child, WpCanvasElement):
				#item.itemChange.connect(self._onItemChanged)
				if child.obj in self.objDelegateMap:
					self.objDelegateMap[child.obj].remove(child)

			if isinstance(child, ConnectionPoint):
				# remove all individual groups / lines using this point
				if child in self.relationGraph: # in case this gets fired twice
					for i in tuple(nxlib.multiGraphAdjacentNodesForKey(
						self.relationGraph, child, key="connectGroup"
					)):
						self.relationGraph.remove_node(i)
						self.removeItem(i)

		super().removeItem(item)

	def _onItemChanged(self, item:WpCanvasElement,
	                   change:QtWidgets.QGraphicsItem.GraphicsItemChange,
	                   value:T.Any):
		"""process each change of each item -
		look at relation graph to see if connected items need updating -

		but can't we achieve this with QGraphicsPathItem?
		why are we even doing this
		BECAUSE I don't see another good way to set up bidirectional
			dependency between items; even if we look at connection points
			within a pathItem's path() method, that method still has
			to fire when the points move
		 """
		#log("scene _onItemChanged", item, change, value)

		if change in (QtWidgets.QGraphicsItem.ItemPositionChange,
						QtWidgets.QGraphicsItem.ItemPositionHasChanged,
						QtWidgets.QGraphicsItem.ItemScenePositionHasChanged):
			if isinstance(item, ConnectionPoint):
				for group in nxlib.multiGraphAdjacentNodesForKey(
					self.relationGraph, item, key="connectGroup"
				): #type:QtWidgets.QGraphicsItem
					group.update()

	def isDragging(self):
		return self._dragSource is not None
	def isConnectionPointAvailable(self, srcPt:ConnectionPoint, dstPt:ConnectionPoint):
		if dstPt.acceptsIncomingConnection(fromObj=srcPt):
			if srcPt.acceptsOutgoingConnection(toObj=dstPt):
				return True
		return False
	def onConnectionDragBegin(self, fromObj:ConnectionPoint):
		"""start drawing path (s)

		TODO: have template edges inherit colours of plugs they come from
		"""
		#log("scene connection begin")
		self._dragSource = fromObj
		self._dragPath.show()

		# filter over all connection points in scene, check if they can
		# receive this drag connection
		for point in self.connectionPoints():
			if not point.canAcceptDragConnections():
				continue
			# check both sides - here we don't care about input/output in graph, this is
			# only the sequence they're connected by the user
			state = self.isConnectionPointAvailable(fromObj, point)
			point.setDragAvailable(state)

	DRAG_SNAP_MAX_DIST = 50

	def mouseMoveEvent(self, event):
		#log("scene mouse move event", self.isDragging())
		if self.isDragging():
			# update path
			point, vec = self._dragSource.connectionPoint(None)
			point = QtCore.QPointF(*point)
			point = self._dragSource.mapToScene(point)
			mousePos = event.scenePos()

			# check any nearby available connectionPoint objects -
			# if any within range, snap to the nearest one
			rect = QtCore.QRectF(0, 0, self.DRAG_SNAP_MAX_DIST,
			                     self.DRAG_SNAP_MAX_DIST)
			rect.moveCenter(mousePos)
			nearConnectPoint : ConnectionPoint = None
			nearConnectPos : QtCore.QPointF = None
			dist = 1000000
			for i in self.connectionPoints():
				if not i.isDragAvailable():
					continue
				testPos, testVec = i.connectionPoint(None)
				testPoint = QtCore.QPointF(*testPos)
				testDist = (testPoint - mousePos).manhattanLength()
				if testDist < dist:
					dist = testDist
					nearConnectPoint = i
					nearConnectPos = testPoint

			# if point is within range, snap the end of the path to it, and
			# set it as the candidate point to connect on mouse up
			if dist <= pow(self.DRAG_SNAP_MAX_DIST, 2): #
				self._candidateConnectPoint = nearConnectPoint
				mousePos = nearConnectPos

			path = QtGui.QPainterPath(point)
			path.lineTo(mousePos)
			self._dragPath.setPath(path)
		return super().mouseMoveEvent(event)

	def mouseReleaseEvent(self, event):
		"""if a candidate connection point has been
		set by a scene connectionPoint object, make the connection?

		remove source point, hide path, come out of dragging state"""

		# if a candidate connection point has been selected,
		# create the connection between those two points
		if self._candidateConnectPoint is not None:
			self.connectPoints(self._dragSource, self._candidateConnectPoint,
			                   connectionGroupDelegate=None
			                   )

		self._dragPath.hide()
		self._dragSource = None
		# CONSCIOUSLY NOT CLEARING PATH SHAPE HERE
		# BECAUSE THE STALE TEMPLATE PATH LOOKS COOL

		return super().mouseReleaseEvent(event)

	def onConnectionDragMove(self, ):
		pass

	def mousePressEvent(self, event):
		if event.button() == QtCore.Qt.RightButton:
			#log("scene context menu")
			#return
			pass
		return super().mousePressEvent(event)

	# # TODO: get this working for large scale scenes, processing multiple nodes at once
	# def _onItemsChanged(self,
	#                     changeItemMap:dict[
	#                                   QtWidgets.QGraphicsItem.GraphicsItemChange :
	#                                   T.Sequence[WpCanvasElement]]):
	# 	"""signal from scene when one or more items change as part of single operation
	# 	passed map of
	# 	{ itemsMoved : (all items that moved) } etc
	#
	# 	TODO: order events by priority? we would want removed to come after everything
	# 	"""

	"""single real object may have multiple delegates, be drawn in multiple
	places at once"""

	# def setDelegatesForItem(self, obj, delegates:T.Sequence[WpCanvasElement]):
	# 	delegates = sequence.toSeq(delegates)
	# 	for i in delegates:
	# 		self.delegateObjMap[hash(i)] = obj
	# 	self.objDelegateMap[obj] = tuple(delegates)

	def delegatesForObj(self, obj)->set[WpCanvasElement]:
		return set(self.objDelegateMap.get(obj, ()))



	def objFromDelegate(self, item:(QtWidgets.QGraphicsItem,
	                                WpCanvasElement)):
		"""OVERRIDE if for some reason we need to use extra logic,
		but can't use the delegate set dict"""
		if not isinstance(item, WpCanvasElement):
			return None
		return item.obj

	def itemsDragged(self, items:list[QtWidgets.QGraphicsItem],
	                 delta:tuple[int, int]):
		#log("items dragged", delta, items)
		#delta = toArr(delta)
		for i in items:
			#item.setPos(i.pos())
			#continue
			# log("pointArr", np.array(i.pos().toTuple()))
			# log("deltaArr", np.array(delta))
			# log("added", np.array(i.pos().toTuple()) + np.array(delta))
			#finalPoint = np.array(i.pos().toTuple()) + np.array(delta, dtype=int)
			finalPoint = i.pos() + delta
			#log("finalPoint", finalPoint)
			self.trySetPos(i, pos=finalPoint)

	def trySetPos(self, item:QtWidgets.QGraphicsItem, pos:tuple[int, int]):
		"""check if there are any constraints, then set position"""
		#log("pos", pos, type(pos))
		item.setPos(pos)

	def select(self, items:list[QtWidgets.QGraphicsItem],
	           mode="add"):
		""" do we keep the same key behaviour as maya, or something slightly
		more kind
		WHY is toggle the default with shift

		mode can be either
		replace (nothing),
		add (control + shift),
		remove (control),
		toggle (shift)

		"""
		#log("select", mode, mode=="replace", items)
		if mode == "replace" :
			self.clearSelection()
			for i in items: i.setSelected(True)
			return
		for i in items:
			if mode == "toggle":
				i.setSelected(not i.isSelected())
			if mode == "add":
				i.setSelected(True)
			if mode == "remove" :
				i.setSelected(False)

	def centreItem(self, item:QtWidgets.QGraphicsItem):
		centrePos = self.sceneRect().center()
		item.setPos(centrePos)

	def drawBackground(self, painter:QtGui.QPainter, rect):
		""""""
		painter.save()
		painter.fillRect(rect, self.baseBrush)

		painter.setPen(self.gridPen)
		drawGridOverRect(rect, painter, cellSize=(self.gridScale, self.gridScale))

		painter.setPen(self.coarsePen)
		drawGridOverRect(rect, painter, cellSize=(self.coarseScale, self.coarseScale))
		painter.restore()

if __name__ == '__main__':

	class TestPoint:pass
	class TestEdge:pass
	g = nx.MultiGraph()
	ptA = TestPoint()
	ptB = TestPoint()
	g.add_node(ptA)
	g.add_node(ptB)
	e = TestEdge()
	g.add_edge(ptA, ptB, key="connectEnd")
	g.add_edge(ptA, e, key="connectLine")
	g.add_edge(ptB, e, key="connectLine")

	print(g.edges(ptA))
	print(g.edges(ptA, keys=True))
	print(*g.neighbors(ptA,))




	pass
