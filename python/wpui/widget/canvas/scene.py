

from __future__ import annotations

import math
import typing as T
from collections import defaultdict
import numpy as np

from PySide2 import QtWidgets, QtCore, QtGui
#from param import rx

from wplib import log, sequence
from wptree import Tree
from wpdex import WpDexProxy
from wplib.serial import Serialisable

from wpui.keystate import KeyState
from wpui import lib as uilib

if T.TYPE_CHECKING:
	from .element import WpCanvasElement

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



def toArr(obj)->np.ndarray:
	"""we will eventually define adaptors for every random numeric type
	of every random library
	to get them into numpy"""
	if isinstance(obj, (QtCore.QPoint, QtCore.QPointF)):
		return np.array(obj.toTuple())


class WpCanvasScene(QtWidgets.QGraphicsScene):
	"""scene for a single canvas

	- setting all the single-use property things to be dynamic/reactive would be rad
	- setting the more complicated event callbacks to be reactive may not be useful

	TODO: ACTIONS
		for changing selection state
		moving objects
		doing anything really - THAT is another advantage of having everything as data
	"""

	def __init__(self, parent=None):
		super().__init__(parent=parent)

		self.objDelegateMap : dict[T.Any, set[QtWidgets.QGraphicsItem]] = defaultdict(set)
		#self.delegateObjMap : dict[int, T.Any] = {} #delegates hold their own object reference

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

	def addItem(self, item):
		from .element import WpCanvasElement
		super().addItem(item)
		log("scene addItem", item, type(item))
		#if not getattr(item, "elementChanged", None): return # only process proper canvasElements
		if not isinstance(item, WpCanvasElement): return # only process proper canvasElements
		item : WpCanvasElement
		item.elementChanged.connect(self._onItemChanged)
		self.objDelegateMap[item.obj].add(item)
		log("after addItem", item, item.obj)
		return item

	def removeItem(self, item):
		#if getattr(item, "elementChanged", None):
		if isinstance(item, WpCanvasElement):
			log("removeItem", item)
			item: WpCanvasElement
			#item.itemChange.connect(self._onItemChanged)
			if item.obj in self.objDelegateMap:
				self.objDelegateMap[item.obj].remove(item)
		super().removeItem(item)

	def _onItemChanged(self, item:WpCanvasElement,
	                   change:QtWidgets.QGraphicsItem.GraphicsItemChange,
	                   value:T.Any):
		"""process each change of each item"""


	# TODO: get this working for large scale scenes, processing multiple nodes at once
	def _onItemsChanged(self,
	                    changeItemMap:dict[
	                                  QtWidgets.QGraphicsItem.GraphicsItemChange :
	                                  T.Sequence[WpCanvasElement]]):
		"""signal from scene when one or more items change as part of single operation
		passed map of 
		{ itemsMoved : (all items that moved) } etc
		
		TODO: order events by priority? we would want removed to come after everything
		"""

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