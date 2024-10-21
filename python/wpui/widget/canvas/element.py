
from __future__ import annotations

import math
import typing as T

import numpy as np

from PySide2 import QtWidgets, QtCore, QtGui
#from param import rx

from wplib import log, sequence
from wptree import Tree
from wpdex import WpDexProxy
from wplib.serial import Serialisable

from wpui.keystate import KeyState
from wpui import lib as uilib

"""
bases for items in canvas, 
functions for selecting, dragging, etc
"""


class ModelBased(Serialisable):
	"""defining objects that draw from an internal primitive
	data model
	TODO: - obviously move this
	 - don't bother rolling this back into tree etc yet, til
	    we know it's useful
	    """

	@classmethod
	def newModel(cls):
		raise NotImplementedError

	def __init__(self, model=None):
		self.model = model or self.newModel()

	# region serialisation -
	# literally just serialise and store the model
	def encode(self, encodeParams:dict=None) ->dict:
		"""a hierarchy of model objects will only be serialised once,
		at the root"""
		return self.model

	@classmethod
	def decode(cls, serialData:dict, decodeParams:dict=None) ->T.Any:
		return cls(serialData)




class CanvasField(Tree):
	pass


def newSceneModel():
	t = CanvasField("canvas")
	t["items"] = {} # should be a list of typed item datas


class MoveableGraphicsItem(QtWidgets.QGraphicsItem):
	"""test base class for things that can be moved and dragged"""


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

	# def usedRect(self)->QtCore.QRect:
	# 	"""return the area of scene comprising actual elements -
	# 	only look at top-level children, this won't be exact"""
	# 	rect = QtCore.QRect()
	# 	for i in self.items(): #type:QtWidgets.QGraphicsItem
	# 		rect = rect.united(i.sceneBoundingRect())
	# 	return rect
	# NEVERMIND we already have that in itemsBoundingRect()


	def drawBackground(self, painter:QtGui.QPainter, rect):
		""""""
		painter.save()
		painter.fillRect(rect, self.baseBrush)

		painter.setPen(self.gridPen)
		# gridTransform = QtGui.QTransform.fromScale(self.gridScale, self.gridScale)
		# gridRect = gridTransform.mapRect(rect)
		#drawGridOverRect(gridRect, painter,)
		drawGridOverRect(rect, painter, cellSize=(self.gridScale, self.gridScale))

		painter.setPen(self.coarsePen)
		# coarseTransform = QtGui.QTransform.fromScale(self.coarseScale, self.coarseScale)
		# coarseRect = coarseTransform.mapRect(rect)
		#drawGridOverRect(coarseRect, painter)
		drawGridOverRect(rect, painter, cellSize=(self.coarseScale, self.coarseScale))

		# #painter.setWorldTransform(self.fineGridBrush.transform())
		# painter.fillRect(rect, self.fineGridBrush)
		# #painter.setWorldTransform(self.coarseGridBrush.transform())
		# painter.setPen(QtCore.Qt.PenStyle.DotLine)
		# painter.fillRect(rect, self.coarseGridBrush)
		painter.restore()


class WpCanvasMiniMap(QtWidgets.QWidget):
	"""give a houdini-style overview of where the viewport is, in relation
	to the rest of the scene -
	only basic rectangles for now"""
	
	def __init__(self, parent:WpCanvasView):
		super().__init__(parent=parent)
		self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
		self.setWindowFlag(QtCore.Qt.FramelessWindowHint)

class WpCanvasView(QtWidgets.QGraphicsView):
	"""add some conveniences to serialise camera positions
	surely selected items should be per-viewport? not per-scene?
	well per-scene is how qt handles it and for now we're keeping it
	"""

	#scene : T.Callable[[], WpCanvasScene]
	if T.TYPE_CHECKING:
		def scene(self)->WpCanvasScene: pass

	def __init__(self, parent=None,
	             scene:WpCanvasScene=None):
		super().__init__(parent)
		assert scene
		self.setScene(scene)


		self.ks = KeyState()

		self.setTransformationAnchor(self.NoAnchor) # ????
		self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
		self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

		#TODO: rewrite this with the model stuff
		self.data = {
			"position" : [0, 0],
			"zoom" : 1.0,
			"selectionMode" : "rect" # or "lasso"
		}

		# selection tracking
		self._selPath : list[T.Sequence[int]] = []
		self._selGeo : (QtGui.QPainterPath, QtCore.QRect) = None

		self.moveCamera([0, 0], relative=False)

	def selectionMode(self)->str:
		#TODO: FIX
		return self.data["selectionMode"]
	def setSelectionMode(self, s:str):
		assert s in ("rect", "lasso")
		self.data["selectionMode"] = s

	def moveCamera(self, pos:T.Sequence[float], relative=True, delay=None):
		"""overall method to move camera by relative or absolute scene coords"""
		arr = uilib.qTransformToArr(self.viewportTransform() )
		#log("arr", arr)
		#log(self.viewportTransform(), tuple(self.viewportTransform()))
		thisPos = (arr[0, 2], arr[1, 2])
		if not relative:

			pos = np.array(pos) - thisPos
		self.translate(*pos)


	def mousePressEvent(self, event):
		self.ks.mousePressed(event)
		scenePressPos = self.mapToScene( event.pos())

		if event.button() == self.ks.lmbKey:
			items = self.scene().itemAt(scenePressPos, QtGui.QTransform())
			if items:
				items = sequence.toSeq(items)
				mode = "replace"
				if self.ks.SHIFT and self.ks.CTRL: mode = "add"
				elif self.ks.SHIFT: mode = "toggle"
				elif self.ks.CTRL : mode = "remove"

				self.scene().select(items, mode=mode)

		#log("end selection", self.scene().selectedItems())




	def mouseReleaseEvent(self, event):
		self.ks.mouseReleased(event)

	def _updateIncludedSelectionItems(self):
		"""check any """

	def _updateSelectionDrawing(self):
		if self.selectionMode() == "lasso": # draw a path and fill in included area
			path = QtGui.QPainterPath(self._selPath[0])
			for i in self._selPath[1:]:
				path.lineTo(i)
			# close back to start
			path.lineTo(self._selPath[0])
			self._selGeo = path
		else:
			rect = QtCore.QRect(self._selPath[0], self._selPath[1])


	def _updateSelectionPositions(self, event:QtGui.QMoveEvent):
		if self.selectionMode() == "lasso" :
			self._selPath.append(event.pos())
		else:
			self._selPath[0] = event.pos()
		self._updateSelectionGeo()
		self._updateIncludedSelectionItems()

	def _clearSelectionData(self):
		self._selPath = []
		self._selGeo = None

	def mouseMoveEvent(self, event):
		"""if alt + lmb is held, move the camera"""
		self.ks.mouseMoved(event)
		if self.ks.ALT and self.ks.MMB:
			self.moveCamera(self.ks.mouseDelta(), relative=True)
			return # moving the camera should override everything?

		if self.ks.LMB:
			if self.ks.SHIFT or self.ks.CTRL: # selection modifiers:
				self._updateSelectionPositions(event)
				return
			if self.scene().selectedItems():
				# we send the drag event to scene
				self.scene().itemsDragged(items=self.scene().selectedItems(),
				                          delta=self.ks.mouseDelta(forKey=self.ks.LMB))



	# def dragMoveEvent(self, event:QtGui.QDragMoveEvent):
	#
	# 	self.ks.mouseMoved(event)









class WpCanvasItem(QtWidgets.QGraphicsItem,
                    Serialisable):
	"""
	base class for moveable, selectable items in canvas
	for ease, we don't disagree with the normal selection system


	how would we set the positions of qt items reactively?

	TODO: allow in world-space, local space, or view space for ui elements
		for view, allow pinning to edges, etc, setting offset from edges, corners
	"""

	@classmethod
	def newDataModel(cls):
		return {"name" : "",
		        "position" : [0, 0]}

	if T.TYPE_CHECKING:
		#parent:
		pass

	def __init__(self,
	             parent:QtWidgets.QGraphicsItem=None,
	             data:WpDexProxy=None,
	             selectable=True,
	             movable=True,
	             ):
		super().__init__(parent=parent)
		#data.ref("pos").drive( self.setPos)
		self._selectable = selectable
		self._movable = movable

	def isSelectable(self)->bool:
		return self._selectable
	def setSelectable(self, state):
		self._selectable = state

	def isMoveable(self)->bool:
		return self._movable
	def setMovable(self, state):
		self._movable = state

	def paint(self, painter, option, widget=...):
		QtWidgets.QGraphicsRectItem.paint(self, painter, option, widget)


if __name__ == '__main__':

	app = QtWidgets.QApplication()

	scene = WpCanvasScene()
	w = WpCanvasView(parent=None, scene=scene,
	                 )
	# item = WpCanvasItem(parent=None,
	#                     )
	item = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0, 0, 100, 100))
	item.setBrush(QtGui.QBrush(QtCore.Qt.red))
	item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)
	scene.addItem(item)
	scene.centreItem(item)
	w.show()
	app.exec_()
