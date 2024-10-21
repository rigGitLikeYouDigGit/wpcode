
from __future__ import annotations
import typing as T

"""
- camera position and view transform
- visibility filtering
"""
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
	from .scene import WpCanvasScene
	from .element import WpCanvasItem

class WpCanvasMiniMap(QtWidgets.QWidget):
	"""give a houdini-style overview of where the viewport is, in relation
	to the rest of the scene -
	only basic rectangles for now"""

	if T.TYPE_CHECKING:
		def parent(self)->WpCanvasView: pass

	minimapDragged = QtCore.Signal(dict)

	def __init__(self, parent: WpCanvasView):
		super().__init__(parent=parent)
		self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
		self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
		self.ks = KeyState()

	def scene(self)->WpCanvasScene:
		return self.parent().scene()

	def _onViewCameraChanged(self, camData:dict):
		"""fired whenever the view camera moves, updates drawing of
		minimap region"""
		self.repaint()

	def mousePressEvent(self, event):
		self.ks.mousePressed(event)

	def mouseReleaseEvent(self, event):
		self.ks.mouseReleased(event)

	def mouseMoveEvent(self, event):
		"""check if we need to move the view camera around - this will
		also update the drawing here"""
		self.ks.mouseMoved(event)
		#log("map mouse moved")
		#log("lastPressed", self.ks.lastPressed)
		if self.ks.LMB in self.ks.lastPressed:
			#log("drag minimap")
			self.minimapDragged.emit({
				"delta" : self.ks.mouseDelta(forKey=self.ks.LMB)
			})

	def paintEvent(self, event):
		"""draw the minimap rectangle -
		this probably needs more complex treatment if we ever allow wrapping,
		infinite scene sizes, etc"""
		#log("")
		painter = QtGui.QPainter(self)
		# draw the minimap rounded, looks nicer
		path = QtGui.QPainterPath()
		path.addRoundedRect(QtCore.QRectF(self.rect()), 5, 5)
		# draw the transparent background for minimap, representing whole scene
		painter.setBrush(QtGui.QBrush(
			QtGui.QColor.fromRgbF(0.3, 0.3, 0.3, 0.3)
		))
		#painter.fillRect(self.rect(), painter.brush())
		painter.fillPath(path, painter.brush())
		# draw the white outline for where the view actually is

		minimapArr = uilib.qRectToArr(self.rect(), originSize=True)
		#log("minimapArr", minimapArr)

		sceneRect = self.scene().itemsBoundingRect() # global space
		sceneRect = self.scene().sceneRect() # global space
		#log("sceneRect", sceneRect)
		sceneArr = uilib.qRectToArr(sceneRect, originSize=False)
		#log("sceneArr", sceneArr)

		#viewMappedRect = self.parent().viewportTransform().mapRect(QtCore.QRect())
		viewMappedRect = self.parent().mapToScene(self.parent().rect()).boundingRect()
		#log("viewRect", viewMappedRect)
		viewMappedArr = uilib.qRectToArr(viewMappedRect, originSize=False)
		#log("viewArr", viewMappedArr)

		scaledArr = viewMappedArr / sceneArr[1] * minimapArr[1]
		#log("scaledArr", scaledArr)

		toDrawRect = QtCore.QRect()
		toDrawRect.setCoords(*scaledArr.ravel())
		toDrawRect = toDrawRect.intersected(self.rect().marginsRemoved(
			QtCore.QMargins(2, 2, 2, 2)
		))
		# toDrawRect.setTopLeft(toDrawRect.topLeft() + QtCore.QPoint(2, 2))
		# toDrawRect.setBottomRight(toDrawRect.bottomRight() - QtCore.QPoint(2, 2))
		#log("toDraw", toDrawRect)
		#
		# viewRatioSize = QtCore.QSizeF(viewMappedRect.size().width()) / sceneRect.size().width()
		# viewMappedRect.setSize(viewMappedRect.size() * viewRatioSize) # scaled down
		# minimapRect = self.rect() * viewMappedRect
		painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(1.0, 1.0, 1.0, 0.5)))
		painter.drawRoundRect(toDrawRect, 2, 2)




class WpCanvasView(QtWidgets.QGraphicsView):
	"""add some conveniences to serialise camera positions
	surely selected items should be per-viewport? not per-scene?
	well per-scene is how qt handles it and for now we're keeping it

	TODO: momentum?
	"""

	if T.TYPE_CHECKING:
		def scene(self)->WpCanvasScene: pass

	cameraChanged = QtCore.Signal(dict)

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

		# minimap
		self.minimap = WpCanvasMiniMap(parent=self)
		self.minimap.minimapDragged.connect(self._onMiniMapDragged)
		self.cameraChanged.connect(self.minimap._onViewCameraChanged)


		self.moveCamera([0, 0], relative=False)

	def selectionMode(self)->str:
		#TODO: FIX
		return self.data["selectionMode"]
	def setSelectionMode(self, s:str):
		assert s in ("rect", "lasso")
		self.data["selectionMode"] = s

	def resizeEvent(self, event):
		super().resizeEvent(event)

		# want the map to start at (0.9, 0.9) normalised
		widthStep = self.width() / 10.0
		heightStep = self.height() / 10.0
		self.minimap.setGeometry(widthStep * 9, heightStep * 9,
		                         widthStep * 0.9, heightStep * 0.9)

	def moveCamera(self, pos:T.Sequence[float], relative=True, delay=None):
		"""overall method to move camera by relative or absolute scene coords"""

		arr = uilib.qTransformToArr(self.viewportTransform() )
		if isinstance(pos, (QtCore.QPoint, QtCore.QPointF)):
			pos = np.array(pos.toTuple())
		#log("arr", arr)
		#log(self.viewportTransform(), tuple(self.viewportTransform()))
		thisPos = (arr[0, 2], arr[1, 2])
		if not relative:

			pos = np.array(pos) - thisPos
		self.translate(*pos)

		self.cameraChanged.emit({"old" : thisPos,
		                         "new" : pos})


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

	def _onMiniMapDragged(self, data):
		self.moveCamera(data["delta"], relative=True)




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





