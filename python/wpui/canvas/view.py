
from __future__ import annotations

import types
import typing as T

"""
- camera position and view transform
- visibility filtering
"""
import numpy as np
from collections import namedtuple
from dataclasses import dataclass

from PySide2 import QtWidgets, QtCore, QtGui
#from param import rx

from wplib import log, sequence
from wptree import Tree
from wpdex import WpDexProxy
from wplib.serial import Serialisable

from wpui.keystate import KeyState
from wpui import lib as uilib, constant as uiconstant

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
				"delta" : - self.ks.mouseDelta(forKey=self.ks.LMB)
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


class ViewEventFilter(QtCore.QObject):

	def __init__(self, parent=None):
		super().__init__(parent)
		self._processingTab = False

	def eventFilter(self, watched:WpCanvasView, event):
		"""
		I was previously sending a keyPressEvent for the tab, on both the key-down event
		AND the key-up event
		I am unsure why this has only just started affecting the program, it somehow worked before
		you seemed unhappy - I present you a fool

		TODO: later we might have our own way to focus out/through a node view,
			as well as the directional focus idea
			either way, for now I think we entirely remove automated tab focus from play in view, it's just annoying

		Pressing tab first sends a ShortcutOverride keyEvent

		because of the weird parentless node widgets, and the order of events going from view->scene->graphicsItem->graphicsProxyWidget->grandchild widget->parent widget->grandparentWidget,
		- only follows the right order after passing through the proxy widget
		so the main view and a random node widget can get focus at the same time.
		honestly not one clue how to live with this

		"""
		#log("eventFilter", event.type(), event,)

		if isinstance(event, QtGui.QFocusEvent):
			eventTypeNameMap = {}
			for k, v in QtCore.QEvent.__dict__.items():
				try:
					eventTypeNameMap[v] = k
				except: continue
			log("eventFilter FOCUS event",eventTypeNameMap[event.type()], event.reason())
			eventReasonNameMap = {}
			for k, v in QtCore.Qt.FocusReason.__dict__.items():
				try:
					eventReasonNameMap[v] = k
				except: continue
			if event.type() in (
				#event.FocusOut,
				event.FocusAboutToChange,
			):

				watched.releaseKeyboard()
				log("released keyboard")
			if event.reason() in (
					QtCore.Qt.FocusReason.TabFocusReason,
					QtCore.Qt.FocusReason.BacktabFocusReason,
					QtCore.Qt.FocusReason.OtherFocusReason,

			): # get outta here
				#log("blocking focus event")
				#event.ignore()
				#self.
				return False

		### first line is CORRECT, since we only want to trigger on key down
		# but without also checking for focus events, this still triggers the focus system
		# to pass back the focus to whatever previous widget, EVEN IF THE KEY EVENT IS FILTERED.
		# the tab focus system seems super deeply embedded in Qt
		if isinstance(event, QtGui.QKeyEvent):
			log(event.key())
			if event.type() == QtGui.QKeyEvent.ShortcutOverride:
				log("blocking shortcut")
				##### why do neither of these options do anything, the doc says you accept the shortcut event to disable it
				event.accept()
				#event.ignore()
				return True
		if isinstance(event, QtGui.QKeyEvent) and event.type() == QtCore.QEvent.KeyPress:

		#if isinstance(event, QtGui.QKeyEvent):
			log("eventFilter key event", watched, event.key())
			log("focused widget", watched.focusWidget(), watched.hasFocus())
			if event.key() in (QtCore.Qt.Key_Tab, QtCore.Qt.Key_Backtab):
				#log("no tab for you")
				if self._processingTab: return True # prevent feedback
				if not watched.hasFocus(): return True
				#if not watched.keyboardGrabber(): return True
				self._processingTab = True
				newEvent = QtGui.QKeyEvent(event.type(), event.key(), event.modifiers(),
					                event.nativeScanCode(), event.nativeVirtualKey(),
					                event.nativeModifiers(), event.text(),
					                event.isAutoRepeat(), event.count())
				watched.keyPressEvent(newEvent)
				newEvent.accept()
				self._processingTab = False
				return True
		if isinstance(event, QtGui.QHoverEvent):
			"""update the ks mouse history """
			watched.ks.mouseMoved(event)

		return False
		#return super().eventFilter(watched, event)

class WpCanvasView(QtWidgets.QGraphicsView):
	"""add some conveniences to serialise camera positions
	surely selected items should be per-viewport? not per-scene?
	well per-scene is how qt handles it and for now we're keeping it

	TODO: camera momentum?

	TODO: context menu / radial menu setup

	TODO: E_V_E_M_T_S
		i forgot how fun this is - events pass through this view before they hit
		embedded widgets, so we need a way of checking if a graphics widget is
		under the cursor to receive focus, if it's already got focus, etc
	"""

	### does absolutely nothing :D
	# def focusNextChild(self):
	# 	return True
	# def focusPreviousChild(self):
	# 	return False

	# def nextInFocusChain(self):
	# 	return self
	# def previousInFocusChain(self):
	# 	return self

	if T.TYPE_CHECKING:
		def scene(self)->WpCanvasScene: pass

	@dataclass
	class KeySlot:
		"""easier declaration of hotkey-triggered events"""
		fn : T.Callable[[QtGui.QKeyEvent, WpCanvasView], (QtWidgets.QWidget, T.Any)]
		keys : tuple[QtCore.Qt.Key] = ()
		closeWidgetOnFocusOut : bool = True

	cameraChanged = QtCore.Signal(dict)

	# def focusNextPrevChild(self, next:bool):
	# 	self.setFocus()
	# 	return True

	def __init__(self,scene:WpCanvasScene, parent=None,
	             ):
		super().__init__(parent)
		self.setScene(scene)

		# self.setTabOrder(self, self)
		# self.parent().setTabOrder(self, self)

		#self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

		self.ks = KeyState()
		self.filter = ViewEventFilter(parent=self)
		self.installEventFilter(self.filter)
		#self.setMouseTracking(True)


		self.setTransformationAnchor(self.NoAnchor) # ????
		self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
		self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

		# self.setRenderHints(QtGui.QPainter.Antialiasing# | QtGui.QPainter.HighQualityAntialiasing
		#                     )

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

		# set up functions to fire when specific keys are pressed
		# if these functions return a widget, widget will be shown and
		# given modal focus
		self.keySlotMap : dict[tuple[QtCore.Qt.Key],
			WpCanvasView.KeySlot] = {}

		# set init camera pos
		self.moveCamera([0, 0], relative=False)
		self.setFocusPolicy(QtCore.Qt.StrongFocus)

	# region focus

	# def focusNextPrevChild(self, next):
	# 	"""this holds the focus on this widget, prevents Tab from moving it
	# 	FOR NOW this is ok, inkeeping with Maya, Houdini node editor conventions
	#
	# 	HOWEVER, later allow separate mode for full Vim key master focus switching -
	# 		maybe that needs some extra treatment
	# 	"""
	# 	return True
	# def focusNextChild(self):
	# 	return True
	# def nextInFocusChain(self):
	# 	return self
	#
	# def focusOutEvent(self, event:QtGui.QFocusEvent):
	# 	log("focusOut event", event.reason() in (QtCore.Qt.FocusReason.TabFocusReason,
	# 	                                         QtCore.Qt.FocusReason.BacktabFocusReason))
	# 	self.clearFocus()
	# 	return True

	# endregion

	def addKeyPressSlot(self,
	                    slot: (T.Callable[[WpCanvasView], (QtWidgets.QWidget, T.Any)],
	                           WpCanvasView.KeySlot),
	                    keys:tuple[QtCore.Qt.Key]=(),

	                    ):
		"""set a function to fire when the set keys are pressed
		TODO: could probably spin this out into some kind of component -
		"""
		if not isinstance(slot, WpCanvasView.KeySlot):
			slot = WpCanvasView.KeySlot(slot, keys=keys)
		self.keySlotMap[tuple(slot.keys)] = slot

	def _getMousePosForObjCreation(self)->QtCore.QPoint:
		log("mousePos", self.ks.mousePositions[0], self.ks.mousePositions[0] == QtCore.QPoint(0, 0))
		if self.ks.mousePositions[0] == QtCore.QPoint(0, 0):
			return self.rect().center()
		return self.ks.mousePositions[0]


	def checkFireKeySlots(self, event:QtGui.QKeyEvent):
		"""check if any key functions should be fired -
		if so, activate functions, then show the widget if returned
		and give it focus

		Q : Why not just use QT hotkey actions?
		A : they mess up the control flow of the whole program, even if you block events
			the hotkeys can "leak" up if you declare them higher
		"""
		result = False # if true, a slot was fired
		# this is some of the dumbest code I've ever written
		for keys, slot in self.keySlotMap.items():

			matches = True
			for i in keys:
				if i in self.ks.keyPressedMap.keys():
					if not self.ks.keyPressedMap[i]:
						matches = False
						break
				else:
					if i != event.key():
						matches = False
						break
			if not matches: continue
			# code equivalent of shovelling mud

			result = slot.fn(event, self)
			if isinstance(result, QtWidgets.QWidget): # show the returned widget
				# unsure if we should enforce it being a parent of this widget
				result.setEnabled(True)
				result.show()
				result.setFocus()
				# move it to the last shown mouse position
				pos = self._getMousePosForObjCreation()
				pos = self.mapTo(result.parent(), pos)
				result.move(pos)
				result.setFocus()

				# if we say to close after losing focus, set up that as a patch
				if slot.closeWidgetOnFocusOut:
					def _patchFocusEvent(focusEvent:QtGui.QFocusEvent):
						log("run slot focus out event")
						type(result).focusOutEvent(result, focusEvent)
						result.hide(); result.setEnabled(False)
						self.setFocus()
					result.focusOutEvent = _patchFocusEvent


	def keyPressEvent(self, event):
		"""
		TODO: for some reason tab events trigger this 3 times at once -
			fix this sometime, for now push through it
		"""
		#log("key press", event.key(), uiconstant.keyDict[event.key()])

		self.ks.keyPressed(event)
		result = self.checkFireKeySlots(event)
		if result:
			return True
		super().keyPressEvent(event)


	def keyReleaseEvent(self, event):
		self.ks.keyReleased(event)
		super().keyReleaseEvent(event)


	def selection(self)->list[QtWidgets.QGraphicsItem]:
		"""seems weird that the view can't access this natively"""
		return self.scene().selectedItems()
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
				#return

		elif event.button() == self.ks.rmbKey:
			"""begin context menu -
			defer down to scene items to show final menu,
			here only supply actions to take based on wider state"""
			#log("view context menu")
		super().mousePressEvent(event)
		#log("end selection", self.scene().selectedItems())

	def _onMiniMapDragged(self, data):
		self.moveCamera(data["delta"], relative=True)


	def mouseReleaseEvent(self, event):
		self.ks.mouseReleased(event)
		super().mouseReleaseEvent(event)

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
		super().mouseMoveEvent(event)



	# def dragMoveEvent(self, event:QtGui.QDragMoveEvent):
	#
	# 	self.ks.mouseMoved(event)





