
from __future__ import annotations

import math
import typing as T

import numpy as np

from PySide2 import QtWidgets, QtCore, QtGui
#from param import rx

from wplib import log, sequence
from wplib.object import Signal
from wptree import Tree
from wpdex import WpDexProxy
from wplib.serial import Serialisable

from wpui.keystate import KeyState
from wpui import lib as uilib, treemenu

"""
bases for items in canvas, 
functions for selecting, dragging, etc
"""




base = object
if T.TYPE_CHECKING:
	base = QtWidgets.QGraphicsItem
	from .view import WpCanvasView
	from .scene import WpCanvasScene

class WpCanvasElement(base,
                   # Serialisable
                   ):
	#elementChanged = QtCore.Signal(QtWidgets.QGraphicsItem, int, object)
	"""
	base class for moveable, selectable items in canvas
	for ease, we don't disagree with the normal selection system


	how would we set the positions of qt items reactively?

	TODO: allow in world-space, local space, or view space for ui elements
		for view, allow pinning to edges, etc, setting offset from edges, corners
	"""

	# def __hash__(self):
	# 	return id(self)

	def __init__(self,
	             #scene:WpCanvasScene=None,
	             obj=None):
		"""use "obj" to denote a drawn delegate for a "real" python
		object"""
		self.obj = obj
		# can't use proper QT signals since graphicsItems aren't QObjects
		self.elementChanged = Signal("elementChanged")

		# disable these if things get as slow as the docs warn - we rely on
		# efficient filtering at scene level
		self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
		self.setFlag(QtWidgets.QGraphicsItem.ItemSendsScenePositionChanges)
		pass

	def itemChange(self,
	               change:QtWidgets.QGraphicsItem.GraphicsItemChange,
	               value):
		self.elementChanged.emit(self, change, value)
		return super().itemChange(change, value)

	def getContextMenuTree(self,
	                       event:QtGui.QMouseEvent=None,
	                       )->Tree:
		"""
		get element-specific context menu tree

		"""
		# t = Tree("root")
		# t["action"] = lambda : print("eyyy")
		# return t
		#pass

	def mousePressEvent(self, event):
		"""EVENT FLOW goes
		view -> scene -> graphicsItem -
		relies on super() being called on each one"""
		#log("element mouse press")
		if event.button() == QtCore.Qt.RightButton:
			log("element right click")
			tree = self.getContextMenuTree()
			if not tree:
				return
			menu = treemenu.buildMenuFromTree(tree)
			fn = lambda *a, **k : menu.exec_(event.screenPos())
			self.clearFocus()
			fn()
			# TODO:
			#   weird behaviour where right-clicking first opens menu, then
			#   right-clicking again anywhere in the scene reopens it again -
			#   events are still getting fed here.
			#   not a big issue but fix it

			#### the below did not work, to trigger the menu after this event is
			#  done processing
			# showTimer = QtCore.QTimer()
			# showTimer._event = event
			# showTimer._menu = menu
			# self._tempTimer = showTimer
			# #showTimer.timeout.connect()
			# showTimer.singleShot(10, fn)
			event.accept()
		return super().mousePressEvent(event)
		pass


class WpCanvasProxyWidget(QtWidgets.QGraphicsProxyWidget):
	"""subclass to set up signals for focus gained and lost,
	tearing my hair out on how to do it otherwise"""

	# signatures are (top proxy widget, widget that gained/lost focus)
	focusGained = QtCore.Signal(object, object)
	focusLost = QtCore.Signal(object, object)

	def focusInEvent(self, event):
		#log("proxy focus in")
		# no good way to tell where event came from right now,
		# might remove the second arg
		self.focusGained.emit(self, self)
		return super().focusInEvent(event)

	def focusOutEvent(self, event):
		#log("proxy focus out")
		self.focusLost.emit(self, self)
		return super().focusOutEvent(event)
