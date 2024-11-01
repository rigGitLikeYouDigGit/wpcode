
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
from wpui import lib as uilib

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

	def __hash__(self):
		return id(self)

	def __init__(self,
	             #scene:WpCanvasScene=None,
	             obj=None):
		"""use "obj" to denote a drawn delegate for a "real" python
		object"""
		self.obj = obj
		# can't use proper QT signals since graphicsItems aren't QObjects
		self.elementChanged = Signal("elementChanged")
		pass

	def itemChange(self, change, value):
		self.elementChanged.emit(self, change, value)