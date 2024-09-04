
from __future__ import annotations
import typing as T

"""
core scene for canvas
usually we won't store much state in canvas itself

"""

from PySide2 import QtWidgets, QtCore, QtGui

class WpCanvasScene(QtWidgets.QGraphicsScene):
	"""scene for a single canvas"""

	def __init__(self, parent=None):
		super().__init__(parent=parent)

		self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 255)))
		self.setSceneRect(0, 0, 1000, 1000)

