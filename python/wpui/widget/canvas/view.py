
from __future__ import annotations
import typing as T

"""
- camera position and view transform
- visibility filtering
"""

import numpy as np

from wplib.serial import Serialisable

from PySide2 import QtWidgets, QtCore, QtGui

if T.TYPE_CHECKING:
	from .scene import WpCanvasScene

class WpCanvasView(QtWidgets.QGraphicsView,
                 #Serialisable
                 ):
	"""canvas view for a single canvas

	todo:
	 - visibility filtering
	 - camera position and view transform
	 - momentum scrolling
	 - "mini-map" view
	 - selection by box, lasso, name
	"""

	def __init__(self, parent=None, canvas:WpCanvasScene=None):
		super().__init__(parent=parent)

		if canvas:
			self.setScene(canvas)


	def dream(self):
		"""if we could have a single checkbox pointing to momentum,
		linked directly"""





