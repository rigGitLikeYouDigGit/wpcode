
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui
#from param import rx

from wplib.serial import Serialisable

"""
bases for items in canvas, 
functions for selecting, dragging, etc
"""

class WpCanvasItem(QtWidgets.QGraphicsItem,
                    Serialisable):
	"""
	base class for moveable, selectable items in canvas
	for ease, we don't disagree with the normal selection system


	how would we set the positions of qt items reactively?

	TODO: allow in world-space, local space, or view space for ui elements
		for view, allow pinning to edges, etc, setting offset from edges, corners
	"""

	if T.TYPE_CHECKING:
		#parent:
		pass

	def __init__(self, parent:QtWidgets.QGraphicsItem=None,
	             data=None):
		super().__init__(parent=parent)
		self.drive(data["position"], self.setPos)
		self.drive(data, key="position", fn=self.setPos)



	def isSelectable(self)->bool:
		return False

	def isMoveable(self)->bool:
		return False