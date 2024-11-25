from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtGui, QtWidgets
from wplib.object import Signal

"""small base for widgets and graphicsItems only appearing when
mouse hovers near or over them"""

base = object
if T.TYPE_CHECKING:
	base = QtWidgets.QGraphicsItem

class MouseOverGraphicsItem(base):
	"""contains() by default accounts for different shaped
	borders and paths, use that

	maybe this shouldn't be a subclass, but a transparent parent?

	"""


	def __init__(self, *args, **kwargs):
		#super().__init__()
		# if these are None, mouse is not within range
		self._mouseNear : QtCore.QPointF = None
		self._mouseOver : QtCore.QPointF = None


	def appearOverArea(self)->QtCore.QRectF:
		"""return the area within which to display this item,
		not necessarily highlighted -
		by default this is the same as the highlight area,
		so the item will always appear immediately highlighted
		"""
		raise NotImplementedError(self)
	def highlightOverArea(self)->(QtCore.QRectF):
		return self.boundingRect()

	def sceneEvent(self, event:QtWidgets.QGraphicsSceneMouseEvent):
		if event.type() not in (QtCore.QEvent.Type.GraphicsSceneMouseMove,
		                        #QtCore.QEvent.Type.MouseMove,
		                        ):
			return super().sceneEvent(event)
		if self.appearOverArea().contains(event.scenePos()):
			self._mouseNear = event.scenePos()
		else:
			self._mouseNear = None
			self._mouseOver = None
			return super().sceneEvent(event)
		if self.highlightOverArea().contains(event.scenePos()):
			self._mouseOver = event.scenePos()
		else:
			self._mouseOver = None
			
		return super().sceneEvent(event)


	def paint(self, painter, option, widget=...):
		"""OVERRIDE for clarity,
		if _mouseNear isn't set, mouse is not within visible range, so
		don't paint item"""
		if self._mouseNear is None:
			return
		return super().paint(painter, option, widget)

