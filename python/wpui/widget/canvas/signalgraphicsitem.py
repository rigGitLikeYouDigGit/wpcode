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

class SignalGraphicsItem(base):
	"""
	mixin to create object-specific signals
	when mouse is pressed, released, moved, etc
	signals emit (self, event)

	TODO: potentially mouseMove, factor out from mouseover mixin
		if needed
	"""

	def __init__(self, *args, **kwargs):
		self.mousePressedSignal = Signal("mousePressed")
		self.mouseReleasedSignal = Signal("mouseReleased")

	def mousePressEvent(self, event):
		self.mousePressedSignal.emit(sender=self, event=event)
		return super().mousePressEvent(event)
	def mouseReleaseEvent(self, event):
		self.mouseReleasedSignal.emit(sender=self, event=event)
		return super().mouseReleaseEvent(event)

