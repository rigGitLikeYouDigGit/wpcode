
from __future__ import annotations
import typing as T

from wplib import log, Sentinel, TypeNamespace
from wpdex import *
from PySide2 import QtCore, QtWidgets, QtGui

"""
the first step towards making tools a bit more beautiful

small indicator light at the side of widgets, can display warnings, queries,
progress

"""

class Status(TypeNamespace):
	"""move this higher later, but the idea is that tools should have
	a consistent set of states they can reach -
	success, error, warning, in progress etc

	colours as floats seems most general - at least this way we only have
	to convert one way, and not worry about dividing by 128 first
	"""

	class _Base(TypeNamespace.base()):
		colour = (0.5, 0.5, 0.5)
		pass

	class Success(_Base): # green
		colour = (0.1, 0.9, 0.3)
		pass

	class Warning(_Base): # orange
		colour = (0.8, 0.6, 0.1)

	class Failure(_Base): # red
		colour = (0.9, 0.0, 0.4)
		pass

	class Neutral(_Base): # grey
		pass

	class Incoming(_Base): # incoming usually more important than outgoing
		colour = (1.0, 0.5, 0.3)  # orange
	class Outgoing(_Base):
		colour = (0.3, 0.7, 1.0) # light blue


class Lantern(QtWidgets.QWidget):

	Status = Status
	
	def __init__(self, value=Status.Neutral, parent=None):
		super().__init__(parent)
		self._value = rx(value)
		self.setAutoFillBackground(True)
		#self.setWindowOpacity(1.0)
		self.setContentsMargins(0, 0, 0, 0)
		self.setFixedSize(10, 10)
		self._value.rx.watch(lambda *a : self.repaint(),
		                     onlychanged=False)

	def value(self):
		return self._value.rx.value
	def rxValue(self):
		return self._value
		#self.status. self.update()


	def setStatus(self, status:Status.T()):
		self._value.rx.value = status
		#self.repaint()

	def paintEvent(self, event):
		painter = QtGui.QPainter(self)
		status = self.value()
		#brush = QtGui.QBrush(QtGui.QColor.fromRgb(128, 200, 128, 128))
		brush = QtGui.QBrush(QtGui.QColor.fromRgbF(*status.colour, 1))
		painter.setBrush(
			#QtGui.QBrush(QtGui.QColor.fromRgbF(*status.colour, 0.5))
			brush
		)
		painter.fillRect(#0, 0, 100, 100,
			self.rect(),
		                 brush
		                 #QtCore.Qt.BrushStyle.RadialGradientPattern
		)


