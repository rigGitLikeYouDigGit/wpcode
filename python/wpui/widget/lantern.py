
from __future__ import annotations
import typing as T

from wplib import log, Sentinel, TypeNamespace

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

	class Success(_Base):
		colour = (0.1, 0.9, 0.3)
		pass

	class Failure(_Base):
		colour = (0.9, 0.0, 0.4)
		pass

	class Neutral(_Base): pass


class Lantern(QtWidgets.QWidget):
	
	def __init__(self, status=Status.Neutral, parent=None):
		super().__init__(parent)
		self.status = status
		self.setAutoFillBackground(True)
		#self.setWindowOpacity(1.0)
		self.setContentsMargins(0, 0, 0, 0)



	def setStatus(self, status:Status.T()):
		self.status = status
		self.repaint()

	def paintEvent(self, event):
		painter = QtGui.QPainter(self)
		status = self.status or Status.Neutral
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


