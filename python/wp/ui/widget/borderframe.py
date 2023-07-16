
from __future__ import annotations
import typing as T
from dataclasses import dataclass

from PySide2 import QtWidgets, QtCore, QtGui


"""mixin for easier control when drawing widget borders - 
different colours, styles, etc"""


@dataclass
class BorderData:
	"""dataclass for border data"""
	colour:QtGui.QColor = QtGui.QColor(128, 128, 128)
	width:int = 2
	style:QtCore.Qt.PenStyle = QtCore.Qt.SolidLine
	radius:int = 6

base = object
if T.TYPE_CHECKING: pass
base = QtWidgets.QFrame


class BorderFrame(#QtWidgets.QFrame
                  ):

	borderDataCls = BorderData

	def getBorderData(self) -> borderDataCls:
		"""return border data to use in painting border"""
		return self.borderDataCls()

	def paintEvent(self, arg__1:PySide2.QtGui.QPaintEvent) -> None:
		"""reimplement paint event to draw a border"""
		painter = QtGui.QPainter(self)
		painter.setRenderHint(QtGui.QPainter.Antialiasing)
		borderData = self.getBorderData()
		painter.setPen(QtGui.QPen(
				borderData.colour,
				borderData.width,
				borderData.style))
		painter.drawRoundedRect(self.contentsRect(),
		                        borderData.radius, borderData.radius)


