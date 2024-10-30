from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
from dataclasses import dataclass
from typing import TypedDict

from PySide2 import QtCore, QtGui, QtWidgets

from wpui import lib as uilib

"""sketch for applying layers on top of a full qt window - 
drawing outlines over windows, highlighting widgets
that give errors, etc

also previews of where focus will move on
pressing tab + something else?
"""

class HighlightData(TypedDict):
	w : QtWidgets.QWidget
	colour : tuple[float, ...]
	width : int

class Overlay(QtWidgets.QWidget):
	"""test working as a transparent top-level widget
	"""

	@classmethod
	def defaultHighlightData(cls)->HighlightData:
		return HighlightData(w=None,
		                     colour=(0.0, 1.0, 0.0),
		                     width=2)
	
	def __init__(self, name:str, topLevelParent:QtWidgets.QWidget):
		super().__init__(topLevelParent) # no parent
		self.setObjectName(name)
		self.highlightDatas : list[HighlightData] = []
		self.setAutoFillBackground(False)
		self.setHidden(True)

	def setHighlights(self, datas:list[HighlightData]):
		self.highlightDatas = datas
		self.repaint()

	def paintEvent(self, event:QtGui.QPaintEvent):
		"""draw outline rects around highlighted widgets"""
		painter = QtGui.QPainter(self)
		for i in self.highlightDatas:
			i = {**self.defaultHighlightData(), **i}
			col = QtGui.QColor.fromRgbF(*i["colour"])
			rect = self.parent().mapFrom(i["w"], i["w"].rect())
			pen = QtGui.QPen(col)
			pen.setWidth(2)
			painter.setPen(pen)
			painter.drawRoundRect(rect)






if __name__ == '__main__':
	app = QtWidgets.QApplication()
	w = Overlay()
	pass



