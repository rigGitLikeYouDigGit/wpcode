from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtGui, QtWidgets

class DragItemWidget(QtWidgets.QWidget):
	"""small holder widget to allow dragging model items
	up and down"""
	def __init__(self, innerWidget:QtWidgets.QWidget, parent=None):
		super().__init__(parent)
		self.dragHandle = QtWidgets.QLabel("☰", parent=self)
		self.dragHandle.setFont(QtGui.QFont("monospace", 8))
		self.innerWidget = innerWidget
		#self.innerWidget.setParent(self)
		layout = QtWidgets.QHBoxLayout()
		layout.addWidget(self.dragHandle)
		layout.addWidget(innerWidget)
		self.setLayout(layout)
		layout.setContentsMargins(0, 0, 0, 0)
		self.setContentsMargins(0, 0, 0, 0)
		layout.setAlignment(self.dragHandle, QtCore.Qt.AlignTop)

