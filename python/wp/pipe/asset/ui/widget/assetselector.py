
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

from wpdex.ui import ReactiveWidget


class AssetSelector(QtWidgets.QWidget, ReactiveWidget):

	def __init__(self, parent=None):
		super().__init__(parent)

		vlayout = QtWidgets.QVBoxLayout()
		self.line = QtWidgets.QLineEdit(parent=self)





