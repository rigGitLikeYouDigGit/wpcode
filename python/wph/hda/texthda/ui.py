from __future__ import annotations
import types, typing as T
import pprint
#from wplib import log

from PySide6 import QtCore, QtWidgets, QtGui


class SaveNewDefDialog(QtWidgets.QWidget):
	"""When user presses button to save delta as new
	definition, show them this -
	maybe try and emulate normal houdini dialog, with namespaces,
	authors?

	"""
	def __init__(self, parent=None):
		super().__init__(parent)

		self.setWindowTitle("Create new TextHDA definition:")

		self.fileLine = QtWidgets.QLineEdit(self)
		self.fileLine.setPlaceholderText("Name")


