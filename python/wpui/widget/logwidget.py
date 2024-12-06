from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
from queue import Queue
from PySide2 import QtCore, QtWidgets, QtGui


class LogWidget(QtWidgets.QTextBrowser):
	"""realistically this is bottom of all priorities,
	I am a basic printing boy for life"""

	# if user clicks a hyperlink in text log, for an error line, active options, or something
	linkClicked = QtCore.Signal(dict)

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setAutoFillBackground(True)
		self.setTextBackgroundColor(QtCore.Qt.black)

