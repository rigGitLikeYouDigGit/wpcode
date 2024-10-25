

from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

"""
each package can define an "onStartup" function,
fully immersed in dcc domain code, and idem calls it
once dcc is running


"""

class Window(QtWidgets.QWidget):

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setObjectName("idem")
		self.setWindowTitle("idem")

	@classmethod
	def launch(cls):
		app = QtWidgets.QApplication()
		w = cls()
		w.show()
		app.exec_()

