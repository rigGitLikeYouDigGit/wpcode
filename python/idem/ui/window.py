

from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

"""
each package can define an "onStartup" function,
fully immersed in dcc domain code, and idem calls it
once dcc is running


"""

class Window(QtWidgets.QWidget):

	@classmethod
	def launch(cls):
		app = QtWidgets.QApplication()
		w = cls()
		w.show()
		app.exec_()

