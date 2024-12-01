from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtWidgets, QtGui

class Window(QtWidgets.QWidget):

	def __init__(self, parent=None):
		super().__init__(parent)

		self.lineA = QtWidgets.QLineEdit("lineA", self)
		self.lineB = QtWidgets.QLineEdit("lineB", self)

		vl = QtWidgets.QVBoxLayout(self)
		self.setLayout(vl)
		vl.addWidget(self.lineA)
		vl.addWidget(self.lineB)

if __name__ == '__main__':

	app = QtWidgets.QApplication()

	w = Window()
	w.show()

	app.exec_()




