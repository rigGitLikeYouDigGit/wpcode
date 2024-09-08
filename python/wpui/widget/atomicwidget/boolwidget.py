"""
widget for representing single bool value
for now only checkbox
"""

from __future__ import annotations
import builtins
import inspect, pprint, dis
import typing as Ty
from enum import Enum

from PySide2 import QtCore, QtWidgets, QtGui
from tree.lib.constant import AtomicWidgetType
from tree.ui.atomicwidget import AtomicWidgetParams
from tree.ui.atomicwidget.base import AtomicWidget


class BoolWidgetBase(AtomicWidget):
	"""base class defining init arguments"""

	def _processValueForUi(self, rawValue):
		return bool(rawValue)

	def _rawUiValue(self):
		return self.isChecked()

	def _setRawUiValue(self, value):
		self.setChecked(value)


class BoolCheckBoxWidget(QtWidgets.QCheckBox, BoolWidgetBase):

	atomicType = AtomicWidgetType.BoolCheckBox

	def __init__(self, value=None, params:AtomicWidgetParams=None, parent=None):
		QtWidgets.QCheckBox.__init__(self, parent)
		self.stateChanged.connect(self._onWidgetChanged)
		AtomicWidget.__init__(self, value=value, params=params)
		self.setText(self._params.name)
		

class BoolCheckButtonWidget(QtWidgets.QPushButton, BoolWidgetBase):

	atomicType = AtomicWidgetType.BoolCheckButton

	def __init__(self, value=None, params:AtomicWidgetParams=None, parent=None):
		QtWidgets.QPushButton.__init__(self, parent)
		self.clicked.connect(self._onWidgetChanged)
		AtomicWidget.__init__(self, value=value, params=params)
		self.setCheckable(True)
		self.setText(self._params.name)


def test():

	import sys
	app = QtWidgets.QApplication(sys.argv)
	win = QtWidgets.QMainWindow()

	def printValue(member):
		print("new value", member)

	params = AtomicWidget.paramCls(None, name="test")
	#widg = BoolCheckBoxWidget(value=True,parent=win)
	widg = BoolCheckButtonWidget(value=True, params=params, parent=win)
	widg.atomValueChanged.connect(printValue)
	print("w value", widg.atomValue())

	win.setCentralWidget(widg)
	win.show()
	sys.exit(app.exec_())

if __name__ == '__main__':
	test()



