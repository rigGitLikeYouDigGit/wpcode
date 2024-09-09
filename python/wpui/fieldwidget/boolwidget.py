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
from tree.lib.constant import FieldWidgetType
from tree.ui.fieldWidget import FieldWidgetParams
from tree.ui.fieldWidget.base import FieldWidget


class BoolWidgetBase(FieldWidget):
	"""base class defining init arguments"""

	def _processValueForUi(self, rawValue):
		return bool(rawValue)

	def _rawUiValue(self):
		return self.isChecked()

	def _setRawUiValue(self, value):
		self.setChecked(value)


class BoolCheckBoxWidget(QtWidgets.QCheckBox, BoolWidgetBase):

	atomicType = FieldWidgetType.BoolCheckBox

	def __init__(self, value=None, params:FieldWidgetParams=None, parent=None):
		QtWidgets.QCheckBox.__init__(self, parent)
		self.stateChanged.connect(self._onWidgetChanged)
		FieldWidget.__init__(self, value=value, params=params)
		self.setText(self._params.name)
		

class BoolCheckButtonWidget(QtWidgets.QPushButton, BoolWidgetBase):

	atomicType = FieldWidgetType.BoolCheckButton

	def __init__(self, value=None, params:FieldWidgetParams=None, parent=None):
		QtWidgets.QPushButton.__init__(self, parent)
		self.clicked.connect(self._onWidgetChanged)
		FieldWidget.__init__(self, value=value, params=params)
		self.setCheckable(True)
		self.setText(self._params.name)


def test():

	import sys
	app = QtWidgets.QApplication(sys.argv)
	win = QtWidgets.QMainWindow()

	def printValue(member):
		print("new value", member)

	params = FieldWidget.paramCls(None, name="test")
	#widg = BoolCheckBoxWidget(value=True,parent=win)
	widg = BoolCheckButtonWidget(value=True, params=params, parent=win)
	widg.atomValueChanged.connect(printValue)
	print("w value", widg.atomValue())

	win.setCentralWidget(widg)
	win.show()
	sys.exit(app.exec_())

if __name__ == '__main__':
	test()



