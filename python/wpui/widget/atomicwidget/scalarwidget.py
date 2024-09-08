"""
widget for representing scalar numbers
slider or spinbox
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


class ScalarWidgetBase(AtomicWidget):

	def __init__(self, value,
	             params: AtomicWidgetParams = None,
	             ):

		self.scalarType = type(value)
		super(ScalarWidgetBase, self).__init__(value, params)

		self.minVal = self._params.min or 0.0
		self.maxVal = self._params.max or 1.0

		if self.minVal:
			self.setMinimum(self.minVal)
		if self.maxVal:
			self.setMaximum(self.maxVal)


	def _processUiResult(self, rawResult):
		return self.scalarType(rawResult)
	def _processValueForUi(self, rawValue):
		return self.scalarType(rawValue)
	def _setRawUiValue(self, value):
		self.setValue(value)
	def _rawUiValue(self):
		return self.value()

spinBoxClsMap = {int : QtWidgets.QSpinBox,
	float : QtWidgets.QDoubleSpinBox}

"""some repetition in these classes is fine"""
class FloatSpinBox(QtWidgets.QDoubleSpinBox, ScalarWidgetBase):

	atomicType = AtomicWidgetType.ScalarSpinBox
	def __init__(self, value, params:AtomicWidgetParams=None, parent=None):
		QtWidgets.QDoubleSpinBox.__init__(self, parent)
		ScalarWidgetBase.__init__(self, value, params=params)
		self.valueChanged.connect(self._onWidgetChanged)
		self.setOrientation(self._getParamsQtOrient())
class IntSpinBox(QtWidgets.QSpinBox, ScalarWidgetBase):
	atomicType = AtomicWidgetType.ScalarSpinBox
	def __init__(self, value, params:AtomicWidgetParams=None, parent=None):
		QtWidgets.QSpinBox.__init__(parent)
		ScalarWidgetBase.__init__(self, value, params)
		self.valueChanged.connect(self._onWidgetChanged)
		self.setOrientation(self._getParamsQtOrient())

scalarSpinBoxClsMap = {int : IntSpinBox,
	float : FloatSpinBox}

# class ScalarSpinBoxWidget(type):
# 	"""main imported class"""
#
# 	def __new__(mcs, value, resultParams:AtomicWidgetParams=None, parent=None):
# 		scalarType = type(value)
# 		widgetCls = scalarSpinBoxClsMap[scalarType]
# 		return widgetCls( value, resultParams, parent=None)

"""slider classes"""
class ScalarSliderBase(ScalarWidgetBase):
	pass

class FloatSlider(QtWidgets.QSlider, ScalarSliderBase):

	atomicType = AtomicWidgetType.ScalarSlider
	def __init__(self, value, params:AtomicWidgetParams=None,
	             parent=None):
		QtWidgets.QSlider.__init__(self, parent)
		ScalarSliderBase.__init__(self, value, params)
		self.setOrientation(self._getParamsQtOrient())
		# assume that 2 decimals are good enough precision
		self.setTracking(True)
		self.setRange(self.minVal * 100, self.maxVal * 100)
		self.valueChanged.connect(self._onWidgetChanged)
		#self.setAtomValue(value)

	def _processValueForUi(self, rawValue):
		return rawValue * 100
	def _processUiResult(self, rawResult):
		return rawResult / 100


class IntSlider(QtWidgets.QSlider, ScalarSliderBase):
	atomicType = AtomicWidgetType.ScalarSlider
	def __init__(self, value, params:AtomicWidgetParams=None, parent=None):
		QtWidgets.QSlider.__init__(self, parent)
		ScalarSliderBase.__init__(self, value, params)
		self.setOrientation(self._getParamsQtOrient())
		self.setTracking(True)
		self.valueChanged.connect(self.setAtomValue)


scalarSliderClsMap = {int : IntSlider,
	float : FloatSlider}
# class ScalarSliderWidget(type):
# 	"""main imported class"""
#
# 	def __new__(mcs, value, resultParams:AtomicWidgetParams=None,
# 	            parent=None, *args, **kwargs):
# 		scalarType = type(value)
# 		widgetCls = scalarSliderClsMap[scalarType]
# 		return widgetCls( value, resultParams, parent)

def test():

	import sys
	app = QtWidgets.QApplication(sys.argv)
	win = QtWidgets.QMainWindow()

	def printValue(member):
		print("new value", member)


	params = AtomicWidgetParams(None, min=0, max=13)
	#widg = FloatSpinBox(value=5.0, resultParams=resultParams, parent=win)
	widg = FloatSlider(value=5.0, params=params, parent=win)
	#widg = ScalarSliderWidget(value=5, minVal=0, maxVal=10, parent=win)

	print("widg", widg)
	widg.atomValueChanged.connect(printValue)
	print("w value", widg.atomValue())

	win.setCentralWidget(widg)
	win.show()
	sys.exit(app.exec_())

if __name__ == '__main__':
	test()



