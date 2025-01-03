from __future__ import annotations

"""general lib widget for representing options
of a python enum or any other sequence or mapping

option values converted to string internally, retrieved on request

"""
import builtins
import inspect, pprint, dis
import typing as T
from enum import Enum
from dataclasses import dataclass

from PySide2 import QtCore, QtWidgets, QtGui
from tree.lib.constant import FieldWidgetType

from tree.ui.fieldWidget import FieldWidgetParams
from tree.ui.fieldWidget.base import FieldWidget
from tree.lib.option import OptionItem, optionType, optionMapFromOptions, optionKeyFromValue, optionFromKey

class OptionWidgetMode(Enum):
	Combo = "combo"
	Radio = "radio"

class OptionWidgetBase(FieldWidget):
	"""Any common abstract processing of enum widgets"""
	valueType = Enum
	def __init__(self, value=None, params:FieldWidgetParams=None, parent=None):
		FieldWidget.__init__(self, value, params)
		self.optionMap = optionMapFromOptions(self._params.options)


# todo: processing to split enum value name by camelcase
class OptionRadioButton(QtWidgets.QRadioButton):
	"""Radio button for a single enum option"""

	def __init__(self, key:str, option, parent: OptionButtonWidget = None
	             ):
		self.option = option
		super(OptionRadioButton, self).__init__(key,  parent
		                                        )
class OptionStickyButton(QtWidgets.QPushButton):
	"""Radio button for a single enum option"""

	def __init__(self, key:str, option, parent: OptionButtonWidget = None
	             ):
		self.option = option
		super(OptionStickyButton, self).__init__(key,  parent
		                                        )
		self.setCheckable(True)

buttonTypeClsMap = {
	FieldWidgetParams.OptionType.Radio : OptionRadioButton,
	FieldWidgetParams.OptionType.Sticky : OptionStickyButton
}

class OptionButtonWidget(QtWidgets.QGroupBox, OptionWidgetBase):
	"""puts all enum options to radio buttons"""

	atomicType = FieldWidgetType.OptionRadio

	def __init__(self,
	             value=None,
	             params:FieldWidgetParams=None,
	             parent=None):
		QtWidgets.QGroupBox.__init__(self, parent)
		OptionWidgetBase.__init__(self, value=value, params=params, parent=parent)
		name = params.caption or ""
		self.setTitle(name)


		self.radioButtonCls = buttonTypeClsMap[self._params.optionType]
		self.buttons = {k : self.radioButtonCls(k, v, parent=self) for k, v in self.optionMap.items()}

		layout = QtWidgets.QHBoxLayout(self) if self._getParamsQtOrient() == QtCore.Qt.Orientation.Horizontal else QtWidgets.QVBoxLayout(self)
		for i in self.buttons.values():
			i.clicked.connect(self._onWidgetChanged)
			layout.addWidget(i)
		self.setLayout(layout)
		self.setAtomValue(tuple(self.buttons.values())[0].option)

		if value:
			self.setAtomValue(value)


	def _onWidgetChanged(self, *args, **kwargs):
		radioBtn : OptionRadioButton = self.sender()
		if radioBtn is None:
			return
		optionValue = radioBtn.option
		self.setAtomValue(optionValue)

	def _matchUiToValue(self, value):
		key = optionKeyFromValue(value, self.optionMap)
		self.buttons[key].click()

class OptionComboWidget(QtWidgets.QComboBox, OptionWidgetBase):
	"""same as above but in combo box form
	no label directly included"""

	atomicType = FieldWidgetType.OptionMenu

	def __init__(self, value=None,
	             params: FieldWidgetParams = None,
	             parent=None):
		#super(OptionComboWidget, self).__init__(parent)
		QtWidgets.QComboBox.__init__(self, parent)
		OptionWidgetBase.__init__(self, value=value, params=params)
		for k, v in self.optionMap.items():
			self.addItem(k)
		self.currentIndexChanged.connect(self._onWidgetChanged)
		if value:
			self.setAtomValue(value)

	def _onWidgetChanged(self, *args, **kwargs):
		member : OptionItem = self.optionMap[self.currentText()]
		self.setAtomValue(member)

	def _matchUiToValue(self, value:Enum):
		key = optionKeyFromValue(optionValue=value, optionMap=self.optionMap)
		itemIndex = self.findText(key)
		self.setCurrentIndex(itemIndex)




def testKeys():
	keyCls = type({}.keys())
	p = r"F:\all_projects_desktop\common\edCode\tree\ui\out.txt"
	dis.dis(dict, file=open(p, "w"))
	#d = dis.disassemble(keyCls, file=p)
	#print(d)




def test():
	OptionButtonWidget

	import sys
	app = QtWidgets.QApplication(sys.argv)
	win = QtWidgets.QMainWindow()

	def printValue(member):
		print("new enum value", member)

	class TestEnum(Enum):
		OptionA = "option_a"
		OptionB = "other_option"

	options = TestEnum

	options = ("a", "b", "c")

	#options = {"optionB" : "valueB", 33 : 55}
	params = OptionButtonWidget.paramCls(
		None, "testWidget",
		options=options
	)
	#widg = OptionButtonWidget(None, resultParams, parent=win)
	widg = OptionComboWidget(None, params, parent=win)

	# widg = OptionComboWidget(TestEnum, value=TestEnum.OptionA, parent=win)
	widg.atomValueChanged.connect(printValue)
	print("w value", widg.atomValue())

	win.setCentralWidget(widg)
	win.show()
	sys.exit(app.exec_())

if __name__ == '__main__':
	test()


