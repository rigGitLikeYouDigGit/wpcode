from __future__ import annotations

import typing as T
from enum import Enum

from PySide2 import QtWidgets, QtCore, QtGui

from wp.treefield import TreeField, TreeFieldParams
from wp import option, constant



"""
simple selector to choose between options

"""

class OptionWidgetMode(Enum):
	Combo = "combo"
	Radio = "radio"

class TreeFieldWidgetBase:
	"""base class for generating UI elements from tree fields"""
	def __init__(self, tree:TreeField, parent=None):
		self.tree : TreeField = tree
		#super(TreeFieldWidgetBase, self).__init__(parent)

		# connect immediate signal to update UI value
		self.tree.valueChanged.connect(self._matchUiToValue)

	def fieldParams(self)->TreeFieldParams:
		return self.tree.params

	def _onUserInput(self, *args, **kwargs):
		"""User input has changed the value in the widget-
		update the tree value to match and fire signal to propagate"""

		pass

	def _matchUiToValue(self, *args, **kwargs):
		"""set the ui to match the current value"""
		pass

class OptionWidgetBase(TreeFieldWidgetBase):
	"""Any shared abstract processing of enum widgets"""
	valueType = Enum

	def __init__(self, tree:TreeField, parent=None):
		super(OptionWidgetBase, self).__init__(tree, parent)
		self.optionItems = option.optionItemsFromOptions(
			self.fieldParams().options)


#
# # todo: processing to split enum value name by camelcase
# class OptionRadioButton(QtWidgets.QRadioButton):
# 	"""Radio button for a single enum option"""
#
# 	def __init__(self, key:str, option, parent: OptionButtonWidget = None
# 	             ):
# 		self.option = option
# 		super(OptionRadioButton, self).__init__(key,  parent
# 		                                        )
# class OptionStickyButton(QtWidgets.QPushButton):
# 	"""Radio button for a single enum option"""
#
# 	def __init__(self, key:str, option, parent: OptionButtonWidget = None
# 	             ):
# 		self.option = option
# 		super(OptionStickyButton, self).__init__(key,  parent
# 		                                        )
# 		self.setCheckable(True)
#
# # buttonTypeClsMap = {
# # 	AtomicWidgetParams.OptionType.Radio : OptionRadioButton,
# # 	AtomicWidgetParams.OptionType.Sticky : OptionStickyButton
# # }
#
# class OptionButtonWidget(QtWidgets.QGroupBox, OptionWidgetBase):
# 	"""puts all enum options to radio buttons"""
#
# 	#atomicType = AtomicWidgetType.OptionRadio
#
# 	def __init__(self,
# 	             value=None,
# 	             params:AtomicWidgetParams=None,
# 	             parent=None):
# 		QtWidgets.QGroupBox.__init__(self, parent)
# 		OptionWidgetBase.__init__(self, value=value, params=params, parent=parent)
# 		name = params.caption or ""
# 		self.setTitle(name)
#
#
# 		self.radioButtonCls = buttonTypeClsMap[self._params.optionType]
# 		self.buttons = {k : self.radioButtonCls(k, v, parent=self) for k, v in self.optionMap.items()}
#
# 		layout = QtWidgets.QHBoxLayout(self) if self._getParamsQtOrient() == QtCore.Qt.Orientation.Horizontal else QtWidgets.QVBoxLayout(self)
# 		for i in self.buttons.values():
# 			i.clicked.connect(self._onWidgetChanged)
# 			layout.addWidget(i)
# 		self.setLayout(layout)
# 		self.setAtomValue(tuple(self.buttons.values())[0].option)
#
# 		if value:
# 			self.setAtomValue(value)
#
#
# 	def _onWidgetChanged(self, *args, **kwargs):
# 		radioBtn : OptionRadioButton = self.sender()
# 		if radioBtn is None:
# 			return
# 		optionValue = radioBtn.option
# 		self.setAtomValue(optionValue)
#
# 	def _matchUiToValue(self, value):
# 		key = optionKeyFromValue(value, self.optionMap)
# 		self.buttons[key].click()

class OptionComboWidget(QtWidgets.QComboBox, OptionWidgetBase):
	"""same as above but in combo box form
	no label directly included"""


	def __init__(self, tree:TreeField, parent=None):
		QtWidgets.QComboBox.__init__(self, parent)
		OptionWidgetBase.__init__(self, tree, parent)

		for i in self.optionItems:
			self.addItem(i.name)
		self.currentIndexChanged.connect(self._onWidgetChanged)


	def _onWidgetChanged(self, *args, **kwargs):
		member : option.OptionItem = self.optionMap[self.currentText()]
		self.setAtomValue(member)

	def _matchUiToValue(self, value:option.optionType):
		print("match ui to value", value)
		key = option.optionKeyFromValue(optionValue=value, optionMap=self.optionMap)
		itemIndex = self.findText(key)
		self.setCurrentIndex(itemIndex)




if __name__ == '__main__':

	from tree.lib.object import TypeNamespace

	class TestEnum(Enum):
		One = 1
		Two = 2
		Three = 3

	field = TreeField(
		"test",
		params=TreeFieldParams(
			options=TestEnum
		)
	    )

	printMsg = lambda x: print("tree value changed to", x)
	field.valueChanged.connect(printMsg)

	app = QtWidgets.QApplication([])
	w = OptionComboWidget(field)
	w.show()
	app.exec_()

