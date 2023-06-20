from __future__ import annotations

from enum import Enum

from PySide2 import QtWidgets

from wp.treefield import TreeField, TreeFieldParams
from wp import option
from wp.ui.treefieldwidget.base import TreeFieldWidgetBase

"""
simple selector to choose between options

"""

class OptionWidgetMode(Enum):
	Combo = "combo"
	Radio = "radio"


class OptionWidgetBase(TreeFieldWidgetBase):
	"""Any shared abstract processing of enum widgets"""
	valueType = Enum

	def __init__(self, tree:TreeField, parent=None):
		super(OptionWidgetBase, self).__init__(tree, parent)
		self.optionItems = option.optionItemsFromOptions(
			self.fieldParams().options)

	def _optionMap(self)->dict:
		"""map enum values to option items"""
		return {item.name: item for item in self.optionItems}

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

class OptionComboWidget(OptionWidgetBase):
	"""same as above but in combo box form
	no label directly included"""


	def __init__(self, tree:TreeField, parent=None):
		#QtWidgets.QComboBox.__init__(self, parent)
		OptionWidgetBase.__init__(self, tree, parent)

		self.activeWidget = QtWidgets.QComboBox(self)
		self.layout().addWidget(self.activeWidget)

		for i in self.optionItems:
			self.activeWidget.addItem(i.name)
		self.activeWidget.currentIndexChanged.connect(self._onWidgetChanged)


	def _onWidgetChanged(self, *args, **kwargs):
		member : option.OptionItem = self._optionMap()[self.activeWidget.currentText()]
		self.tree.setValue(member.value)

	def _matchUiToValue(self, value:option.optionType):
		#print("match ui to value", value)
		key = option.optionKeyFromValue(optionValue=value, optionMap=self._optionMap())
		itemIndex = self.activeWidget.findText(key)
		self.activeWidget.setCurrentIndex(itemIndex)




if __name__ == '__main__':

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

