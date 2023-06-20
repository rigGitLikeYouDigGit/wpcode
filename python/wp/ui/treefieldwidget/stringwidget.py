from __future__ import annotations

from enum import Enum

from PySide2 import QtWidgets

# still not sure what to do with general lib widgets,
# totally possible Chimaera could depend on them
from tree.ui.libwidget.filebutton import FileBrowserButton

from wp.treefield import TreeField, TreeFieldParams
from wp import option
from wp.ui.treefieldwidget.base import TreeFieldWidgetBase

"""
string widget for string entry
"""


class StringWidget(QtWidgets.QLineEdit, TreeFieldWidgetBase):
	"""simple line edit
	if params define options, turn string widget into search widget

	"""

	def __init__(self, tree:TreeField, parent=None):
		TreeFieldWidgetBase.__init__(self, tree, parent)

		self.activeWidget = QtWidgets.QLineEdit(parent=parent)
		self.layout().addWidget(self.activeWidget)
		self.activeWidget.textChanged.connect(self._onUserInput)

		if self.fieldParams().placeholderText:
			self.activeWidget.setPlaceholderText(self.fieldParams().placeholderText)

	def _onUserInput(self, *args, **kwargs):
		"""User input has changed the value in the widget-
		update the tree value to match and fire signal to propagate"""
		self.tree.setValue(self.activeWidget.text())

	def _matchUiToValue(self, *args, **kwargs):
		"""set the ui to match the current value"""
		self.activeWidget.setText(self.tree.value())

if __name__ == '__main__':



	field = TreeField(
		"test",
		params=TreeFieldParams(
			isPath=True,
		)
	    )

	field.description = "test description"

	printMsg = lambda x: print("tree value changed to", x)
	field.valueChanged.connect(printMsg)

	app = QtWidgets.QApplication([])
	w = StringWidget(field)
	w.show()
	app.exec_()
