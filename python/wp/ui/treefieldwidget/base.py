from __future__ import annotations

from enum import Enum

from PySide2 import QtWidgets

from wp.treefield import TreeField, TreeFieldParams


class TreeFieldWidgetBase(QtWidgets.QWidget):
	"""base class for generating UI elements from tree fields
	base class is outer holder widget"""
	def __init__(self, tree:TreeField, parent=None):
		self.tree : TreeField = tree
		super(TreeFieldWidgetBase, self).__init__(parent)

		self.labelWidget : QtWidgets.QLabel = None
		self.activeWidget : QtWidgets.QWidget = None

		self.setLayout(QtWidgets.QHBoxLayout())

		if self.fieldParams().showLabel:
			self.labelWidget = self.createLabelWidget()
			self.layout().addWidget(self.labelWidget)

		# connect immediate signal to update UI value
		self.tree.valueChanged.connect(self._matchUiToValue)

	def _labelTooltip(self)->str:
		"""return tooltip for label"""
		lines = "\n".join([
			self.tree.stringAddress(includeSelf=True),
			self.tree.description,
			"\n\t".join(str(self.tree.params).split(", "))
			])
		return lines

	def createLabelWidget(self)->QtWidgets.QLabel:
		"""create a label displaying data about this field -
		name, full path, parametres, etc"""
		label = QtWidgets.QLabel(self.tree.getName(), self)
		label.setToolTip(self._labelTooltip())

		label.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
		label.setStyleSheet("QLabel{margin-left: 5px; border-radius: 5px; background: white;}")
		return label



	def fieldParams(self)->TreeFieldParams:
		return self.tree.params

	def _onUserInput(self, *args, **kwargs):
		"""User input has changed the value in the widget-
		update the tree value to match and fire signal to propagate"""

		pass

	def _matchUiToValue(self, *args, **kwargs):
		"""set the ui to match the current value"""
		pass