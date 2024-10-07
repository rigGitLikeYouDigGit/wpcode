
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

from wpdex.ui import ReactiveWidget, WidgetHook

from wp.pipe.asset import Asset, Show, StepDir


class AssetSelectorWidget(QtWidgets.QWidget, ReactiveWidget):
	"""widget representing an asset reference -
	asset path or expression, version options, vcs status.
	rich data not represented

	how do we do nested interactions within a reactive widget like this?
	we can set asset = this ;
	can we set asset/version = this ?


	TODO: line edit should be replaced with an expression widget
	 exp line widget should accept arbitrary rules for validation, prediction, suggestion etc

	reactive widget probably always needs a concrete reference to a
	model in order to work, right? model could be wpdex, tree etc
	"""

	def __init__(self, parent=None, name="asset"):
		super().__init__(parent)
		self.line = QtWidgets.QLineEdit(self)
		self.line.setPlaceholderText("path/uid/exp...")

		ReactiveWidget.__init__(self, name)


	def _isValid(self):
		return Asset.isAssetDir(self.line.text())

	def getValue(self, **kwargs):
		return Asset.fromPath(self.line.text())

	def setValue(self, val:Asset, **kwargs):
		self.line.setText(val.strPath(root=1))





