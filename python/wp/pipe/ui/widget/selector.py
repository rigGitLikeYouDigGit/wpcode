
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

import whoosh

from wplib import log
from wpui.widget.lantern import Status, Lantern
from wpdex.ui import ReactiveWidget, WidgetHook

from wp.pipe.asset import Asset, Show, StepDir, search


class AssetCompleter(QtWidgets.QCompleter):
	pass

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

	def __init__(self, parent=None, name="asset",
	             default:Asset=None):
		super().__init__(parent)
		self.line = QtWidgets.QLineEdit(self)
		self.line.setPlaceholderText("path/uid/exp...")

		ReactiveWidget.__init__(self, name)
		# get all scanned asset paths to
		self.completer = AssetCompleter(search.allPaths(), parent=self)
		self.line.setCompleter(self.completer)

		self.lantern = Lantern(status=Status.Neutral, parent=self)

		self.line.textChanged.connect(self._onTextChanged)
		self.line.editingFinished.connect(self._tryCommitText)

		asset = default or Asset.topAssets()[0]
		self.line.setText(asset.strPath())
		self._lastValue = asset

		layout = QtWidgets.QHBoxLayout()
		layout.addWidget(self.line)
		layout.addWidget(self.lantern)
		self.setLayout(layout)

		self.lantern.setFixedSize(10, 10)

	def _uiChangeQtSignals(self):
		return []

	#def _onTextChangedByUi(self, *args, **kwargs ):
	def _onTextChanged(self, *args, **kwargs ):
		if self.getValue():
			self.lantern.setStatus(Status.Success)
		else:
			self.lantern.setStatus(Status.Failure)

	def _tryCommitText(self, *args, **kwargs):
		if self.getValue():
			self._lastValue = Asset.fromPath(self.line.text())
			log("commit new value")
		else:
			self.setValue(self._lastValue)
			log("fail commit")

	def getValue(self, **kwargs):
		try:
			return Asset.fromPath(self.line.text())
		except (Asset.PathKeyError, KeyError):
			return None

	def setValue(self, val:Asset, **kwargs):
		self.line.setText(val.strPath(root=0))


if __name__ == '__main__':
	app = QtWidgets.QApplication()
	w = AssetSelectorWidget()
	w.show()
	app.exec_()


