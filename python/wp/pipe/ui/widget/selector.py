
from __future__ import annotations
import typing as T

import time

from PySide2 import QtCore, QtGui, QtWidgets

import whoosh

from wplib import log, inheritance
from wptree import Tree
from wpui import lib
from wpui.widget.lantern import Status, Lantern
from wpui.treemenu import buildMenuFromTree
from wpdex.ui import AtomicWidget, StringWidget, FileStringWidget, FileBrowserButton
from wpdex import react

from wp.pipe.asset import Asset, Show, StepDir, search


class AssetCompleter(QtWidgets.QCompleter):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		#self.setModelSorting(self.ModelSorting.CaseInsensitivelySortedModel)

		# show all options by default - handle the highlighted completion section
		# in line edit
		self.setCompletionMode(self.CompletionMode.UnfilteredPopupCompletion)
		#self.setCompletionMode(self.CompletionMode.InlineCompletion)

	def splitPath(self, path):
		"""by default it splits on slashes - here we prevent this"""
		return [path]
	pass








class AssetSelectorWidget(
	QtWidgets.QWidget, AtomicWidget,
	metaclass=inheritance.resolveInheritedMetaClass(QtWidgets.QWidget, AtomicWidget)
):
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

	AssetField(StrField):
	- validation rule set
	- syntax passes for asset and tag expressions
	- whether the field is optional
	- whether multiple results should be returned
	- return a status enum based on the current exp - success for a valid expression, or if the field is empty and optional
	failure for invalid -
	a proper text field might just pick up those conditions natively


	it turns out QCompleter is quite difficult to dissuade from changing a lineedit's text as you change options - after trying a load of signal stuff and rolling own, just gonna hack around it somehow, and manage the cursor and selection manually

	TODO: we still have the annoying bug where the highlighting behaviour breaks after you
		show, hide, then show the completion widget again
		but I've spent too long on this

	TODO: as seen here, we can't easily compose atomic widgets of other ones
		hmmmmmmmm
	"""

	if T.TYPE_CHECKING:
		def value(self)->Asset: pass

	def __init__(self, parent=None, name="asset",
	             value:Asset=None
	             ):
		QtWidgets.QWidget.__init__(self, parent)
		AtomicWidget.__init__(self, value=value or Asset.topAssets()[0])
		#self.line = QtWidgets.QLineEdit(self)

		log("assetSelector init", react.canBeSet(self.rxValue), self.rxValue().rx.value)
		#testResult = self.rxValue().rx.where("a", "b")
		#log("test r", testResult.rx.value)
		#testResult = self.rxValue().rx.where(self.rxValue().upper(), "b")
		#log("test r", testResult.rx.value)

		self.line = StringWidget(
			# value=self.rxValue().rx.where(
			# 	self.rxValue().strPath(),
			# 	"<None>"),
			value=self.rxValue().rx.where(
				self.rxValue().strPath(),
				""),
			options=lambda : search.allPaths(),
			enableInteractionOnLocked=True
		                         )
		self.line.setPlaceholderText("path/uid/exp...")

		openAction = Tree("open folder...", value=lambda : lib.openExplorerOnPath(self.value().diskPath()))
		openAction.auxProperties["enable"] = lambda : self.value() is not None
		self.line.menuTree.addBranch(openAction)

		#self.line.completer().setModel(QtCore.QStringListModel(search.allPaths()))

		# get all scanned asset paths to
		#log("all paths", list(search.allPaths()))


		statusMap = {None : Lantern.Status.Neutral,
		             False : Lantern.Status.Failure,
		             }
		self.lantern = Lantern(
			value=self.rxImmediateValue().rx.pipe(statusMap.get, Lantern.Status.Success),
			parent=self
		)

		# the line is effectively the display for the asset
		self.line.valueCommitted.connect(self._fireDisplayCommitted)
		self.line.displayEdited.connect(self._fireDisplayEdited)

		#self.line.afterOptionHighlighted.connect(self._onOptionHighlighted)

		layout = QtWidgets.QHBoxLayout()
		layout.addWidget(self.line)
		layout.addWidget(self.lantern)
		self.setLayout(layout)

		self.postInit()

	def _resultForString(self, s)->(None, False, Asset):
		"""if not result : catches both fail states"""
		if not s or s == "<None>":
			return None
		return Asset.fromPath(self.line.text(), default=False)

	def setValue(self, value:(str, Asset)):
		if isinstance(value, str):
			value = self._resultForString(value)
		AtomicWidget.setValue(self, value)

	def _rawUiValue(self):
		return self.line.text()
	def _setRawUiValue(self, value):
		"""we don't actually set anything directly here,
		value asset updating automatically triggers the line edit"""
		pass

	def _processValueForUi(self, value):
		if value is None:
			return ""
		assert isinstance(value, Asset) # or list of assets?
		return value.strPath()
		pass
		#return value.strPath()
	def _processResultFromUi(self, value):
		return self._resultForString(value)
	def _tryCommitValue(self, value):
		#if value is None:
		if not value:
			return
		assert isinstance(value, Asset)
		self._commitValue(value)


	#def _onTextChangedByUi(self, *args, **kwargs ):
	def _onTextChanged(self, *args, **kwargs ):
		if self.getValue():
			self.lantern.setStatus(Status.Success)
		else:
			self.lantern.setStatus(Status.Failure)

		#self.line.selectAll()

	def _tryCommitText(self, *args, **kwargs):
		if self.getValue():
			self._lastValue = Asset.fromPath(self.line.text())
			log("commit new value")
		else:
			self.setValue(self._lastValue)
			log("fail commit")



if __name__ == '__main__':
	app = QtWidgets.QApplication()
	w = AssetSelectorWidget()
	w.show()
	app.exec_()


