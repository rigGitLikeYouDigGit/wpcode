
from __future__ import annotations
import typing as T

import time

from PySide2 import QtCore, QtGui, QtWidgets

import whoosh

from wplib import log
from wpui.widget.lantern import Status, Lantern
from wpdex.ui import ReactiveWidget, WidgetHook

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

class MenuLineEdit(QtWidgets.QLineEdit):
	"""completer seems to always modify the text in its line -
	we want to micromanage how that happens, so we can't use completer.
	line edit where the completer just chills the f out
	don't actually modify the text of lineedit, just emit
	that a selection has changed
	"""

	# fires after option has been highlighted, and all the string restoring has calmed down
	afterOptionHighlighted = QtCore.Signal(str)

	def __init__(self, *args,
	             suggestionsForTextFn:T.Callable[[str], list[str]]=None,
	             **kwargs):
		super().__init__(*args, **kwargs)
		#self.setEditable(True)
		self.suggestionsForTextFn=suggestionsForTextFn
		self.lastSuggestions = []
		self._completer = QtWidgets.QCompleter(self)
		self._completer.setObjectName("completer")
		self.setCompleter(self._completer)
		self.setObjectName("line")
		self._prevText = ""
		self._prevCursor = None
		self._lastHighlighted = "" # i commend to thee my soul o god
		self.completer().highlighted.connect(self._onHighlighted)
		self.textChanged.connect(self._onTextChanged)

	def _saveText(self):
		self._prevCursor = self.cursorPosition()
		self._prevText = self.text()

	def _restoreText(self):
		self.setText(self._prevText)
		self.setCursorPosition(self._prevCursor)

	def keyPressEvent(self, arg__1):
		# feels filthy to put this here
		if arg__1.key() in (QtCore.Qt.Key_Down, QtCore.Qt.Key_Up):
			pass
		else:
			self._lastHighlighted = "" # see the great shame of man
			self._saveText()
		super().keyPressEvent(arg__1)
		#print("end key press", self._lastHighlighted)


	def _onHighlighted(self, *args, **kwargs):
		"""restore previous text"""
		#print("")
		#log("onHighlighted", self._lastHighlighted, self.text(), self._prevText)
		#log("current text", self.text())
		if not self._lastHighlighted:
			self._saveText()
		self._lastHighlighted = args[0]
		#print("end highlighted")

	def _onTextChanged(self, *args, **kwargs):
		"""naively one might think that
		highlighting new options in the completer dropdown would set the completer
		to be the signal sender
		only a fool would be so naive
		I can't see a strong way to tell a highlight signal from a text changed signal,
		so we do something quite disgusting here

		textChanged() also gets some kind of double-tap if you bring up the completion popup, hide it by typing input that matches no options, then bring it back again -
		for that reason we need to split the "_lastHighlighted" state tracking across these
		signal slots AND keyPressEvent
		"""
		#log("onTextChanged", self._lastHighlighted, self.text(), self._prevText)
		if self._lastHighlighted:
			self.blockSignals(True)
			self._restoreText()
			self.blockSignals(False)
			tempLastHighlighted = self._lastHighlighted
			self._lastHighlighted = ""
			self.afterOptionHighlighted.emit(tempLastHighlighted)
			self._lastHighlighted = tempLastHighlighted
			#self._lastHighlighted = ""






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
	"""

	def __init__(self, parent=None, name="asset",
	             default:Asset=None):
		super().__init__(parent)
		#self.line = QtWidgets.QLineEdit(self)
		self.line = MenuLineEdit(self,
		                         suggestionsForTextFn=search.searchPaths)
		self.line.setPlaceholderText("path/uid/exp...")

		self.line.completer().setModel(QtCore.QStringListModel(search.allPaths()))

		ReactiveWidget.__init__(self, name)
		# get all scanned asset paths to
		log("all paths", list(search.allPaths()))

		self.lantern = Lantern(status=Status.Neutral, parent=self)

		self.line.textChanged.connect(self._onTextChanged)
		self.line.editingFinished.connect(self._tryCommitText)

		self.line.afterOptionHighlighted.connect(self._onOptionHighlighted)


		asset = default or Asset.topAssets()[0]
		self.line.setText(asset.strPath())
		self._lastValue = asset

		layout = QtWidgets.QHBoxLayout()
		layout.addWidget(self.line)
		layout.addWidget(self.lantern)
		self.setLayout(layout)

		self.lantern.setFixedSize(10, 10)

	def _onOptionHighlighted(self, *args, **kwargs):
		"""complete only the rest of the highlighted option in the line

		this fires after option selected in completer dropdown
		"""
		#log("highlighted", args, kwargs)
		baseText = self.line.text()
		#log("base text", baseText)
		suggestPath = args[0]
		highlightStartId = len(baseText)
		self.line.setText(suggestPath)
		if baseText in suggestPath:
			self.line.setSelection(highlightStartId, len(suggestPath) - 1)
		#log("line selected text", self.line.selectedText())

	def _uiChangeQtSignals(self):
		return []

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


