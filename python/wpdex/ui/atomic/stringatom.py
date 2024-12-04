

from __future__ import annotations

import pathlib
import traceback
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

import wplib.sequence
from wplib import log, inheritance
from wplib.object import Signal
from wptree import Tree

from wpexp.tostr import toStr
from wpdex import *
from wpdex import react
#from wpdex.ui.react import ReactiveWidget
from wpdex.ui.atomic.base import AtomicWidgetOld
from wpui.widget import FileBrowserButton, Lantern
from wpui.treemenu import buildMenuFromTree

# i summon ancient entities
from wpexp.syntax import SyntaxPasses, ExpSyntaxProcessor


"""



"""

class _MenuLineEdit(QtWidgets.QLineEdit):
	"""completer seems to always modify the text in its line -
	we want to micromanage how that happens, so we can't use completer.
	line edit where the completer just chills the f out
	don't actually modify the text of lineedit, just emit
	that a selection has changed

	couldn't figure this out, and at the moment it's not
	worth the time
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


class ExpWidget(
	inheritance.MetaResolver,
	QtWidgets.QLineEdit, AtomicWidgetOld,
	# metaclass=inheritance.resolveInheritedMetaClass(
	# QtWidgets.QLineEdit, AtomicWidgetOld
	# )
):
	"""TODO:
	control settings through dex or path rules or something, it's getting
	too complex to define as __init__ args
	"""

	def __init__(self, value="", parent=None,
	             options:T.Sequence[str]=(),
	             conditions:T.Sequence[AtomicWidgetOld.Condition]=(),
	             warnLive=False,
	             light=False,
	             enableInteractionOnLocked=False,
	             placeHolderText="",
	             allowEval=False,
	             forceResultType:(type, tuple[type])=None,

	             ):
		QtWidgets.QLineEdit.__init__(self, parent=parent)
		AtomicWidgetOld.__init__(self, value=value,
		                         conditions=conditions,
		                         warnLive=warnLive,
		                         enableInteractionOnLocked=enableInteractionOnLocked
		                         )

		self.setCompleter(QtWidgets.QCompleter(self))

		self.menuTree = Tree("menu")
		self.allowEval = allowEval
		self.forceResultType = forceResultType

		self.options = options if isinstance(options, rx) else WX(options)
		self.options.rx.watch(
			self._setOptions,
			onlychanged=False,
			fire=True
		)

		self._placeHolderText = WX(placeHolderText)
		self._placeHolderText.rx.watch(self.setPlaceholderText, fire=True)
		# canBeSet = react.canBeSet(self.rxValue())
		# log(self, "init can be set", canBeSet)
		#assert not canBeSet

		# connect signals
		self.textChanged.connect(self._syncImmediateValue)
		self.textEdited.connect(self._fireDisplayEdited)
		self.editingFinished.connect(self._fireDisplayCommitted)
		self.returnPressed.connect(self._fireDisplayCommitted)

		#self.postInit()

	def contextMenuEvent(self, arg__1:QtGui.QContextMenuEvent):
		baseMenu = self.createStandardContextMenu()
		if self.menuTree.branches:
			treeMenu = buildMenuFromTree(self.menuTree, menu=baseMenu)
		baseMenu.setParent(self)
		baseMenu.move(arg__1.globalPos()) # interesting that even if you
			# set the menu's parent to this widget,
			# you still have to move the menu to the event globalPos(), not local pos()
		baseMenu.show()


	def _setOptions(self, *args):

		a = EVAL(args[0])
		log("setOptions", args, a)
		if not isinstance(a, QtCore.QStringListModel):
			a = QtCore.QStringListModel(a)
		self.completer().setModel(a)

	def _setRawUiValue(self, value):
		self.setText(value)

	def _rawUiValue(self):
		return self.text()

	def _processValueForUi(self, rawValue):
		return toStr(rawValue)

	def _processResultFromUi(self, rawResult:str):
		"""
		here we do any dynamic eval if it's allowed
		"""
		if not EVAL(self.allowEval):
			return rawResult

		#log("eval-ing", rawResult, str(rawResult), f"{rawResult}")
		if not rawResult: # empty string, return empty result - by default a new init of whatever the current value type is
			return type(self.value())()

		# set syntax passes - for now only one to catch raw strings
		#TODO: rework a decent bit of the syntax passes, but it's finally
		#   WORKING IN SITU :D
		rawStrPass = SyntaxPasses.NameToStrPass()
		ensureLambdaPass = SyntaxPasses.EnsureLambdaPass()
		processor = ExpSyntaxProcessor(
			syntaxStringPasses=[rawStrPass, ensureLambdaPass],
			syntaxAstPasses=[rawStrPass, ensureLambdaPass]
		)
		result = processor.parse(rawResult, {})
		#log("parsed", result, type(result))
		# always returns a lambda function - more consistent that way
		result = result()

		# check if result type has to be coerced
		if self.forceResultType is None:
			return result

		forceResultType = EVAL(self.forceResultType)
		# check if it's already an instance of the type(s) given
		if isinstance(result, forceResultType):
			return result
		# try to coerce type one by one
		forceResultType = forceResultType if isinstance(forceResultType, tuple) else ( forceResultType, )
		for i in forceResultType:
			try:
				return i(result)
			except:
				continue
		raise TypeError(self, f"Cannot convert string eval'd value {result} to one of types {forceResultType}")


	#### TODO #####
	#       come back and tangle with the highlighting on option selection
	#           really make it sorry for talking back
	#               rub some dirt in it
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

	# DECLARING THE BELOW METHOD DOUBLE-TRIGGERS A TAB KEY PRESS ON QAPPLICATION
	# I do not know why this happens, and I assume I never will

	# def sizeHint(self):
	# 	size = self.fontMetrics().boundingRect(self.text()).size()
	# 	return size

		


class FileStringWidget(ExpWidget):
	"""
	we might declare this as
	path = PathWidget(parent=None,
	                  default="strategy.chi",
	                  parentDir=lambda: ref("asset").diskPath(),
	                  conditions=(),

	                  postProcessFn=lambda rawPath: Path(rawPath).relativeTo(self.parentDir())
	                  )

	some duplication of string widget above, but having one inherit from the
	other was a bit more annoying to wrangle layouts, size hints etc to account
	for the new button


	if parentDir is supplied, we only show a relative path?
	value is always absolute, displayed may be relative

	retrieved value may not be the same as that stored -
		if parent dir is specified


	"""

	Mode = FileBrowserButton.Mode

	def __init__(self, value="",
	             parent=None, name="",
	             parentDir=None,
	             fileSelectMode=FileBrowserButton.Mode.File,
	             fileMask="",
	             allowMultiple=False,
	             dialogCaption="",
	             showRelativeFromParent=True,
				pathCls=pathlib.Path,
	             placeHolderText=""
	             ):
		"""
		"""
		self.showRelativeFromParent = rx(showRelativeFromParent)
		self.pathCls = pathCls
		self.parentDir = rx(parentDir or "").rx.pipe(self.pathCls)
		self.fileSelectMode = rx(fileSelectMode)
		self.fileMask = rx(fileMask)
		self.allowMultiple = rx(allowMultiple)
		self.dialogCaption = rx(dialogCaption) if dialogCaption else \
			rx("Choose {}").format(self.fileMask.rx.pipe(str))

		ExpWidget.__init__(self, value, parent,
		                   placeHolderText=placeHolderText)

		# AtomicWidget.__init__( self, value=value, #name=name
		#                        )
		#self.line = QtWidgets.QLineEdit(self)
		# self.line = StringWidget(value=value, parent=self)

		# self.placeholderText = rx(placeHolderText)
		# self.placeholderText.rx.watch(self.line.setPlaceholderText)

		# add button to bring up file browser
		self.button = FileBrowserButton(parent=self,
		                                defaultBrowserPath=self.parentDir,
		                                mode=self.fileSelectMode,
		                                fileMask=self.fileMask,
		                                browserDialogCaption=self.dialogCaption,
		                                allowMultiple=self.allowMultiple

		                                )

		self.button.pathSelected.connect(self._onPathSelected)
		#self.postInit()
		# # connect signals
		# self.line.displayEdited.connect(self._fireDisplayEdited)
		# self.line.displayCommitted.connect(self._fireDisplayCommitted)

		# # layout
		# hl = QtWidgets.QHBoxLayout(self)
		# hl.addWidget(self.line)
		# self.setLayout(hl)
		# self.layout().setContentsMargins(1, 1, 1, 1)
		# self.layout().setSpacing(0)


		#self.layout().addWidget(self.button)

	def resizeEvent(self, event):
		self.setContentsMargins(0, 0, self.button.sizeHint().width(), 0)
		self.button.setGeometry(
			self.rect().width() - self.button.sizeHint().width(), 0,
			self.button.sizeHint().width(), self.button.sizeHint().height()
		)
		super().resizeEvent(event)

	def _onPathSelected(self, paths:list[pathlib.Path]):
		"""
		TODO: consider how to handle / display multiple paths
		 separate by semicolon? how do we push that through all the machinery
		"""
		log("_onPathSelected", paths)
		if not EVAL(self.allowMultiple):
			result = str(paths[0])
		else:
			result = " ; ".join(map(str, paths))

		log("result", result)

		# if relative, remove parentdir from path to display
		if EVAL(self.showRelativeFromParent):

			try:
				result = str(pathlib.Path(result).relative_to(EVAL(self.parentDir)))
			except Exception as e:
				log("error getting relative path")
				traceback.print_exc()
				result = str(result)
			# self.setValue(result)
			# return
		log("result", result)
		self._setRawUiValue(result)
		self._fireDisplayCommitted()


	def _tryCommitValue(self, value):
		if isinstance(value, (list, tuple)) and not EVAL(self.allowMultiple):
			raise RuntimeError(f"Widget {self} does not allow multiple files")
		super()._tryCommitValue(value)

	def _processResultFromUi(self, value:str):
		"""widget value is always directly what it says -
		parent directory only used to truncate results from path window.
		Matching relative widget paths up to parent dir has to be done
		as a separate process, too confusing otherwise


		"""
		if not value.strip():
			return None
		# check if there are semicolons for multiple files
		strPaths = [i.strip() for i in str(value).split(";")]

		# if EVAL(self.showRelativeFromParent): # don't extend value to global
		# 	paths = [EVAL(self.parentDir) / i for i in strPaths]
		# else:
		# 	paths = [self.pathCls(i) for i in strPaths]
		paths = [self.pathCls(i) for i in strPaths]

		if EVAL(self.allowMultiple):
			return paths
		return paths[0]

	def _processSinglePathForRelative(self, value):
		if EVAL(self.showRelativeFromParent):
			try:
				return self.pathCls(value).relative_to(EVAL(self.parentDir))
			except:
				return self.pathCls(value)
		return self.pathCls(value)

	def _processValueForUi(self, value)->str:
		if not value: return ""
		if isinstance(value, (tuple, list)):
			return " ; ".join(map(toStr, value))
		return toStr(value)

		# if isinstance(value, (tuple, list)):
		# 	paths = map(self._processSinglePathForRelative,
		# 	            value)
		# 	return " ; ".join(map(str, paths))
		# return str(self._processSinglePathForRelative(value))
		# value = wplib.sequence.toSeq(value)

		# return " ; ".join(map(str, value))



# class FileStringWidget(QtWidgets.QWidget, AtomicWidget):
# 	"""line edit with a file button next to it"""
#
# 	atomicType = FieldWidgetType.File
#
# 	def __init__(self, value=None, params:FieldWidgetParams=None, parent=None):
#
# 		QtWidgets.QWidget.__init__(self, parent)
# 		self.line = QtWidgets.QLineEdit(self)
# 		StringWidgetBase.__init__( self, value, params)
#
# 		self.line.textChanged.connect(self._onWidgetChanged)
#
# 		# layout
# 		hl = QtWidgets.QHBoxLayout(self)
# 		hl.addWidget(self.line)
# 		self.setLayout(hl)
# 		self.layout().setContentsMargins(1, 1, 1, 1)
# 		self.layout().setSpacing(0)
#
# 		# add button to bring up file browser
# 		self.button = FileBrowserButton(
# 			name="...", parent=self,
# 			defaultBrowserPath=self._params.defaultFolder,
# 			mode=self._params.fileSelectMode,
# 			fileMask=self._params.fileMask,
# 			browserDialogCaption=self._params.caption
# 			)
# 		self.layout().addWidget(self.button)
# 		self.postInit()
#
#
# 	def _rawUiValue(self):
# 		return self.line.text()
# 	def _setRawUiValue(self, value):
# 		self.line.setText(value)
# 	def _processUiResult(self, rawResult):
# 		return Path(rawResult)
# 	def _processValueForUi(self, rawValue):
# 		return str(rawValue)

if __name__ == '__main__':
	"""test initialising objects on rx"""

	# base = "a/b/c"
	# rxbase = rx(base)
	#
	# #path = pathlib.PurePath(rxbase) # errors
	# #print(path, type(path))
	# path = rxbase.rx.pipe(pathlib.PurePath)
	#
	# print(type(path), path.rx.value, type(path.rx.value))
	#

	app = QtWidgets.QApplication()

	v = "hello"
	v = rx(v)

	v.rx.watch(lambda x : print("i say", x))

	#w = StringWidget(value=v, parent=None)
	#w = FileStringWidget(value=v, parentDir="F:", allowMultiple=True)
	w = FileStringWidget(value=v, parentDir="F:",
	                     fileSelectMode=FileStringWidget.Mode.Dir,
	                     allowMultiple=True)
	# w = FileStringWidget(value=v, parentDir="F:",
	#                      fileSelectMode=FileStringWidget.Mode.Dir,
	#                      allowMultiple=False)
	w.show()
	app.exec_()






