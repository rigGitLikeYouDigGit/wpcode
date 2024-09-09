
from __future__ import annotations
"""unlikely to be necessary as Qt's default line edit options
are usually fine

only specialised for strings that can be files

also specialise for expressions, maybe, or wait for later class - 
mouse over widget to show:
- raw expression
- preprocessed expression
- available expression locals

"""

import inspect, pprint, dis
import typing as Ty
from enum import Enum

from PySide2 import QtCore, QtWidgets, QtGui

from tree.lib.path import Path, PurePath
from tree.lib.constant import FieldWidgetType, FileSelectMode, pathTypes
from tree.ui.fieldWidget import FieldWidgetParams
from tree.ui.fieldWidget.base import FieldWidget
from tree.ui.libwidget.filebutton import FileBrowserButton

class StringWidgetBase(FieldWidget):
	# def __init__(self, value=None, resultParams:FieldWidgetParams=None,
	#              # default=None, defaultFolder=None,
	#              # fileMask=None,
	#              # fileSelectMode=None
	#              ):
	# 	self.default = default
	# 	self.defaultFolder = defaultFolder
	# 	self.fileMask = fileMask
	# 	self.fileSelectMode=fileSelectMode
	# 	super(StringWidgetBase, self).__init__(value)
	# 	pass
	pass


class StringWidget(QtWidgets.QLineEdit, StringWidgetBase):
	"""simple line edit
	if resultParams define options, turn string widget into search widget

	"""

	atomicType = FieldWidgetType.String

	def __init__(self, value=None, params:FieldWidgetParams=None, parent=None):
		QtWidgets.QLineEdit.__init__(self, parent=parent)
		StringWidgetBase.__init__(self, value, params)
		self.textChanged.connect(self._onWidgetChanged)

		if self._params.placeholder:
			self.setPlaceholderText(self._params.default)

		self.postInit()

	def _rawUiValue(self):
		return self.text()
	def _setRawUiValue(self, value):
		self.setText(value)
	def _processUiResult(self, rawResult):
		return str(rawResult)
	def _processValueForUi(self, rawValue):
		return str(rawValue)


class FileStringWidget(QtWidgets.QWidget, StringWidgetBase):
	"""line edit with a file button next to it
	small duplications here but it's fine"""

	atomicType = FieldWidgetType.File

	def __init__(self, value=None, params:FieldWidgetParams=None, parent=None):

		QtWidgets.QWidget.__init__(self, parent)
		self.line = QtWidgets.QLineEdit(self)
		StringWidgetBase.__init__( self, value, params)

		self.line.textChanged.connect(self._onWidgetChanged)

		# layout
		hl = QtWidgets.QHBoxLayout(self)
		hl.addWidget(self.line)
		self.setLayout(hl)
		self.layout().setContentsMargins(1, 1, 1, 1)
		self.layout().setSpacing(0)

		# add button to bring up file browser
		self.button = FileBrowserButton(
			name="...", parent=self,
			defaultBrowserPath=self._params.defaultFolder,
			mode=self._params.fileSelectMode,
			fileMask=self._params.fileMask,
			browserDialogCaption=self._params.caption
			)
		self.layout().addWidget(self.button)
		self.postInit()


	def _rawUiValue(self):
		return self.line.text()
	def _setRawUiValue(self, value):
		self.line.setText(value)
	def _processUiResult(self, rawResult):
		return Path(rawResult)
	def _processValueForUi(self, rawValue):
		return str(rawValue)


def test():

	import sys
	app = QtWidgets.QApplication(sys.argv)
	win = QtWidgets.QMainWindow()

	def printValue(val):
		print("new string val", val)


	baseVal = __file__
	params = FieldWidget.paramCls(
		None, "testStringWidget",
		fileSelectMode=FileSelectMode.File
	)
	widg = FileStringWidget(value=baseVal, parent=win)
	widg.atomValueChanged.connect(printValue)
	print("w value", widg.atomValue())

	win.setCentralWidget(widg)
	win.show()
	sys.exit(app.exec_())

if __name__ == '__main__':
	test()

