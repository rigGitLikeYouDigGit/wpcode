

from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

import reactivex as rx

from wplib import log
from wplib.object import Signal

from wpdex import WpDexProxy, Reference
#from wpdex.ui.react import ReactiveWidget
from wpdex.ui.atomic.base import AtomicWidget
from wpui.widget import Lantern
"""



"""

class FileStringWidget(QtWidgets.QWidget, AtomicWidget):
	"""
	we might declare this as
	path = PathWidget(parent=None,
	                  default="strategy.chi",
	                  parentDir=lambda: ref("asset").diskPath(),
	                  conditions=(),

	                  postProcessFn=lambda rawPath: Path(rawPath).relativeTo(self.parentDir())
	                  )
	issue of pre- and post-process functions to format values for display,

	"""

	def __init__(self, parent=None, name="",
	             default=None,
	             parentDir=None

	             ):
		QtWidgets.QWidget.__init__(parent)

		self.line = QtWidgets.QLineEdit(self)
		AtomicWidget.__init__( self, name=name)

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

	def _uiChangeQtSignals(self):
		return [self.line.textEdited]


class FileStringWidget(QtWidgets.QWidget, AtomicWidget):
	"""line edit with a file button next to it"""

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
