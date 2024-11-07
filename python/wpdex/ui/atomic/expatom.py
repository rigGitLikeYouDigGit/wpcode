from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtGui, QtWidgets
from wpdex.ui.atomic.base import AtomicWidget
from wpdex import *


def toStr(x):
	"""TODO: obviously
	consider the full send - to be robust to different ways
	of representing objects, could define a new hierarchy of
	string-adaptor for each kind of expression syntax, and
	implement each object type bespoke

	"""
	if isinstance(x, WpDexProxy):
		return toStr(x._proxyTarget())
	return str(x)



class ExpWidget(QtWidgets.QLineEdit, AtomicWidget):
	"""general-purpose text edit to
	allow defining new values as whatever is eval'd
	on commit
	"""
	forTypes = (IntDex, )


	def __init__(self, value=0, parent=None,
	             #options:T.Sequence[str]=(),
	             conditions:T.Sequence[AtomicWidget.Condition]=(),
	             warnLive=False,
	             light=False,
	             enableInteractionOnLocked=False,
	             placeHolderText=""
	             ):
		QtWidgets.QLineEdit.__init__(self, parent=parent)
		AtomicWidget.__init__(self, value=value,
		                      conditions=conditions,
		                      warnLive=warnLive,
		                      enableInteractionOnLocked=enableInteractionOnLocked
		                      )


		# connect signals
		self.textChanged.connect(self._syncImmediateValue)
		self.textEdited.connect(self._fireDisplayEdited)
		self.editingFinished.connect(self._fireDisplayCommitted)
		self.returnPressed.connect(self._fireDisplayCommitted)
		self.postInit()

	def _rawUiValue(self):
		return self.text()
	def _setRawUiValue(self, value):
		self.setText(value)

	def _processResultFromUi(self, rawResult):
		"""TODO: add in syntax passes here for
		    raw string values without quotes
		"""
		#log("eval-ing", rawResult, str(rawResult), f"{rawResult}")
		if not rawResult:
			return type(self.value())()
		return eval(rawResult)
	def _processValueForUi(self, rawValue):
		return toStr(rawValue)

	def sizeHint(self)->QtCore.QSize:
		# size = self.fontMetrics().size(
		# 	QtCore.Qt.TextWordWrap, self.text())
		#log("sizeHint for", self.text(), " : ", size)
		size = self.fontMetrics().tightBoundingRect(self.text()).size()
		return size