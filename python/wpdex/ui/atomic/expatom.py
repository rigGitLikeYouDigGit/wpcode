from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtGui, QtWidgets
from wplib.inheritance import MetaResolver
from wpdex.ui.atomic.base import AtomicWidget, toStr
from wpdex import *

# i summon ancient entities
from wpexp.syntax import SyntaxPasses, ExpSyntaxProcessor


class ExpWidget(MetaResolver, QtWidgets.QLineEdit, AtomicWidget):
	"""general-purpose text edit to
	allow defining new values as whatever is eval'd
	on commit
	"""
	forTypes = (IntDex,
	            StrDex)


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
		if not rawResult: # empty string, return empty result
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
		return result()

	def _processValueForUi(self, rawValue):
		return toStr(rawValue)

	def sizeHint(self)->QtCore.QSize:
		# size = self.fontMetrics().size(
		# 	QtCore.Qt.TextWordWrap, self.text())
		#log("sizeHint for", self.text(), " : ", size)
		size = self.fontMetrics().tightBoundingRect(self.text()).size()
		return size