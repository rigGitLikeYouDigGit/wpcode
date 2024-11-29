from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtGui, QtWidgets
from wplib.serial import serialise, deserialise
from wplib.inheritance import MetaResolver
from wpdex.ui.atomic.base import AtomicWidgetOld
from wpexp.tostr import toStr
from wpdex import *

# i summon ancient entities
from wpexp.syntax import SyntaxPasses, ExpSyntaxProcessor


class ExpWidget(MetaResolver, QtWidgets.QLineEdit, AtomicWidgetOld):
	"""general-purpose text edit to
	allow defining new values as whatever is eval'd
	on commit

	TODO: to use this more generally, maybe we go with textEdit instead?
		that way we can allow multi-line literals for nested lists/dicts, in
		situations where the full item view is overkill

	my approach to writing this ui code is similar to my approach to drawing.
	if you just draw every possible line, the picture will be in there somewhere.
	"""
	forTypes = (IntDex,
	            StrDex)


	def __init__(self, value=0, parent=None,
	             #options:T.Sequence[str]=(),
	             conditions:T.Sequence[AtomicWidgetOld.Condition]=(),
	             warnLive=False,
	             light=False,
	             enableInteractionOnLocked=False,
	             placeHolderText=""
	             ):
		QtWidgets.QLineEdit.__init__(self, parent=parent)
		AtomicWidgetOld.__init__(self, value=value,
		                         conditions=conditions,
		                         warnLive=warnLive,
		                         enableInteractionOnLocked=enableInteractionOnLocked
		                         )


		# connect signals
		self.textChanged.connect(self._syncImmediateValue)
		self.textEdited.connect(self._fireDisplayEdited)
		self.editingFinished.connect(self._fireDisplayCommitted)
		self.returnPressed.connect(self._fireDisplayCommitted)


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
		size = self.fontMetrics().boundingRect(self.text()).size()
		return size

if __name__ == '__main__':


	d = ["a", "b", "c"]
	p = WpDexProxy(d)
	app = QtWidgets.QApplication()
	w = QtWidgets.QWidget()
	expW = ExpWidget(parent=w,
	                 value=p.ref(1))
	btn = QtWidgets.QPushButton("display", parent=w)
	btn.clicked.connect(lambda :pprint.pprint(serialise(p)))
	w.setLayout(QtWidgets.QVBoxLayout())
	w.layout().addWidget(expW)
	w.layout().addWidget(btn)
	w.show()
	app.exec_()



