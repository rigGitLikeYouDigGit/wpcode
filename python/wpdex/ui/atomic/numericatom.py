from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtGui, QtWidgets
from wpdex.ui.atomic.base import AtomicWidget
from wpdex import *


"""TODO:
get rid of all this, unify the primitive widgets with
line edit
"""
class IntWidget(AtomicWidget, QtWidgets.QSpinBox ):
	forTypes = ()


	def __init__(self, value=0, parent=None,
	             #options:T.Sequence[str]=(),
	             conditions:T.Sequence[AtomicWidget.Condition]=(),
	             warnLive=False,
	             light=False,
	             enableInteractionOnLocked=False,
	             placeHolderText=""
	             ):
		QtWidgets.QSpinBox.__init__(self, parent=parent)
		AtomicWidget.__init__(self, value=value,
		                      conditions=conditions,
		                      warnLive=warnLive,
		                      enableInteractionOnLocked=enableInteractionOnLocked
		                      )


		# connect signals
		self.valueChanged.connect(self._syncImmediateValue)
		#self.valueChanged.connect(self._fireDisplayEdited)
		self.editingFinished.connect(self._fireDisplayCommitted)
		#self.returnPressed.connect(self._fireDisplayCommitted)
		self.postInit()

	def _rawUiValue(self):
		return QtWidgets.QSpinBox.value(self)
	def _setRawUiValue(self, value):
		QtWidgets.QSpinBox.setValue(self, value)