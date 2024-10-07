from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

import reactivex as rx

from wplib import log
from wplib.object import Signal

from wpdex import WpDexProxy, Reference
# from wpdex.ui.react import ReactiveWidget
from wpdex.ui.atomic.base import AtomicWidget

"""

"""


class OptionAtomicWidget(AtomicWidget):
	"""use this for enums, modes, etc -
	for searches, use a text widget and supply it options"""

	def _uiChangeQtSignals(self) -> list[QtCore.Signal]:
		return [self.stateChanged]

	def getValue(self) -> bool:
		return self.isChecked()

	def setValue(self, value, **kwargs):
		self.setChecked(value)