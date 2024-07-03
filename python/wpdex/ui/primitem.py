from __future__ import annotations

import typing as T

from PySide2 import QtWidgets

from wplib import log

from wplib.constant import LITERAL_TYPES
from wpdex.primdex import PrimDex
from wpdex.ui.base import WpDexWidget, WpDexWindow, WpTypeLabel


class PrimDexWidget(WpDexWidget):
	"""widget for a literal -
	for now just a text box

	augment with a type label to keep track of a consistent type,
	or to change type if needed
	"""
	forTypes = (PrimDex, )

	def buildChildWidgets(self):
		pass

	def buildExtraWidgets(self):
		"""build and populate model and view"""
		self.textbox = QtWidgets.QLineEdit(
			self.dex().asStr(),
			parent=self)

	def buildLayout(self):
		layout = QtWidgets.QHBoxLayout()
		layout.addWidget(self.textbox)
		self.setLayout(layout)

	def setAppearance(self):
		self.layout().setContentsMargins(0, 0, 0, 0)
		#self.setContentsMargins(0, 0, 0, 0)
