from __future__ import annotations

import typing as T

from PySide2 import QtWidgets

from wplib import log

from wplib.constant import LITERAL_TYPES
from wpdex.strexpdex import StrDex
from wpdex.ui.base import WpDexView, WpDexWindow, WpTypeLabel


class StrDexWidget(WpDexView):
	"""widget for strings - might get super complex
	"""
	forTypes = (StrDex, )

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
