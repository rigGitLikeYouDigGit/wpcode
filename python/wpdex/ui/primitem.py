from __future__ import annotations

import typing as T

from PySide2 import QtWidgets

from wplib import log

from wplib.constant import LITERAL_TYPES
from wpdex.primdex import PrimDex
from wpdex.ui.base import WpDexWidget, WpDexWindow, WpTypeLabel


class InnerWidget(QtWidgets.QWidget):

	def __init__(self, parent=None):
		super().__init__(parent)
		self.textbox = QtWidgets.QLineEdit("TEST", parent=self)
		self.spare = QtWidgets.QLabel("SPARE", parent=self)
		layout = QtWidgets.QHBoxLayout(self)
		layout.addWidget(self.textbox)
		layout.addWidget(self.spare)
		self.setLayout(layout)
		layout.setContentsMargins(0, 0, 0, 0)

		# self.textbox.setContentsMargins(0, 0, 0, 0)
		# self.spare.setContentsMargins(0, 0, 0, 0)


		self.setContentsMargins(0, 0, 0, 0)
		#self.setContentsMargins(50, 50, 50, 50)

		#self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)


class PrimDexWidget(WpDexWidget):
	"""widget for a literal -
	for now just a text box

	augment with a type label to keep track of a consistent type,
	or to change type if needed
	"""
	forTypes = (PrimDex, )

	def __init__(self, dex:PrimDex, parent=None):
		QtWidgets.QWidget.__init__(self, parent)
		#super().__init__()

		self._dex = dex

		self.w = InnerWidget(self)
		#self.setContentsMargins(0, 0, 0, 0)
		#self.setBaseSize(self.w.sizeHint())
		#self.setFixedSize(500, 100)


		# # self.textbox = QtWidgets.QLineEdit(
		# # 	self.dex().asStr(),
		# # 	parent=self)
		# self.textbox = QtWidgets.QLabel(#self.dex().asStr(),
		# 	"TEST",
		#                                 parent=self)
		# self.textbox.setAutoFillBackground(True)
		#
		# # completely disabling layout lets this work as an indexWidget
		#
		# layout = QtWidgets.QHBoxLayout()
		# # layout.addWidget(self.typeLabel)
		# layout.addWidget(self.textbox)
		#
		# self.spareW = QtWidgets.QLabel("SPARE", parent=self)
		# layout.addWidget(self.spareW)
		# #self.setLayout(layout)

		# # self.buildExtraWidgets()
		# # self.buildLayout()
		# # self.buildChildWidgets()
		# # self.syncItemLayout()
		# self.setAutoFillBackground(True)

	def sizeHint(self): # gets read correctly
		return self.w.sizeHint()

	def buildChildWidgets(self):
		pass

	def buildExtraWidgets(self):
		"""build and populate model and view"""
		#super().buildExtraWidgets()
		# self.typeLabel = WpTypeLabel(parent=self, closedText="[...]",
		#                              openText="[")
		self.textbox = QtWidgets.QLineEdit(
			self.dex().asStr(),
			parent=self)
		print("self as str", self.dex().asStr())

	def buildLayout(self):
		layout = QtWidgets.QHBoxLayout()
		# layout.addWidget(self.typeLabel)
		layout.addWidget(self.textbox)
		self.setLayout(layout)


