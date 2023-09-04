
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets


class muteQtSignals:
	"""small context for muting qt signals around a block"""
	def __init__(self, obj:QtCore.QObject):
		self.obj = obj
	def __enter__(self):
		self.obj.blockSignals(True)
	def __exit__(self, exc_type, exc_val, exc_tb):
		self.obj.blockSignals(False)
