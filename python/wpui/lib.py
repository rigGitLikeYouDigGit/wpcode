
from __future__ import annotations
import typing as T

import numpy as np

from PySide2 import QtWidgets, QtCore, QtGui


class muteQtSignals:
	"""small context for muting qt signals around a block"""
	def __init__(self, obj:QtCore.QObject):
		self.obj = obj
	def __enter__(self):
		self.obj.blockSignals(True)
	def __exit__(self, exc_type, exc_val, exc_tb):
		self.obj.blockSignals(False)

def arrToQMatrix(arr:np.ndarray)->QtGui.QMatrix:
	"""convert numpy array to QMatrix"""
	return QtGui.QMatrix(arr[0, 0], arr[0, 1], arr[1, 0], arr[1, 1], arr[0, 2], arr[1, 2])

def qmatrixToArr(mat:QtGui.QMatrix)->np.ndarray:
	"""convert QMatrix to numpy array"""
	return np.array([[mat.m11(), mat.m12(), mat.dx()],
	                 [mat.m21(), mat.m22(), mat.dy()],
	                 [0, 0, 1]])
