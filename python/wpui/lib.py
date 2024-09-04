
from __future__ import annotations
import typing as T

import numpy as np

from PySide2 import QtWidgets, QtCore, QtGui



def arrToQMatrix(arr:np.ndarray)->QtGui.QMatrix:
	"""convert numpy array to QMatrix"""
	return QtGui.QMatrix(arr[0, 0], arr[0, 1], arr[1, 0], arr[1, 1], arr[0, 2], arr[1, 2])

def qmatrixToArr(mat:QtGui.QMatrix)->np.ndarray:
	"""convert QMatrix to numpy array"""
	return np.array([[mat.m11(), mat.m12(), mat.dx()],
	                 [mat.m21(), mat.m22(), mat.dy()],
	                 [0, 0, 1]])
