from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import random

from PySide6 import QtCore, QtGui, QtWidgets

def randomTuple(seed=None)->tuple[float, float, float]:
	return (random.random(), random.random(), random.random())

def randomColour(seed=None):
	return QtGui.QColor.fromRgbF(*randomTuple(seed))

def pastelColourInSequence(
		index=0, startOffset=10.0, coloursPerRotation=5.0
):
	""""""

w = QtWidgets.QWidget()
w.setPalette(QtGui.QPalette())


