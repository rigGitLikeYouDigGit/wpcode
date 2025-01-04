from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtGui, QtWidgets

"""functions for pulling references to Maya's own qt widgets,
qapplication etc"""

from maya import utils
def getMayaQApp()->QtWidgets.QApplication:
	pass