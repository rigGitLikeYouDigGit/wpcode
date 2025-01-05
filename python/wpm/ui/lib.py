from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from maya import OpenMayaUI as omui1
from shiboken2 import wrapInstance
from ctypes import c_long
from PySide2 import QtCore, QtGui, QtWidgets

def getMainWindowWidget()->QtWidgets.QWidget:
	mainWindowPtr = omui1.MQtUtil.mainWindow()
	return wrapInstance(
		#long(main_window_ptr),
		int(mainWindowPtr),
	        QtWidgets.QWidget)
