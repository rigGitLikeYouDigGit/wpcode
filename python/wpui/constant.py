
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

keyDict = {v : k for k, v in QtCore.Qt.__dict__.items() if "Key_" in k}

dropActionDict = {v : k for k, v in QtCore.Qt.DropAction.__dict__.items()
                  if "Action" in k}

tabKeys = (QtCore.Qt.Key_Tab, QtCore.Qt.Key_Backtab)
enterKeys = (QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return)
deleteKeys = (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace)
shiftKeys = (QtCore.Qt.Key_Shift, ) # no qt equivalent for left/right shift
escKeys = (QtCore.Qt.Key_Escape,)
spaceKeys = (QtCore.Qt.Key_Space, )



