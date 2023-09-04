
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

keyDict = {v : k for k, v in QtCore.Qt.__dict__.items() if "Key_" in k}

dropActionDict = {v : k for k, v in QtCore.Qt.DropAction.__dict__.items()
                  if "Action" in k}

