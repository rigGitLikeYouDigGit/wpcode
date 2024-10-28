from __future__ import annotations
import types, typing as T
import pprint
from wplib import log




"""hold any modifications we need to do to create a common
dark theme for wp tools
"""
from PySide2 import QtCore, QtWidgets, QtGui
import qdarkstyle

# extern, export, out variable STYLE_SHEET to apply to applications


def applyStyle(app:QtWidgets.QApplication,
               styleSheet="dark"):
	if styleSheet == "dark":
		STYLE_SHEET = qdarkstyle.load_stylesheet(qt_api="pyside2")
	app.setStyleSheet(STYLE_SHEET)