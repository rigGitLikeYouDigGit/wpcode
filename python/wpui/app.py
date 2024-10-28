from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtWidgets, QtGui

def handleMessage(msgType:QtCore.QtMsgType, msg:str):
	if "Qt: Untested Windows version 10.0 detected!" in msg:
		return


def muteWindowsVersionWarning(app:QtCore.QCoreApplication):
	QtCore.qInstallMessageHandler(handleMessage)
