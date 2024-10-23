
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log

class DCCSessionWidget(QtWidgets.QPushButton):
	"""widget to monitor a named, connected session of a program,
	or hook up an existing one
	keep palette off to side of session to show available programs

	"""

