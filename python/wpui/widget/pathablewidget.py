
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.pathable import Pathable


class PathableWidget( Pathable
                     ):
	"""allow traversing a ui widget structure by paths
	"""
	forTypes = [QtWidgets.QWidget]
	dispatchInit = False

	def __init__(self, obj, parent, key):
		super().__init__(obj, parent=parent, key=key)
