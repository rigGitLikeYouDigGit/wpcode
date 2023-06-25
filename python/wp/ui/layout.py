
from __future__ import annotations

import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

"""lib for qt layouts"""

def autoLayout(rootWidg:QtWidgets.QWidget, recurse=True):
	"""for iteration on tools - generates an automatic
	vertical layout for a widget and its children.

	Replace with proper layout setup once tool is stable.

	"""
	vl = QtWidgets.QVBoxLayout(rootWidg)
	for i in rootWidg.children():
		if not isinstance(i, QtWidgets.QWidget):
			continue
		vl.addWidget(i)
		if recurse:
			autoLayout(i, recurse=True)
	rootWidg.setLayout(vl)
	return vl


