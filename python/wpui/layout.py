
from __future__ import annotations

import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

"""lib for qt layouts"""

def genAutoLayout(rootWidg:QtWidgets.QWidget, recurse=True):
	"""for iteration on tools - generates an automatic
	vertical layout for a widget and its children.

	Replace with proper layout setup once tool is stable.

	recurse doesn't work, and probably shouldn't -
	call this single-level function on children if you need it

	"""
	#vl = QtWidgets.QVBoxLayout(rootWidg)
	vl = QtWidgets.QVBoxLayout()
	#rootWidg.setLayout(vl)

	for i in rootWidg.children():
		if not isinstance(i, QtWidgets.QWidget):
			continue
		print("add w ", i)
		vl.addWidget(i)
	#rootWidg.setLayout(vl)
	return vl


