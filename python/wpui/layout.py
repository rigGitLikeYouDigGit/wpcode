
from __future__ import annotations

import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

"""lib for qt layouts
TODO: should we just move this into the big ui lib file?
"""

def genAutoLayout(rootWidg:QtWidgets.QWidget, recurse=True):
	"""for iteration on tools - generates an automatic
	vertical layout for a widget and its children.

	Replace with proper layout setup once tool is stable.

	recurse doesn't work, and probably shouldn't -
	call this single-level function on children if you need it

	"""
	#vl = QtWidgets.QVBoxLayout(rootWidg)
	vl = QtWidgets.QVBoxLayout(rootWidg)
	#rootWidg.setLayout(vl)

	for i in rootWidg.children():
		if not isinstance(i, QtWidgets.QWidget):
			continue
		#print("add w ", i)
		vl.addWidget(i)
		# if recurse: # check for any widgets that don't already have a layout?
		# 	pass
	rootWidg.setLayout(vl)
	return vl


seqObj = QtWidgets.QWidget
seqType = (seqObj, list[seqObj])
def literalLayout(literalWidgetSeq:T.Sequence[QtWidgets.QWidget]):
	"""
	absolutely deranged test for drawing widget layout literally in code

	layout = QtWidgets.QGridLayout()
	layoutSpacing = [
		(   [ self.typeLabel    ],  [ self.superView, 1],       ),
		(   0,                      [ 1,              1],       ),
		(   0,                      0,                0,      ),
		(   [ self.superItem,       1,                1],       ),
		(   [ 1,                    1,                1],       ),
	]

	I still hold a candle for this , but I have no idea how
	to make it work
	"""




