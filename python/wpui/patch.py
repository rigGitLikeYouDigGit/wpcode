

from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

"""
instead of full type patching, try instead patching individual
objects to add in mixin-type behaviour?
signals directly from mouse click, mouse hover etc?
"""

# tests for patching ps2 types for quality of life
def _patchSetContentsMargins(layout:QtWidgets.QLayout,
                             *args):
	if len(args) == 1:
		width = args[0]
		layout.setContentsMargins(width, width, width, width)
	layout.setContentsMargins(*args)
def patchLayout():
	"""patch the layout class"""
	# patch
	QtWidgets.QLayout.setContentsMargins = _patchSetContentsMargins


