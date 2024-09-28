from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log

"""
abc for uniform widget interface for getting and setting values
"""


class AtomicWidget:
	"""MAYBE it's worth turning fields like value, options etc
	into generated descriptors or something

	i say "etc" but I can't think of any real uses outside those
	"""

	def getValue(self):
		raise NotImplementedError
	def setValue(self, value, **kwargs):
		raise NotImplementedError