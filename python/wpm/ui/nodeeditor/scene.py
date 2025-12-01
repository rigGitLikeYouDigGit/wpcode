from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


from PySide6 import QtCore, QtWidgets, QtGui

from wpui.canvas import WpCanvasScene

if T.TYPE_CHECKING:
	from .graphics import QNodeItem


class QNodeEditorScene(WpCanvasScene):
	"""scene backing single node view - holds
	visibility list/expressions """
	
	def __init__(self, parent=None):
		
		super().__init__(parent)

		# list of names to save?
		self.visNameList = []
		# list of mobjects
		self.visObjList = []


	pass


