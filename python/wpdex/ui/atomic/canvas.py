
from __future__ import annotations
import typing as T, types

from PySide2 import QtCore, QtGui, QtWidgets

from wptree import Tree
from wpui.canvas import *
from wplib.inheritance import resolveInheritedMetaClass, MetaResolver
from wpdex import *
from wpdex import WX
from wpdex.ui.atomic.base import AtomicUiInterface



class AtomicCanvasScene(
	MetaResolver,
	WpCanvasScene,
	AtomicUiInterface,
):
	"""base class for showing graphical displays representing
	reactive objects

	This isn't too useful to have as a reactive widget,
	since you normally only set the whole scene at program startup
	"""

	def __init__(self, value=None,
	             parent=None,
	             conditions=(),
	             warnLive=False,
	             commitLive=False,
	             enableInteractionOnLocked=False,
	             **kwargs):
		WpCanvasScene.__init__(self, parent)
		AtomicUiInterface.__init__(
			self,
			value, conditions, warnLive, commitLive, # turn ALL of this into rules good grief
			enableInteractionOnLocked,
			**kwargs
		                           )

	def _setRawUiValue(self, value):
		"""basically just run the paint function"""
		if value is None:
			self.clear()


class AtomicCanvasElement(
	MetaResolver,
	WpCanvasElement,
	AtomicUiInterface
):
	"""Atomic entity in a QGraphics scene -
	more useful than the scene above
	DON'T FORGET elements still need to inherit from an actual
	QtGraphics item in final class
	"""
	def __init__(self, value=None,
	             parent=None,
	             **kwargs):
		WpCanvasElement.__init__(self, obj=value)
		AtomicUiInterface.__init__(
			self,
			value,
			**kwargs
		                           )

if __name__ == '__main__':
	app = QtWidgets.QApplication()
	s = AtomicCanvasScene()
	print(s)


