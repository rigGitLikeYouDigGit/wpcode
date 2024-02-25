
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

"""
absolutely NO idea how to structure this,
rewrite completely as the urge takes you
"""

if T.TYPE_CHECKING:
	pass
	class SignalType:
		def __init__(self, *args:tuple[type]):
			self.args = args

		def connect(self, fn:T.Callable, *args, **kwargs):
			pass
		def emit(self, *args, **kwargs):
			pass
		def disconnect(self, *args, **kwargs):
			pass

	QtCore.Signal = SignalType


class Pressable:
	"""mix-in for pressable widgets
	for one-line connections to custom signals"""
	def __init__(self,
	             onPressed: T.Callable[[...], None]=None,
	             onRightPressed: T.Callable[[...], None]=None,
	             ):
		self.onPressed = onPressed
		self.onRightPressed = onRightPressed


class WpButton(QtWidgets.QPushButton, Pressable):
	"""push button with custom signals"""
	wpPressed = QtCore.Signal()
	wpRightPressed = QtCore.Signal()

	def __init__(self,
	             onPressed: T.Callable[[...], None]=None,
	             onRightPressed: T.Callable[[...], None]=None,
	             *args, **kwargs):
		QtWidgets.QPushButton.__init__(self, *args, **kwargs)
		Pressable.__init__(self, onPressed=onPressed, onRightPressed=onRightPressed)
		self.pressed.connect(self.wpPressed)
		self.wpPressed.connect(self.onPressed)
		self.wpRightPressed.connect(self.onRightPressed)

#

