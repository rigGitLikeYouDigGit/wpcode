from __future__ import annotations
import typing as T

from collections import defaultdict

from PySide2 import QtCore, QtGui, QtWidgets

from wpui.constant import keyDict, dropActionDict

class KeyState(object):
	""" holds variables telling if shift, LMB etc are held down
	currently requires events to update, may not be a good idea to
	query continuously """

	class _BoolRef(object):
		""" wrapper for consistent references to bool value """
		def __init__(self, val):
			self._val = val
		def __call__(self, *args, **kwargs):
			self._val = args[0]
		def __str__(self):
			return str(self._val)
		def __repr__(self):
			return "<BoolRef({})>".format(self._val)
		def __nonzero__(self):
			return self._val
		def __bool__(self):
			return self._val

	# attach relevant qt constants
	lmbKey = QtCore.Qt.LeftButton
	rmbKey = QtCore.Qt.RightButton
	mmbKey = QtCore.Qt.MiddleButton
	altKey = QtCore.Qt.Key_Alt
	shiftKey = QtCore.Qt.Key_Shift
	ctrlKey = QtCore.Qt.Key_Control


	def __init__(self, mouseTrackLength=3):
		self.LMB = self._BoolRef(False)
		self.RMB = self._BoolRef(False)
		self.MMB = self._BoolRef(False)
		self.alt = self._BoolRef(False)
		self.ctrl = self._BoolRef(False)
		self.shift = self._BoolRef(False)

		self.lastMousePosMap  = {
			QtCore.Qt.LeftButton : (0, 0),
			QtCore.Qt.RightButton : (0, 0),
			QtCore.Qt.MiddleButton : (0, 0),
		} # ordered dict where latest mouse position is first

		self.mouseMap = {
			self.LMB : QtCore.Qt.LeftButton,
			self.RMB : QtCore.Qt.RightButton,
			self.MMB : QtCore.Qt.MiddleButton }

		self.keyMap = {
			self.alt: QtCore.Qt.AltModifier,
			self.ctrl: QtCore.Qt.ControlModifier,
			self.shift: QtCore.Qt.ShiftModifier,
		}
		# shift and ctrl are swapped for me I kid you not
		# I swear to god they actually are

		self.mouseTrackLength = mouseTrackLength

		# add position tracking for mouse
		self.mousePositions = self.initialiseMouseTrackList()

		# test - register functions to be called when key is pressed
		self.keyFunctionMap :dict[QtCore.Qt.Key, set[T.Callable]] = defaultdict(set)

	def initialiseMouseTrackList(self)->list[QtCore.QPoint]:
		return [QtCore.QPoint() for i in range(self.mouseTrackLength)]

	def holdDuration(self):
		"""not sure how to best do this - consider on mouse press / key press,
		set a timer going to increment map values?"""
		pass

	def lastMouseClickPos(self, forButton=None):
		if forButton:
			return self.lastMousePosMap[forButton]
		return tuple(self.lastMousePosMap.values())[-1]

	def registerKeyEventFunction(self, keyPool:tuple[QtCore.Qt.Key], fn:T.Callable,
	                             onlyForObject=None):
		"""register the given function to be called when one of the keys in keyPool is pressed
		optionally restrict only to when the given widget is active

		no support for only specific combinations of keys - check for modifiers
		in called functions if needed
		"""
		for key in keyPool:
			if onlyForObject is not None: # add object's id to key
				key = (key, id(onlyForObject))
			self.keyFunctionMap[key].add(fn)

	def dispatchKeyEventFunctions(self, event:QtCore.QEvent, forObject=None,
	                              pressed=True):
		"""no guarantee of order in dispatched functions
		this will NOT be called automatically by this object - call it yourself from client event functions
		since you'll need to redefine them all anyway"""
		connectedFunctions = self.keyFunctionMap.get(event.key(), set())
		if forObject is not None:
			connectedFunctions.update(self.keyFunctionMap.get( (event.key(), id(forObject)), set()))
		for fn in connectedFunctions:
			fn(event, pressed)


	def mousePressed(self, event:QtGui.QMouseEvent):
		for button, v in self.mouseMap.items():
			button( event.button() == v)
		# update last position
		self.lastMousePosMap.pop(event.button())
		self.lastMousePosMap[event.button()] = event.pos()
		self.syncModifiers(event)

	def mouseReleased(self, event:QtGui.QMouseEvent):
		for button, v in self.mouseMap.items():
			if event.button() == v:
				button(False)
		self.syncModifiers(event)
		# clear mouse tracking if no mouse buttons are pressed
		if not any((self.LMB, self.MMB, self.RMB)):
			self.mousePositions = self.initialiseMouseTrackList()

	def mouseMoved(self, event:QtGui.QMouseEvent):
		for i in range(len(self.mousePositions) - 1):
			self.mousePositions[-(i + 1)] = self.mousePositions[-(i + 2)]
		self.mousePositions[0] = event.pos()

	def mouseDelta(self)->QtCore.QPoint:
		return self.mousePositions[0] - self.mousePositions[1]

	def keyPressed(self, event:QtGui.QKeyEvent):
		self.syncModifiers(event)

	def keyReleased(self, event:QtGui.QKeyEvent):
		self.syncModifiers(event)


	def eventKeyNames(self, event):
		return keyDict.get(event.key())

	def syncModifiers(self, event):
		""" test each individual permutation of keys
		against event """
		for key, v in self.keyMap.items():
			key((event.modifiers() == v)) # not iterable
		if event.modifiers() == (QtCore.Qt.ShiftModifier | QtCore.Qt.ControlModifier):
			self.ctrl(True)
			self.shift(True)


	def debug(self):
		print(self.mouseMap)
		print(self.keyMap)
