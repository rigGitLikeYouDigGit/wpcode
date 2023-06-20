from __future__ import annotations
import typing as T

from tree import Signal

from wpm import cmds, om, oma, WN, createWN


"""scene-level objects and operations
for now use tree signals, upgrade to Qt if needed"""


class CallbackOwner:
	"""Mixin for any objects creating Maya callbacks -
	delete callbacks when object is deleted"""

	def __init__(self):
		self._callbacks = []

	def addOwnedCallback(self, callback):
		"""add callback to list"""
		self._callbacks.append(callback)

	def removeOwnedCallback(self, id:int):
		"""remove callback from list"""
		om.MMessage.removeCallback(id)
		self._callbacks.remove(id)

	def removeAllOwnedCallbacks(self):
		"""remove all callbacks owned by this object"""
		om.MMesssage.removeCallbacks(self._callbacks)
		self._callbacks = []

	def __del__(self):
		self.removeAllOwnedCallbacks()

class SceneListener(CallbackOwner, ):

	def __init__(self):
		super(SceneListener, self).__init__()
		self.selectionChanged = Signal()
		self.sceneNameChanged = Signal()

	def setup(self):
		"""setup callbacks"""
		# selection change
		self.addOwnedCallback(om.MEventMessage.addEventCallback("SelectionChanged", self.selectionChanged.emit))

		# scene name / path change
		self.addOwnedCallback(om.MSceneMessage.addCallback(om.MSceneMessage.kAfterNew, self.sceneNameChanged.emit))
		self.addOwnedCallback(om.MSceneMessage.addCallback(om.MSceneMessage.kAfterOpen, self.sceneNameChanged.emit))
		self.addOwnedCallback(om.MSceneMessage.addCallback(om.MSceneMessage.kSceneUpdate, self.sceneNameChanged.emit))


class SceneGlobals:
	"""now I KNOW global objects are the first janky thing to
	complain about, but this is different
	"""

	def __init__(self):
		self.listener = SceneListener()

	def setup(self):
		"""setup scene globals"""
		self.listener.setup()

obj : SceneGlobals = None

def setupGlobals():
	"""run when wpm is loaded in maya"""
	global obj
	obj = SceneGlobals()
	obj.setup()

def getSceneGlobals()->SceneGlobals:
	"""return global object"""
	global obj
	if obj is None:
		setupGlobals()
	return obj
