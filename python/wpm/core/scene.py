from __future__ import annotations



from wplib.object import Signal

from .cache import om
from .callbackowner import CallbackOwner
"""scene-level objects and operations
for now use tree signals, upgrade to Qt if needed"""


class SceneListener(CallbackOwner):

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
		print("scene listener setup done")


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
	print("getSceneGlobals", obj)
	if obj is None:
		setupGlobals()
	return obj
