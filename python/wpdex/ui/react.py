
from __future__ import annotations

import types
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log
from wplib.object import Signal

from wpdex import WpDexProxy, Reference
#from wpdex.ui.atomic.base import AtomicWidget
"""
could have these descriptors for show(), enabled() etc
"""
class WidgetHook:

	def __init__(self, w:QtWidgets.QWidget,
	             getFn:callable=None,
	             setFn:callable=None,
	             ):
		self.w = w
		self.getFn = getFn
		self.setFn = setFn
		self.uiChangedSignal = Signal(name="uiChanged")
		self.uiChangedSignal.connect(self.onUiChanged)

	def __repr__(self):
		return f"Hook({self.getFn}, {self.setFn}, {self.w})"

	def onUiChanged(self, *args, **kwargs):
		"""triggered whenever the ui changes -
		both on direct edit and on sync"""
		#log("onUIChanged", self, args, kwargs)

	def link(self, ref:Reference):
		"""set up bidirectional dependence between this
		ui field and the given reference -
		we can't insert any transformation here (unless it's reversible)
		"""
		# set to ref value first
		log("setting first")
		self.setFn( ref() )
		def onRefChanged(*args, **kwargs):
			log("on ref changed", self, args, kwargs)
			log("new value", ref() )
			self.setFn( ref() )
		# ref.dex().root.getEventSignal("main").connect(
		# 	lambda event : self.setFn( ref() ))
		ref.dex().root.getEventSignal("main").connect(onRefChanged)
		self.uiChangedSignal.connect(
			lambda *args : ref.dex().write( self.getFn() )
		                       )

	def driveWith(self, source):
		"""when hook cannot write back, for options, validation etc
		how do we signal to widgets to "sync", to pull new values to display?
		"""
		if isinstance(source, types.FunctionType):

			def onRefChanged(*args, **kwargs):
				pass

	# def connections(self)->dict[str, list[callable]]:
	#


base = object
if T.TYPE_CHECKING:
	base = QtWidgets.QWidget
class ReactiveWidget(base):


	def __init__(self, name:str=""):
		self.setObjectName(name)
		self.ENABLED = WidgetHook(
			self, getFn=self.isEnabled, setFn=self.setEnabled	)
		self.VISIBLE = WidgetHook(
			self, getFn=self.isVisible, setFn=self.setVisible		)
		self.VALUE = WidgetHook(
			self, getFn=self.getValue, setFn=self.setValue)

		# connect signals to drive
		for i in self._uiChangeQtSignals():
			i.connect(self.onUiChanged)
		#TODO: the old agony of Qt widget init sequencing might strike here -
		#  it shouldn't matter when hooks are created though

	def __repr__(self):
		return f"{type(self)}({self.objectName()})"

	def _uiChangeQtSignals(self)->list[QtCore.Signal]:
		"""return the qt signals of this widget to connect to the
		onUiChanged() manager function"""
		raise NotImplementedError

	def hooks(self)->dict[str, WidgetHook]:
		return {k : v for k, v in self.__dict__.items() if isinstance(v, WidgetHook)}

	def onUiChanged(self, *args, **kwargs):
		"""top level trigger for any ui change -
		this will trigger regardless of source, so either filter
		or mute	signals outside this scope where needed
		"""
		if self.sender() is self:
			log("skipping box", self)
		for hookName, hook in self.hooks().items():
			hook.uiChangedSignal(*args, **kwargs)

	# uniform interface for getting and setting values on common widgets
	def getValue(self, **kwargs):
		raise NotImplementedError
	def setValue(self, value, **kwargs):
		raise NotImplementedError

	def getOptions(self):
		raise NotImplementedError
	def setOptions(self, value, **kwargs):
		raise NotImplementedError