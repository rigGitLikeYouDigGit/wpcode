from __future__ import annotations

import traceback
import typing as T

from dataclasses import dataclass
from weakref import WeakValueDictionary, WeakSet

from PySide2 import QtCore, QtWidgets, QtGui

from param import rx

from wplib import log
from wplib.object import Signal, Adaptor

from wpdex import *

class Condition:
	"""EXTREMELY verbose, but hopefully this lets us reuse
	rules and logic as much as possible"""
	def __init__(self, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs
	def validate(self, value, *args, **kwargs):
		"""
		return None -> all fine
		raise error -> validation failed
		return (error, val) -> failed with suggested result
		"""
		raise NotImplementedError

	def correct(self, value):
		raise NotImplementedError

	@classmethod
	def checkConditions(cls, value, conditions:T.Sequence[Condition],
	                    *args, **kwargs):
		"""TODO: consider some kind of "report" object
		to collect the results of multiple conditions
		at once"""
		for i in conditions:
			i.validate(value, *args, **kwargs)


base = (Adaptor, )
if T.TYPE_CHECKING:
	base = (QtWidgets.QWidget, Adaptor)

base = object
if T.TYPE_CHECKING:
	base = QtWidgets.QWidget

class AtomicWidget(
	Adaptor,
	base

                   ):
	"""
	base class to formalise some ways of working with widgets to set and display data -

	CONSIDER - add events to hooks to re-check BIND connections, pull new
	values from rx connections, etc

	"""
	# atomic widgets registered as adaptors against dex types
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	forTypes : tuple[type[WpDex]] = ()

	Condition = Condition


	"""connect up signals and filter events in real widgets as
	needed to fire these - once edited and committed signals fire,
	logic should be uniform"""
	# widget display changed live
	displayEdited = QtCore.Signal(object)

	# widget display committed, enter pressed, etc
	# MAY NOT yet be legal, fires to start validation process
	displayCommitted = QtCore.Signal(object)

	# committed - FINAL value, no delta
	valueCommitted = QtCore.Signal(object) # redeclare this with proper signature
	# # checked - continuous as UI changes
	# atomValueChecked = QtCore.Signal(object)
	# # delta - emit before and after
	# atomValueDelta = QtCore.Signal(dict)

	#muteQtSignals = muteQtSignals

	valueType = object



	def __init__(self, value:valueType=None,
	             conditions:T.Sequence[Condition]=(),
	             warnLive=False,
	             commitLive=False,
	             enableInteractionOnLocked=False
	             ):
		"""
		if commitLive, will attempt to commit as ui changes, still ignoring
		any that fail the check -
		use this for fancy sliders, etc
		"""
		# set up all reactive elements first -
		# this is surely massive overkill but we live with it for now
		self._proxy : WpDexProxy = None
		self._dex : WpDex = None
		self._value : WX = rx(None)

		# child widgets of container may not be direct children in Qt
		self._childAtomics : dict[WpDex.keyT, AtomicWidget] = WeakValueDictionary()

		#self._value = rx(value) if not isinstance(value, rx) else value
		self._immediateValue = rx(None) # changes as ui updates


		self.conditions = conditions
		self.warnLive = warnLive

		self.setValue(value)

		self.displayEdited.connect(self._onDisplayEdited)
		self.displayCommitted.connect(self._onDisplayCommitted)

		self._value.rx.watch(self._syncUiFromValue, onlychanged=False)

		"""check that the given value is a root, and can be set -
		if not, disable the widget, since otherwise RX has a fit
		"""
		if not react.canBeSet(self._value):
			if not enableInteractionOnLocked:
				log(f"widget {self} passed value not at root, \n disabling for display only")
				self.setEnabled(False)

		self.setAutoFillBackground(True)

	def _syncImmediateValue(self, *args, **kwargs):
		#log("sync immediate")
		val = self._processResultFromUi(self._rawUiValue())
		#log("processed", val)
		self._immediateValue.rx.value = val

	def postInit(self):
		"""yes I know post-inits are terrifying in qt,
		for this one, just call it manually yourself,
		i'm not your mother
		"""
		#log("postInit", self, self.value())
		#log(self._processValueForUi(self.value()))
		self._syncUiFromValue()
		# try: self._syncUiFromValue()
		# except: pass
		self._syncImmediateValue()
		self.syncLayout()


	def _fireDisplayEdited(self, *args):
		"""TODO: find a way to stop double-firing this
			if you connect 2 signals to it
			"""
		self._syncImmediateValue()
		self.displayEdited.emit(args)

	def _fireDisplayCommitted(self, *args):
		self._syncImmediateValue()
		self.displayCommitted.emit(args)

	def _rawUiValue(self):
		"""return raw result from ui, without any processing
		so for a lineEdit, just text()"""
		raise NotImplementedError(self)

	def _setRawUiValue(self, value):
		"""set the final value of the ui to the exact value passed in,
		this runs after any pretty formatting"""
		raise NotImplementedError(self)

	def rxValue(self)->rx:
		return self._value
	def value(self)->valueType:
		"""return widget's value as python object"""
		return self._value.rx.value

	def setSourceObject(self, obj: (WpDex, WpDexProxy, WX, T.Any)):
		"""test working from a proxy as the source of everything"""
		if isinstance(obj, WpDex):
			proxy = WpDexProxy(obj.obj, wpDex=obj)

	def valueProxy(self) -> WpDexProxy:
		return self._proxy

	def dex(self) -> WpDex:
		return EVAL(self._dex)

	def setValue(self, value:(WpDex, WpDexProxy, WX, valueType)):
		"""
		set value on widget - can be called internally and externally
		if passed a reactive element, set up all children and local
		attributes

		TODO: we can't make a new type of widget from within this one -
			need an onChildValueChanged() function

		"""
		#log("set value", value, type(value), self)
		if not isinstance(value, (WpDex, WpDexProxy, WX)): # simple set value op
			if value == self.value():
				return
			self._tryCommitValue(value)
		if isinstance(value, WpDex):
			self._dex = value
			self._proxy = value.getValueProxy()
			self._value = self.dex().ref()
			self._tryCommitValue(value.obj)
		if isinstance(value, WpDexProxy):
			self._dex = value.dex()
			self._proxy = value
			self._tryCommitValue(value)
		if isinstance(value, WX): # directly from a reference
			self._dex = lambda : value.RESOLVE(dex=True)
			self._proxy = lambda : value.RESOLVE(proxy=True)
			# since it's an rx component, directly supplant the reactive value reference
			# self._value = value
			# self._value.rx.watch(self._syncUiFromValue, onlychanged=False)
			log("before commit wx", value, EVAL(value))
			self._tryCommitValue(EVAL(value))

	def _setChildAtomicWidget(self,
	                   key:WpDex.keyT,
	                   w:AtomicWidget
	                   ):
		"""update this widget's map of children, remove
		the existing widget if found
		also connects up signals"""
		if self._childAtomics.get(key):
			currentChild = self._childAtomics[key]
			currentChild.setParent(None)
			currentChild.deleteLater()
			self._childAtomics.pop(key)
		self._childAtomics[key] = w

		# connect signals
		# not passing widget to the lambda, unsure if that'll count as a reference and keep it around
		# for too long
		w.valueCommitted.connect(lambda obj: self._onChildAtomicValueChanged(key, obj))

		# extend in real class for adding to layout etc
		return w

	def _onChildAtomicValueChanged(self,
	                               key:WpDex.keyT,
	                               value:T.Any,
	                               ):
		"""check if new value needs new widget type -
		if so, remove widget at key and generate a new one

		cyclic link with _setAtomicChildWidget above, but I think it's ok"""
		log("on child atomic widget changed", key, value, self)
		#assert key in self._childAtomics, "no "

		oldDex = self._childAtomics[key].dex()
		newDexType = WpDex.adaptorForType(type(value))

		# if the old dex still fits the new type, nothing needs to be done
		if isinstance(oldDex, newDexType):
			return

		newChildWidget = self._makeNewChildWidget(key, value, newDexType)
		self._setChildAtomicWidget(key, newChildWidget)



		"""
		if new value type not compatible:
		new widget = adaptor for value
		self.seyChildAtomic(newWidget)
		"""
	def _makeNewChildWidget(self,
	                        key,
	                        value,
	                        newDexType):
		newWidgetType: type[AtomicWidget] = AtomicWidget.adaptorForType(newDexType)
		raise NotImplementedError


	def rxImmediateValue(self)->rx:
		return self._immediateValue
	def immediateValue(self):
		return self._immediateValue.rx.value

	def _processResultFromUi(self, rawResult):
		"""override to do any conversions from raw ui representation"""
		return rawResult

	def _processValueForUi(self, rawValue):
		"""any conversions from a raw value to a ui representation"""
		return rawValue

	def _onDisplayEdited(self, *args, **kwargs):
		"""connect to base widget signal to update live -
		an error thrown here shows up as a warning,
		maybe change text colours"""
		#self._immediateValue.rx.value = self._processResultFromUi(self._rawUiValue())
		if self.warnLive:
			self._checkPossibleValue(self._immediateValue.rx.value
				)

	def _onDisplayCommitted(self, *args, **kwargs):
		"""connect to signal when user is happy, pressed enter etc"""
		self._tryCommitValue(
			self._processResultFromUi(self._rawUiValue())
		)
		self.clearFocus()

	def _checkPossibleValue(self, value):
		try:
			Condition.checkConditions(value, self.conditions)
		except Exception as e:
			log(f"Warning from possible value {value}:")
			traceback.print_exception(e)
			return

	def _tryCommitValue(self, value):
		"""TODO:
			calling _commitValue() directly in this prevents us from
			using super() in inheritor classes before extending the checks -
			find a better way to split the validation step up"""
		try:
			Condition.checkConditions(value, self.conditions)
		except Exception as e:
			# restore ui from last accepted value
			self._syncUiFromValue()
			log(f"ERROR setting value {value}")
			traceback.print_exc()
			return
		self._commitValue(value)


	def _syncUiFromValue(self, *args, **kwargs):
		"""update the ui to show whatever the current value is
		TODO: see if we actually need to block signals here"""
		#log("syncUiFromValue", self)

		# block signals around ui update
		self.blockSignals(True)
		toSet = self._processValueForUi(self.value())
		#log("_sync", toSet, type(toSet))
		self._setRawUiValue(self._processValueForUi(self.value()))
		self.blockSignals(False)

	def _commitValue(self, value):
		"""run finally after any checks, update held value and trigger signals"""
		# if widget is expressly enabled, we catch the error from setting value
		# on a non-root rx
		canBeSet = react.canBeSet(self._value)

		#log("root", self._value._root is self._value, self._value._compute_root() is self._value)
		#log("can be set", canBeSet, self)
		if canBeSet:
			self._value.rx.value = EVAL(value) # rx fires ui sync function
		#log("after value set")
		self._syncImmediateValue()
		self.syncLayout()
		self.valueCommitted.emit(value)


	def focusOutEvent(self, event):
		"""ensure we don't get trailing half-finished input left
		in the widget if the user clicks off halfway through editing -
		return it to whatever we have as the value"""
		self._syncUiFromValue()
		super().focusOutEvent(event)

	def syncLayout(self, execute:bool=True):
		"""sync layout of the table"""
		#log("syncLayout", self)
		# for i in self._childAtomics.values():
		# 	i.syncLayout(execute=execute)
		self.update()
		self.updateGeometry()
		if isinstance(self, QtWidgets.QAbstractItemView):
			self.scheduleDelayedItemsLayout()
			if execute:
				self.executeDelayedItemsLayout()
		self.update()
		self.updateGeometry()
		if isinstance(self.parent(), AtomicWidget):
			self.parent().syncLayout(execute)


	# def sizeHint(self):
	#


