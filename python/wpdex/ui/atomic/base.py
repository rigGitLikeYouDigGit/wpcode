from __future__ import annotations

import traceback
import typing as T

from dataclasses import dataclass

from PySide2 import QtCore, QtWidgets, QtGui

from param import rx

from wplib import log
from wplib.object import Signal

from wpdex import WpDexProxy, react

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


base = object
if T.TYPE_CHECKING:
	base = QtWidgets.QWidget

class AtomicWidget(base):
	"""
	base class to formalise some ways of working with widgets to set and display data -

	CONSIDER - add events to hooks to re-check BIND connections, pull new
	values from rx connections, etc

	"""
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
		self._value = rx(value) if not isinstance(value, rx) else value
		self.conditions = conditions
		self.warnLive = warnLive

		self.displayEdited.connect(self._onDisplayEdited)
		self.displayCommitted.connect(self._onDisplayCommitted)

		self._value.rx.watch(self._syncUiFromValue)

		"""check that the given value is a root, and can be set -
		if not, disable the widget, since otherwise RX has a fit
		"""
		if not self._value._compute_root() is self._value:
			if not enableInteractionOnLocked:
				log(f"widget {self} passed value not at root, \n disabling for display only")
				self.setEnabled(False)


	def postInit(self):
		"""yes I know post-inits are terrifying in qt,
		for this one, just call it manually yourself,
		i'm not your mother
		"""
		try: self._syncUiFromValue()
		except: pass


	def _fireDisplayEdited(self, *args):
		self.displayEdited.emit(args)

	def _fireDisplayCommitted(self, *args):
		self.displayCommitted.emit(args)

	def _rawUiValue(self):
		"""return raw result from ui, without any processing
		so for a lineEdit, just text()"""
		raise NotImplementedError

	def _setRawUiValue(self, value):
		"""set the final value of the ui to the exact value passed in,
		this runs after any pretty formatting"""
		raise NotImplementedError

	def rxValue(self):
		return self._value

	def value(self)->valueType:
		"""return widget's value as python object"""
		return self._value.rx.value

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
		if self.warnLive:
			self._checkPossibleValue(
				self._processResultFromUi(self._rawUiValue()))

	def _onDisplayCommitted(self, *args, **kwargs):
		"""connect to signal when user is happy, pressed enter etc"""
		self._tryCommitValue(
			self._processResultFromUi(self._rawUiValue())
		)

	def _checkPossibleValue(self, value):
		try:
			Condition.checkConditions(value, self.conditions)
		except Exception as e:
			log(f"Warning from possible value {value}:")
			traceback.print_exception(e)
			return

	def _tryCommitValue(self, value):
		try:
			Condition.checkConditions(value, self.conditions)
		except Exception as e:
			# restore ui from last accepted value
			self._syncUiFromValue()
			log(f"ERROR setting value {value}")
			traceback.print_exception(e)
			return
		self._commitValue(value)


	def _syncUiFromValue(self, *args, **kwargs):
		"""update the ui to show whatever the current value is
		TODO: see if we actually need to block signals here"""
		# block signals around ui update
		self.blockSignals(True)
		self._setRawUiValue(self._processValueForUi(self.value()))
		self.blockSignals(False)

	def _commitValue(self, value):
		"""run finally after any checks, update held value and trigger signals"""
		# if widget is expressly enabled, we catch the error from setting value
		# on a non-root rx
		if react.canBeSet(self._value):
			self._value.rx.value = value # rx fires ui sync function
		#self._syncUiFromValue()
		self.valueCommitted.emit(value)


	def setValue(self, value:valueType):
		"""
		set value on widget - can be called internally and externally

		"""
		#print("setAtomValue", value, self.atomValue())
		if value == self.value():
			return
		self._tryCommitValue(value)

	def focusOutEvent(self, event):
		"""ensure we don't get trailing half-finished input left
		in the widget if the user clicks off halfway through editing -
		return it to whatever we have as the value"""
		self._syncUiFromValue()
		super().focusOutEvent(event)

