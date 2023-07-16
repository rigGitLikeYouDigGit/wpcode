
from __future__ import annotations
import typing as T

import copy

from dataclasses import dataclass

from PySide2 import QtWidgets, QtCore, QtGui

from wp.constant import Orient
from wplib.sentinel import Sentinel
from wp.object import NodeOverrider

"""when in doubt, make more mixin classes

Abstract base for a ui widget holding / displaying a value, 
optionally editing it.

Value may be copied statically into widget, and retrieved as a newly
constructed copy, or widget may modify a python object live in place.

We probably need some unified type processing for copying objects -
always more to do


For control and appearance, either we do a load of complicated overrides,
OR we inject EVERYTHING.

Whatever functions you pass to this has to take care of any possible setting
for UI's behaviour.

Apparently flat is better than nested

"""

def copyObject(obj:T.Any)->T.Any:
	"""temp function for now - this
	should be properly integrated into larger system later"""
	return copy.deepcopy(obj)


class ValueUiElement(
	NodeOverrider
):
	"""hold, display, edit and retrieve a value.
	Inherit this class into an actual widget class when needed."""

	# valueCopyFn = Overrider.OverrideProperty(
	# 	"valueCopyFn", default=copyObject) # shared copy function
	#
	# uiOrient = Overrider.OverrideProperty(
	# 	"uiOrientation", default=Orient.Horizontal)

	def getOverrideInputs(self) ->dict[str, NodeOverrider]:
		"""return inputs to override -
		test going around the graph structure and directly
		returning parent object"""
		return {"main" : self.parentValueWidget()}

	def __init__(self, value:T.Any=Sentinel.Empty, copyValue:bool=True):
		"""value is the value to hold, copyValue determines if we copy
		the value into the widget, or just hold a reference to it.

		Holding separate copy of value not necessary when copying, since
		we must be able to recover the value from the ui anyway"""

		NodeOverrider.__init__(self)

		self._copyValue = copyValue
		self._internalValue = None

		if value is not Sentinel.Empty:
			self.setElementValue(value)

	def parentValueWidget(self)->ValueUiElement:
		"""return parent value widget, if any"""
		if isinstance(self, QtCore.QObject):
			assert isinstance(self.parent(), ValueUiElement)
			return self.parent()
		raise NotImplementedError

	def shouldCopyValue(self)->bool:
		"""return True if value should be copied into widget -
		any widget that copies a value must necessarily
		copy values for all its children as well."""
		return self.parentValueWidget().shouldCopyValue() or self._copyValue


	def _updateUiFromValue(self, value):
		"""update ui from value - don't emit any signals"""
		raise NotImplementedError


	def _recoverValueFromUi(self)->T.Any:
		"""retrieve value from ui - don't emit any signals.
		used when value can be copied directly"""
		raise NotImplementedError

	def setElementValue(self, value:T.Any):
		"""set value of element"""
		if self._copyValue:
			self._internalValue = self.valueCopyFn(value)
		else:
			self._internalValue = value
		self._updateUiFromValue(value)

	def getElementValue(self)->T.Any:
		"""retrieve value from element - either new copy or reference to
		original value, depending on copyValue flag"""
		if self._copyValue:
			return self._recoverValueFromUi()
		else:
			return self._internalValue

	def childValueWidgets(self)->ValueUiElement:
		"""return child value widgets, if any"""
		return []




