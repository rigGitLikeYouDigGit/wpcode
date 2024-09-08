
"""base class for value widgets providing consistent
interfaces -
value(), setValue(), atomValueChanged signal

no labels directly contained here
"""
from __future__ import annotations
from PySide2 import QtCore, QtWidgets, QtGui
#from tree.signal import Signal

from tree.ui.lib import muteQtSignals
from tree.lib.constant import AtomicWidgetType
from tree.ui.atomicwidget import AtomicWidgetParams


orientMap = {
	AtomicWidgetParams.Orient.Horizontal :
		QtCore.Qt.Horizontal,
	AtomicWidgetParams.Orient.Vertical :
		QtCore.Qt.Vertical,

}

class AtomicWidget(
	#QtCore.QObject
	object
):
	"""base semi-abstract class for atomic widget family -
	NB these are UI ONLY. these classes have no view to maintaining
	abstract python values as a ground truth, only to providing
	a consistent interface to interact with widgets.

	ground truth and pure-python integration is left to
	later objects, that can probably hook into any atomicWidget class

	we attempt to extract the simplest control flow for what a UI element
	should do to keep track of its value

	setAtomValue() -> main entrypoint for owning code - may emit atom signals, but should not emit base qt widget signals

	"""
	paramCls = AtomicWidgetParams
	# promote qt namespace members
	Horizontal = QtCore.Qt.Horizontal
	Vertical = QtCore.Qt.Vertical

	valueType = None
	# atomValueChanged : QtCore.Signal = None # redeclare this with proper signature
	atomValueChanged = QtCore.Signal(object) # redeclare this with proper signature

	# signature {"oldValue" : old value, "newValue" : new value}
	atomValueDelta = QtCore.Signal(dict)

	atomicType : AtomicWidgetType = None

	defaultOrient = Vertical

	muteQtSignals = muteQtSignals

	def __init__(self, value:valueType=None, params:AtomicWidgetParams=None):
		self._value = value
		self._prevValue = None
		self._params = params or AtomicWidgetParams(self.atomicType)


	def postInit(self):
		if self._value is not None:
			self.setAtomValue(self._value)
		self._onWidgetChanged()



	def _getParamsQtOrient(self)->QtCore.Qt.Orientation:
		orient = self._params.orient if self._params.orient is not None else self.defaultOrient
		if isinstance(orient, self._params.Orient):
			orient = orientMap[orient]
		return orient

	def default(self):
		return self._params.default

	def _rawUiValue(self):
		"""return raw result from ui, without any representation"""
		raise NotImplementedError

	def _setRawUiValue(self, value):
		raise NotImplementedError


	def _processUiResult(self, rawResult):
		"""override to do any conversions from raw ui representation"""
		return rawResult

	def _processValueForUi(self, rawValue):
		"""any conversions from a raw value to a ui representation"""
		return rawValue

	def atomValue(self)->valueType:
		"""return widget's value as python object"""
		return self._value

	def setAtomValue(self, value:valueType):
		"""set value on widget - can be called internally and externally"""
		#print("setAtomValue", value, self.atomValue())
		if value == self.atomValue():
			return
		oldValue = self.atomValue()
		self._value = value
		self.atomValueChanged.emit(value)
		self.atomValueDelta.emit({
			"oldValue" : oldValue,
			"newValue" : value
		})
		# block signals around ui update
		self.blockSignals(True)
		self._matchUiToValue(value)
		self.blockSignals(False)

	def _matchUiToValue(self, value):
		"""update ui to match given value"""
		self._setRawUiValue(self._processValueForUi(value))

	def _onWidgetChanged(self, *args, **kwargs):
		"""connect to base widget signal to update"""
		self.setAtomValue(self._processValueForUi(self._rawUiValue()))
