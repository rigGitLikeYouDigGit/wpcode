
"""base class for value widgets providing consistent
interfaces -
value(), setValue(), atomValueChanged signal

no labels directly contained here

FIELDS subclassed from tree branches
also allow SETFIELD() to target arbitrary paths relative to
a given root with wpdex

"""
from __future__ import annotations

import typing
import typing as T
from PySide2 import QtCore, QtWidgets, QtGui

from wplib import TypeNamespace, Adaptor
from wptree import Tree, TreeInterface

from wpui.lib import muteQtSignals

from wpui.fieldwidget.constant import FieldWidgetType
#from wpui.fieldWidget import FieldWidgetParams

"""
# setField to set any bit of an object to be active
def setField(parent:T.Any, key:(str, list[str]),
             params=None,
             setMethod=lambda parent, *args: parent.__setitem__(key, *args),
             ):
	pass
"""
#class

# orientMap = {
# 	FieldWidgetParams.Orient.Horizontal :
# 		QtCore.Qt.Horizontal,
# 	FieldWidgetParams.Orient.Vertical :
# 		QtCore.Qt.Vertical,
#
# }


if not T.TYPE_CHECKING:
	parent = object
if T.TYPE_CHECKING:
	parent = QtCore.QObject
class FieldWidget(
	parent, Adaptor
):
	"""
	rewrite on old bones, finally tying the path ideas together -
	a fieldWidget is given a Root object (usually a tree) and a Key path
	to what it should display / edit - SURELY THIS MEANS we have to pull parametres from those defined on the target field

	"""
	# adaptor setup to associate types of widget with values
	# can be overridden by rules
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	dispatchInit = False

	# paramCls = FieldWidgetParams

	# define param flags for this widget
	# class ParamDict(typing.TypedDict):
	# 	default
	@classmethod
	def getBaseParams(cls):
		"""define keys and structure for params"""
		return {}

	# promote qt namespace members
	Horizontal = QtCore.Qt.Horizontal
	Vertical = QtCore.Qt.Vertical

	valueType = None
	atomValueChanged = QtCore.Signal(object) # redeclare this with proper signature

	# signature {"oldValue" : old value, "newValue" : new value}
	# atomValueDelta = QtCore.Signal(dict) # don't care about deltas at ui level


	defaultOrient = Vertical

	muteQtSignals = muteQtSignals

	def __init__(self, value:valueType=None, params:dict=None):
		self._value = value
		self._prevValue = None
		self._params = params or self.getBaseParams()

	def trySet(self):
		pass

	def postInit(self):
		if self._value is not None:
			self.setAtomValue(self._value)
		self._onWidgetChanged()



	# def _getParamsQtOrient(self)->QtCore.Qt.Orientation:
	# 	orient = self._params.orient if self._params.orient is not None else self.defaultOrient
	# 	if isinstance(orient, self._params.Orient):
	# 		orient = orientMap[orient]
	# 	return orient

	def default(self):
		return self._params.get("default")

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
