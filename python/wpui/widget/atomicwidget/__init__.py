
from __future__ import annotations
from dataclasses import dataclass
import typing as T

from wplib.inheritance import iterSubClasses
from tree.lib.dict import enumKeyLookup
from tree.lib.path import PurePath

from PySide2 import QtCore, QtWidgets

from tree.lib.constant import AtomicWidgetSemanticType, AtomicWidgetType, FileSelectMode
from tree.ui.atomicwidget.constant import *

@dataclass
class AtomicWidgetParams:
	"""uniform parametre object passed to all atomic widgets
	not all fields need be filled"""
	type : AtomicWidgetType
	name : str = ""
	tooltip : str = ""
	min : (int, float) = None
	max: (int, float) = None
	default = None
	options : (tuple, list, dict, T.Type[Enum]) = None
	fileMask : str = None
	defaultFolder : (str, PurePath) = None
	fileSelectMode : FileSelectMode = FileSelectMode.File
	placeholder : str = ""
	caption : str = ""
	orient : Orient = Orient.Horizontal
	optionType : OptionWidgetType = OptionWidgetType.Radio
	if T.TYPE_CHECKING:
		Orient = Orient
		OptionType = OptionWidgetType


# add namespace attributes
AtomicWidgetParams.Orient = Orient
AtomicWidgetParams.OptionType = OptionWidgetType
AtomicWidgetParams.FileSelectMode = FileSelectMode

from tree.ui.atomicwidget.base import AtomicWidget
from tree.ui.atomicwidget.boolwidget import BoolWidgetBase, BoolCheckBoxWidget
from tree.ui.atomicwidget.optionwidget import OptionWidgetBase, OptionButtonWidget, OptionComboWidget, OptionStickyButton, OptionRadioButton
from tree.ui.atomicwidget.scalarwidget import ScalarWidgetBase, FloatSlider, IntSlider, FloatSpinBox, IntSpinBox
from tree.ui.atomicwidget.stringwidget import StringWidgetBase, StringWidget, FileStringWidget

widgetValueTypeMap = {
	AtomicWidgetSemanticType.Scalar : (FloatSlider, IntSlider, FloatSpinBox, IntSpinBox),
	AtomicWidgetSemanticType.Bool : (BoolCheckBoxWidget,),
	AtomicWidgetSemanticType.File : (FileStringWidget,),
	AtomicWidgetSemanticType.String : (StringWidget,),
}

widgetTypeMap : dict[AtomicWidgetType, T.Type[AtomicWidget]] = {
	i.atomicType : i for i in tuple(iterSubClasses(AtomicWidget))[1:] if i.atomicType is not None
}
# widgetTypeMap[AtomicWidgetType.ScalarSlider] = ScalarSliderWidget
# widgetTypeMap[AtomicWidgetType.ScalarSpinBox] = ScalarSpinBoxWidget

def atomicWidgetClsForKey(key:T.Union[Enum, str]):
	return enumKeyLookup(key, widgetTypeMap)

class WidgetNames:
	ScalarSlider = "scalarSlider"
	ScalarSpinBox = "scalarSpinBox"
	BoolCheckBox = "boolCheckBox"


def atomicWidgetForValue(
		value,
		minVal=None, maxVal=None,
		enum=None,
		parent=None,
		enumWidgetMode="radio",
		scalarWidgetMode="spinBox",
		orientation=QtCore.Qt.Horizontal

                         ):
	"""return an atomic widget suiting the given value
	using strings for modes as using enums for modes would
	mean excessive imports"""
	if isinstance(value, bool):
		return BoolCheckBoxWidget(value, parent)
	if isinstance(value, Enum):
		parentEnum = value.__objclass__
		if enumWidgetMode == "radio":
			return OptionButtonWidget(parentEnum, parentEnum.__name__,
			                          orientation=orientation,
			                          value=value,
			                          parent=parent
			                          )
		elif enumWidgetMode == "comboBox":
			return OptionComboWidget(parentEnum,
			                         value=value,
			                         parent=parent
			                         )
	if isinstance(value, (int, float)):
		if scalarWidgetMode == "spinBox":
			return ScalarSpinBoxWidget(value, minVal=minVal, maxVal=maxVal,
			                           parent=parent)
		elif scalarWidgetMode == "slider":
			return ScalarSpinBoxWidget(value, minVal=minVal, maxVal=maxVal,
			                           parent=parent)

	return None


def atomicWidgetForParams(params:AtomicWidgetParams)->(QtWidgets.QWidget, AtomicWidget):
	"""given parametre struct, return new atomic widget
	made way simpler by passing resultParams struct all the way to
	widget level"""
	widgetType = widgetTypeMap[params.type]
	return widgetType(value=params.value, params=params)


if __name__ == '__main__':

	class Parent:
		pass
	class Child(Parent):
		pass

	print(issubclass(Child, Parent))
	print(issubclass(Parent, Parent))




