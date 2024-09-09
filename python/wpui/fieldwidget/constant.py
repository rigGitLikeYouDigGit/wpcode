
from enum import Enum

from PySide2 import QtCore, QtWidgets, QtGui

# ui-focused constants

class FileSelectMode(Enum):
	"""should options look for a file or for a directory"""
	Directory = "directory"
	File = "file"

# type enums for atomic widgets
UI_PROPERTY_KEY = "_uiSettings"
EXPRESSION_PROPERTY_KEY = "_expression"
class FieldWidgetType(Enum):
	BoolCheckBox = "boolCheckBox"
	BoolCheckButton = "boolCheckButton"
	OptionRadio = "optionRadio"
	OptionMenu = "optionMenu"
	File = "file"
	ScalarSpinBox = "scalarSpinBox"
	ScalarSlider = "scalarSpinBox"
	String = "string"

class FieldWidgetSemanticType(Enum):
	Scalar = "scalar"
	Bool = "bool"
	File = "file"
	String = "string"
	Option = "option"

class OptionWidgetType(Enum):
	Radio = "radio"
	Sticky = "sticky"
	Menu = "menu"

class Orient(Enum):
	Horizontal = "horizontal"
	Vertical = "vertical"


orientMap = {
	Orient.Horizontal :	QtCore.Qt.Horizontal,
	Orient.Vertical : QtCore.Qt.Vertical,

}
