
from enum import Enum

class OptionWidgetType(Enum):
	Radio = "radio"
	Sticky = "sticky"
	Menu = "menu"

class Orient(Enum):
	Horizontal = "horizontal"
	Vertical = "vertical"