
from enum import Enum

class ConstraintType(Enum):
	Point = "point"
	Hinge = "hinge"

class BindState(Enum):
	Off = 0
	Bind = 1
	Bound = 2
	Live = 3