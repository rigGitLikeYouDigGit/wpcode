
from __future__ import annotations
import typing as T

import numpy as np



from wplib.object import TypeNamespace, Adaptor

from PySide2 import QtCore, QtWidgets, QtGui

"""label to show the type of a piece of data - 
display a string representation of the type of the data,
and a unique colour

todo: add menu to change type to others, depending on
context, depending on slice, etc

 
"""

# adaptors for appearance of type information -
# in time use this to specify available actions, conversions etc
# move this higher if used elsewhere

class UiDataTypeAdaptor(Adaptor):
	"""base adaptor for ui data type
	"""
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	forTypes = ()
	@classmethod
	def charsForObj(cls, obj:T.Any)->tuple[str, str]:
		"""get the characters to use as bookends for this item -
		"[", "]" for lists, "{", "}" for dicts, etc
		"""
		return (type(obj).__name__, "")

	@classmethod
	def rgbForObj(cls, obj:T.Any)->tuple[int, int, int]:
		"""get the rgb colour for this object
		"""
		return (128, 128, 128)



class DefaultDataTypeAdaptor(UiDataTypeAdaptor):
	"""default adaptor for ui data type
	"""
	forTypes = (object, )



class IntDataTypeAdaptor(UiDataTypeAdaptor):
	"""adaptor for int
	"""
	forTypes = (int, )
	@classmethod
	def rgbForObj(cls, obj:T.Any) ->tuple[int, int, int]:
		return (64, 64, 128)

	@classmethod
	def charsForObj(cls, obj:T.Any)->tuple[str, str]:
		return ("i", "")

class FloatDataTypeAdaptor(UiDataTypeAdaptor):
	"""adaptor for float
	"""
	forTypes = (float, )
	rgb = (128, 64, 64)
	chars = ("f", "")

class StrDataTypeAdaptor(UiDataTypeAdaptor):
	"""adaptor for str
	"""
	forTypes = (str, )
	rgb = (128, 128, 64)
	chars = ("s", "")


# container adaptors
class TupleDataTypeAdaptor(UiDataTypeAdaptor):
	"""adaptor for tuple
	"""
	forTypes = (tuple, )
	rgb = (128, 64, 128)
	chars = ("(", ")")

class ListDataTypeAdaptor(UiDataTypeAdaptor):
	"""adaptor for list
	"""
	forTypes = (list, )
	rgb = (64, 128, 128)
	@classmethod
	def charsForObj(cls, obj:T.Any)->tuple[str, str]:
		return ("[", "]")

class SetDataTypeAdaptor(UiDataTypeAdaptor):
	"""adaptor for set
	"""
	forTypes = (set, )
	rgb = (128, 128, 128)
	chars = ("set(", ")")

class DictDataTypeAdaptor(UiDataTypeAdaptor):
	"""adaptor for dict
	"""
	forTypes = (dict, )
	rgb = (128, 128, 128)
	chars = ("{", "}")

# test defining more complex types
# class MatrixDataTypeAdaptor(UiDataTypeAdaptor):
# 	"""adaptor for matrix
# 	"""
# 	forTypes = (np.ndarray, )
# 	rgb = (64, 64, 128)
# 	chars = ("m", "")

# test using a single datatype for all np arrays
class NpArrDataTypeAdaptor(UiDataTypeAdaptor):
	"""adaptor for np arrays
	"""
	forTypes = (np.ndarray, )
	rgb = (64, 64, 128)
	chars = ("[[", "]]")


class TypeLabel(QtWidgets.QLabel):
	"""visual label showing type of a value, and providing
	type-specific actions when clicked

	this shouldn't hold any reference to the object,
	let owner manage that

	"""

	def __init__(self,
	             uiTypeAdaptor:UiDataTypeAdaptor,
	             forObj:T.Any,
	             parent=None, ):
		"""create a type label -
		we read the colour and text once, then just keep it
		"""
		self.typeAdaptor = uiTypeAdaptor
		chars = self.typeAdaptor.charsForObj(forObj)
		#print("chars", chars)
		super().__init__(chars[0], parent)

	@classmethod
	def forObj(cls, forObj:T.Any, parent=None)->TypeLabel:
		"""create a type label for an object
		"""
		adaptor = UiDataTypeAdaptor.adaptorForType(type(forObj))
		#print("adaptor", adaptor, "for", forObj, type(forObj))
		return cls(adaptor, forObj, parent)







