from __future__ import annotations
import typing as T


from PySide2 import QtCore, QtWidgets, QtGui

from wplib.object import VisitAdaptor
from wpdex import WpDex
from wptree import Tree
from wpui import lib

class WidgetVisitAdaptor(VisitAdaptor):

	@classmethod
	def childObjects(cls, obj:T.Any, params:PARAMS_T) ->CHILD_LIST_T:
		"""return only widgets that have a name"""
		result = []
		for k, v in lib.widgetChildMap(obj, includeObjects=False).items():
			data = VisitAdaptor.ChildData(
				key=k, obj=v, data={}
			)
			result.append(data)
		return result



class WidgetDex(WpDex):
	"""allowing traversal through named qwidgets with paths -
	MAYBE extend to any QObjects, but widgets are enough for now"""
	forTypes = (QtWidgets.QWidget, )




