
from __future__ import annotations
import typing as T

from dataclasses import dataclass

from PySide2 import QtWidgets, QtCore, QtGui

from tree import Tree

from wp import constant

"""tests for ways to control complex generated widget hierarchies.
For now we use trees - investigate a more user-friendly way if needed,
but raw trees are fine for now
"""

#

# @dataclass
# class UiParams:
# 	"""dataclass for ui parameters"""
# 	widgetForValueFn:T.Callable[[T.Any], QtWidgets.QWidget] = None
# 	copyValue:bool = True
#
# 	def asTree(self)->Tree:
# 		"""return ui params as a tree"""
# 		tree = Tree("root")
# 		tree.lookupCreate = True
# 		tree["widgetForValueFn"] = self.widgetForValueFn
# 		tree["copyValue"] = self.copyValue



