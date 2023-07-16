
from __future__ import annotations
import typing as T

from dataclasses import dataclass

from tree import Tree, TreeInterface

from PySide2 import QtWidgets, QtCore, QtGui


"""base class allowing tree indexing into QObjects - 
no further implementation here"""


bases = (TreeInterface, QtCore.QObject)
if not T.TYPE_CHECKING:
	bases = (TreeInterface,)

class QObjectTree(TreeInterface,
                  QtCore.QObject
                  ):

	"""base class allowing tree indexing into QObjects -
	no further implementation here"""

	def _getBranchesInternal(self) ->T.List[QObjectTree]:
		"""return list of branches"""
		return self.children()

	def name(self) ->str:


