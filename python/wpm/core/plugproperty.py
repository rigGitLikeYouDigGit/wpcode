
from __future__ import annotations
"""test for a small object allowing prediction when
accessing common attributes on nodes"""

from edRig.palette import *

if T.TYPE_CHECKING:
	from edRig.maya.core.node import EdNode

class PlugDescriptor:

	def __init__(self):
		pass

	def __get__(self, instance:EdNode, owner):
		"""access node's plug tree with defined key"""

class TriplePlugDescriptor:
	"""things come in 3s"""

