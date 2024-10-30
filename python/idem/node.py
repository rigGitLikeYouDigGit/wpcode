

from __future__ import annotations

import pprint
import typing as T

from wplib import log, Sentinel, TypeNamespace

from chimaera import ChimaeraNode

class IdemGraph(ChimaeraNode):

	def getAvailableNodesToCreate(self)->list[str]:
		"""return a list of node types that this node can support as
		children - by default allow all registered types
		TODO: update this as a combined class/instance method
		"""
		return list(self.nodeTypeRegister.keys())

class DCCSessionNode(ChimaeraNode):
	"""DCC session nodes should show their own status,
	in the graph they can be dormant if their session isn't running
	"""
	@classmethod
	def prefix(cls) ->tuple[str]:
		return ("i", )
	@classmethod
	def canBeCreated(cls):
		return cls.__name__ != "DCCSessionNode"

class MayaSessionNode(DCCSessionNode):
	pass
	pass


