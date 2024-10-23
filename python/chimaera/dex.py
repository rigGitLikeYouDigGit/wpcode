
from __future__ import annotations

import pprint
import typing as T

from wplib import log
from wplib.delta import DeltaAtom
from wptree import Tree, TreeInterface, TreeDeltaAid, TreeDeltas

from chimaera.node import ChimaeraNode

from wpdex import WpDex, WpDexProxy
from wplib.delta import DeltaAtom, DeltaAid, SetValueDelta

class ChiDeltaAid(DeltaAid):
	forTypes = [ChimaeraNode]

class ChiDex(WpDex):

	forTypes = [ChimaeraNode]
	dispatchInit = True

	# don't cover aux data for now
	mutatingMethodNames = {
		"eyyyy"
	}

	obj : Tree

	def compareState(self, newDex:WpDex, baseDex:WpDex=None) ->(dict, list[DeltaAtom]):
		"""trees should recurse down into values for everything other than
		adding/removing branches and changing index"""
		deltas = super().compareState(newDex=newDex, baseDex=baseDex)
		return deltas

	# def _consumeFirstPathTokens(self, path:pathT) ->tuple[list[WpDex], pathT]:
	# 	"""process a path token"""
	# 	path = tuple(path)
	# 	#log("consume first tokens", self, path, path in self.keyDexMap, self.keyDexMap)
	# 	token, *path = path
	# 	# if isinstance(token, int):
	# 	# 	return [self.children[token]], path
	# 	return [self.branchMap()[token, )]], path

