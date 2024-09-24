

from __future__ import annotations
import typing as T
from enum import Enum

if T.TYPE_CHECKING:
	from wptree.interface import TreeInterface

from wplib.object.reference import ObjectReference

"""
More robust way of referring to tree branches, when the objects themselves may be
deleted and recreated, or moved around in the tree.

"""


class TreeReference(ObjectReference):
	"""use to reliably resolve a certain branch
	in a tree - especially when using transform chains

	used as a component to pass-through tree proxies

	TODO: is this REALLY needed, can't we just store the path
	 and let pathable do everything
	"""
	class Mode(Enum):
		Uid = "uid"
		RelPath = "relativePath"
		AbsPath = "absolutePath"

	def __init__(self, branch:TreeInterface=None, relParent=None, mode=Mode.Uid):
		self.mode = mode
		self.uid = branch.uid
		relParent = relParent or branch.root
		self.address = branch.relAddress(fromBranch=relParent)

	def __repr__(self):
		return f"<TreeRef({self.address}, uid={self.uid[:5]}., mode={self.mode})"

	def resolve(self, relativeParent:TreeInterface=None)->TreeInterface:
		from wptree import Tree
		if self.mode == self.Mode.Uid:
			return Tree.uidInstanceMap[self.uid]
		elif self.mode == self.Mode.RelPath:
			return relativeParent(self.address, create=False)

	def __copy__(self):
		from wptree import Tree
		newRef = TreeReference(Tree("temp"),
		                       relParent=None, mode=self.mode)
		newRef.uid = self.uid
		newRef.address = self.address
		return newRef
