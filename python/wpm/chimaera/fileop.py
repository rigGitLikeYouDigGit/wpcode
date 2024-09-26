


from __future__ import annotations
import typing as T

from pathlib import Path

from wptree import Tree

from wplib.inheritance import clsSuper

from ..chimaera import MayaOp


class FileOp(MayaOp):
	"""import a file into maya"""

	@classmethod
	def newFileEntry(cls, path:Path=None)->Tree:
		"""create a new file entry"""
		t = Tree("heading", value=[])
		t["mode"] = "import"
		return t

	@classmethod
	def getDefaultParams(cls, forNode:MayaOp)->Tree:
		"""returns a tree with default params -
		for each main heading, import a file, filter data branch from contents
		todo: update with import/export enum
		"""
		t : Tree = clsSuper(cls).getDefaultParams()
		t.addBranch(cls.newFileEntry())







