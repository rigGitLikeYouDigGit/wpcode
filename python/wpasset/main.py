
from __future__ import annotations
"""main asset object, to be integrated with chimaera"""

from pathlib import Path
import typing as T
import json

from tree.lib.object import UidElement

from wp.constant import getAssetRoot
from .lib import tagsToIndexString, tagsToSemicolonString

class Asset(UidElement):
	"""this may also be used for components"""

	dirNameSep = "___"

	def __init__(self, uid:str=None, tags:dict={}):
		super(Asset, self).__init__(uid)
		self.tags = dict(tags) # tags used for searching

	def __repr__(self):
		return f"Asset({self.tags}, {self.uid})"

	def setTags(self, tags:dict):
		"""sets the tags"""
		self.tags = tags
		return self

	# def getFilePathOverride(self)->Path:
	# 	"""override this to return a special folder depth for this asset"""
	# 	return None

	# def getCombinedTagString(self)->str:
	# 	"""returns the combined tag keys and values as a string -
	# 	used as nice name in main folder dir. May change freely as those tags do.
	#
	# 	Use as unified search field"""
	# 	return "_".join(sorted([f"{k}-{v}" for k, v in self.tags.items()]))


	def getDirName(self)->str:
		"""returns the name of the directory"""
		assert self.tags
		return f"{tagsToIndexString(self.tags)}{self.dirNameSep}{self.uid}"

	# def getDirPath(self)->Path:
	# 	"""returns the path to the asset directory"""
	# 	return getAssetRoot() / self.getDirName()


	# def saveToPath(self):
	# 	"""saves the asset to dir path"""
	#
	# 	pass

	pass







