
from __future__ import annotations
import typing as T
"""main asset object, to be integrated with chimaera
we go back to the original folder-based idea -
if a folder contains a file named "_asset.json", it's an asset
"""

from pathlib import Path, PurePath
import orjson

from wplib.object import UidElement, SmartFolder, DiskDescriptor
from wplib.sequence import toSeq, flatten

from wp.constant import getAssetRoot, WP_ROOT
from wp.pipe.show import Show
from .lib import tagsToIndexString, tagsToSemicolonString

defaultAssetData = {
	"tags" : {},
	"created" : None,
	"version" : {"latest" : 0},
	#"uid" : None
}

class AssetFolder(SmartFolder):
	inDir = DiskDescriptor("_in", create=False)
	outDir = DiskDescriptor("_out", create=True)
	workDir = DiskDescriptor("_work", create=True)
	assetData = DiskDescriptor("_asset.json", create=True,
	                           default=defaultAssetData)

	createMissingFilesOnInit = False


class Asset(UidElement):
	"""this may also be used for components

	"""

	@classmethod
	def tokensFromDirPath(cls, path):
		return Path(str(Path(path)).split(str(WP_ROOT))[-1]).parts[1:]

	def __init__(self, pathTokens:list, uid=None,
	             parentAsset:Asset=None):
		"""parent only used transiently to prepend tokens"""
		super().__init__(uid)
		if parentAsset:
			pathTokens = parentAsset.tokens + pathTokens
		self.tokens = list(pathTokens)
		self._folder = None
		self._data = None

	def exists(self)->bool:
		return self.diskPath().exists()

	def __repr__(self):
		return "<Asset({})>".format(self.path())

	def __truediv__(self, other:(str, list)):
		"""allow going
		Asset(dave) / "head" -> Asset(dave / head)
		"""

		newTokens = self.tokens + flatten([i.split("/") for i in toSeq(other)])
		return Asset(pathTokens=newTokens)

	def showName(self)->str:
		return self.tokens[0]

	def show(self)->Show:
		return Show.get(name=self.showName())

	def smartFolder(self):
		if self._folder is None:
			self._folder = AssetFolder(path=self.diskPath())
		return self._folder

	def diskPath(self)->Path:
		"""feels like muddying, maybe asset shouldn't know its
		direct placement on disk, should go through bank to map
		a show path to real disk.

		then again the folders power everything at the moment"""
		tokens = self.tokens
		if tokens[0] == self.show().name: tokens = tokens[1:]
		return self.show().path() / "/".join(tokens)

	def path(self)->str:
		return "/".join(self.tokens)

	@classmethod
	def isAssetDir(cls, path:Path):
		return ( path / "_asset.json" ).exists()

	def parent(self)->(None, Asset):
		if self.isAssetDir(self.diskPath().parent):
			return Asset(self.tokens[:-1])
		return None

	def create(self):
		"""set up all missing folders and files for a new asset"""
		self.smartFolder().createMissingFiles()

	# store any metadata about asset like tags, uid, etc
	# TODO: do we really want to mess around with live saving / loading to disk when dict changes
	def data(self)->dict:
		if self._data is None:
			self._data = orjson.loads(self.smartFolder().assetData.read_text())
		return self._data
	def saveData(self, data=None):
		self.smartFolder().assetData.write_text(orjson.dumps(data or self._data))

	def tags(self)->dict:
		return self.data()["tags"]











