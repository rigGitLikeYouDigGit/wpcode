from __future__ import annotations

import json, glob
from pathlib import Path

from whoosh.fields import Schema, ID, KEYWORD
from whoosh.index import Index
from whoosh.qparser import QueryParser, AndGroup
from whoosh import index as indexModule

from wp import constant
from .lib import tagsToIndexString
from wpasset.main import Asset

# match these tags against a config
validAssetTags = {
	"character" : {
		"cait", "ben", "james", "josh", "matt", "michael", "mitch", "nick", "sam", "sean", "tom"
	},
	"part" : {
		"body", "head"
		}
}

# build schema for searching assets
assetSchema = Schema(
	uid=ID(stored=True),
	tags=KEYWORD(stored=True)
	#dirPath=STORED()
	# tags=TEXT(stored=True,
	#           phrase=False,
	#           #analyzer=IDTokenizer(), # don't split tags
	#           ),
)
# # add valid tags
# for key, validValues in validAssetTags.items():
# 	assetSchema.add(key, KEYWORD(stored=True))


class AssetBank:
	"""main class managing asset system"""

	def __init__(self):
		self.assets : dict[str, Asset] = {}
		self.index : Index = None

	def internalDataDir(self)->Path:
		"""returns the path to the internal data files for the bank itself"""
		return constant.getAssetRoot() / "_internal"

	def internalSearchDataDir(self)->Path:
		"""returns the path to whoosh index dir"""
		return self.internalDataDir() / "search"

	def assetDirByUid(self, uid:str)->Path:
		"""returns the path to the directory containing the asset with the given uid
		glob for uid, ignore any preceding tags in name"""
		result = glob.glob(constant.getAssetRoot() / ("*" + Asset.dirNameSep + uid))
		return result[0] if result else None

	def assetFromUid(self, uid:str)->Asset:
		"""returns the asset with the given uid
		if asset has been loaded, return the object
		if not, try to load it from disk
		if not, raise KeyError"""
		try:
			return self.assets[uid]
		except KeyError:
			pass
		# try to load it from disk
		dirPath = self.assetDirByUid(uid)
		try:
			asset = self.loadAssetFromDir(dirPath)
			self.assets[uid] = asset
			return asset
		except FileNotFoundError:
			raise KeyError(f"no asset with uid {uid}")

	def getSearchIndex(self)->Index:
		"""returns the whoosh index -
		if needed, loads it and sets it on this object"""

		if self.index is not None:
			return self.index

		# check if folder exists
		if not self.internalSearchDataDir().exists():
			self.internalSearchDataDir().mkdir(parents=True)
		# check if valid index exists
		if not indexModule.exists_in(self.internalSearchDataDir()):
			# create index
			index = indexModule.create_in(self.internalSearchDataDir(), assetSchema)
		else:
			index = indexModule.open_dir(self.internalSearchDataDir())
		self.index = index
		return self.index


	def _addAssetsToIndex(self, assets:list[Asset]):
		"""adds the given asset to the whoosh index"""
		writer = self.getSearchIndex().writer()
		for i in assets:
			writer.add_document(
				uid=i.uid,
				tags=tagsToIndexString(i.tags),

			)

		writer.commit()

	def addAssets(self, assets:list[Asset]):
		"""adds the given asset to the bank"""
		self._addAssetsToIndex(assets)
		for i in assets:
			self.assets[i.uid] = i

	def searchAssetsByTags(self, tags:dict, exact=False)->(list[Asset], Asset):
		"""searches the asset bank for the given query"""
		# combine tags into string
		tagStr = tagsToIndexString(tags)
		if len(tags)==1:
			tagStr = "'" + tagStr + "'" # prevent parsing on single pair
		print("tagstr", tagStr)
		# build query
		#qp = SimpleParser("tags", schema=self.getSearchIndex().schema, group=AndGroup)
		qp = QueryParser("tags", schema=self.getSearchIndex().schema, group=AndGroup)
		q = qp.parse(tagStr)
		#q = And(Term("tags", tagStr ))

		with self.getSearchIndex().searcher() as searcher:
			if not exact: # return all or none found
				results = searcher.search(q)
				print("results", results, *results)
				return [self.assetFromUid(i["uid"]) for i in results]
			# return only those that match all tags, raise KeyError if none found
			results = searcher.search(q, limit=None, )
			print("results", results, *results)
			if not results:
				raise KeyError(f"no exact result for {tagStr}")
			for i in results:
				if set(i["tags"].split(" ")) == set(tagStr.split(" ")):
					return self.assetFromUid(i["uid"])
			raise KeyError((f"no exact asset found for {tagStr}",
			                f"got {*results,}"))


	def dirPathForAsset(self, asset:Asset)->Path:
		"""returns the path to the dir for the given asset"""
		return constant.getAssetRoot() / asset.getDirName()

	def dataFileForAsset(self, asset:Asset)->Path:
		"""returns the path to the data file for the given asset"""
		return self.dirPathForAsset(asset) / "_data.json"

	def saveAssetToFile(self, asset:Asset):
		data = {
			"uid": asset.uid,
			"tags": asset.tags,
			"dirPath": self.dirPathForAsset(asset).as_posix(),
		}

		if not self.dirPathForAsset(asset).exists():
			self.dirPathForAsset(asset).mkdir(parents=True)
		with self.dataFileForAsset(asset).open("w") as f:
			json.dump(data, f, indent=4)


	def loadAssetFromDir(self, path:Path)->Asset:
		"""loads an asset from a directory"""
		with (path / "_data.json").open("r") as f:
			data = json.load(f)
		asset = Asset(uid=data["uid"], tags=data["tags"])
		return asset


	def refreshFromAssetDir(self):
		"""rescan the asset directory and load all assets -
		this is fine until stuff has to be cached"""
		self.assets.clear()
		for i in constant.getAssetRoot().iterdir():
			if i.is_dir():
				try:
					asset = self.loadAssetFromDir(i)
					self.assets[asset.uid] = asset
				except FileNotFoundError:
					pass

