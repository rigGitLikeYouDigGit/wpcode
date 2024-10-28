
from __future__ import annotations

import glob
import typing as T

from pathlib import Path, PurePath
import orjson

from wplib import log, string as libstr, Sentinel
from wplib.object import UidElement, SmartFolder, DiskDescriptor
from wplib.pathable import Pathable, DirPathable, RootDirPathable
from wplib.sequence import toSeq, flatten

from wp.constant import getAssetRoot, WP_ROOT


"""main asset object, to be integrated with chimaera
we go back to the original folder-based idea -
if a folder contains a file named "_asset.json", it's an asset

consider ASSET vs RESOURCE - 
an asset is usually a concrete object like a person or a ford focus that is
used in multiple places and sequences across a project
a RESOURCE can be anything at all in the studio, but should be accessible
retrievable by exactly the same path system

"""

class Show(DirPathable):
	"""
	TODO:
	 - add proper support for show-level rules here


	 """
	parent : AssetRoot

	def configDict(self)->dict:
		"""maybe there's a good way to automate this"""
		return {"prefix" : self.prefix}

	def makeNewShow(self):
		showDir = WP_ROOT / self.name
		if showDir.is_dir():
			raise RuntimeError("SHOW {} ALREADY EXISTS, STOPPING IMMEDIATELY".format(self.name))
		showDir.mkdir(parents=True, exist_ok=True)
		(showDir / "_show.json").write_text(orjson.dumps(self.configDict()))
		#return

	@classmethod
	def isValidDir(cls, path:Path):
		return ( path / "_show.json" ).exists()

	@staticmethod
	def showsDict()->dict[str, Show]:
		return root().branchMap()
	@staticmethod
	def shows()->list[Show]:
		return list(Show.showsDict().values())

	@classmethod
	def fromDiskPath(cls, path:Path)->Show:
		"""from path like 'F:\\wp\\wpcore\\asset\\character\\sourceHuman\\_asset.json'
		return a show that matches its root"""

		# get show
		d = Show.showsDict()
		show = None
		for token in Path(path).parts:
			#log(token, token in d)
			if token in d:
				showKey = token
				show = d[showKey]
				break
		return show

	@classmethod
	def fromPath(cls, path):
		"""could be ANY format of path - find some way
		to get to the show level, else error.

		maybe there should be some unified separate way of resolving
		paths like this
		maybe we should just use USD
		"""
		path = toAssetPath(path)

		try:
			return cls.showsDict()[path[0]]
		except KeyError:
			raise KeyError(f"no show found for path {path}\n in show map\n{cls.showsDict()}")

	def _buildBranchMap(self) ->dict[keyT, Pathable]:
		"""look at top-level folders under this show,
		allowing pathing into rich assets or the medial
		AssetFolder objects
		"""
		children = {}
		for childDir in self.diskPath().glob("*"):
			#print("childDir", childDir)
			if not childDir.is_dir(): continue
			child = self._buildChildPathable(
				childDir, name=childDir.name)
			if child is None: continue
			children[childDir.name] = child
		return children

	def _buildChildPathable(self, obj:Path, name:keyT):
		"""we pass a Path object as obj, check if that should be a full
		Asset wrapper or not"""
		if Asset.isAssetDir(obj):
			return Asset(name, parent=self)
		elif obj.is_dir():
			return StepDir(parent=self, name=name)
		else:
			return None


	@classmethod
	def get(cls, nameOrPath)->(Show, None):
		"""TODO: support paths"""
		if nameOrPath in root().branchMap():
			return root().branchMap()[nameOrPath]
		return None

	def topAssets(self)->list[Asset]:
		"""everyone laughing at me when I hardcode this after
		droning about fractal asset systems for the past four years

		TODO: some recursive iterator for pathable, but allowing control
		 and optimisation for iteration itself - like:

		for it, obj in self.iterRecursive():
			if isAsset(obj):
				it.pruneBranch() # stop iteration going any further down this branch
				yield obj
		"""
		result = []
		for category in self.branchMap()["asset"].branches:
			result.extend(i for i in category.branches if isinstance(i, Asset))
		return result


class StepDir(DirPathable):
	"""represent an empty category folder for assets -
	it's a shame to need these but otherwise the asset
	system seems decently intuitive

	TODO: overhaul file tree pathable stuff
	 - fn to select specifc file wrapper for given file, or given dir
	 - register a set of "valid" folder types on root, or on any child,
	    and check through those to get the right type
	 - uniform data() interface to read file, cache read data, update when changes, etc
	 - LATER, find some way to integrate this with the "expected" subtree descriptors, basically a way to use python classes as schemas to set out consistent folder formats
	"""

	def _buildChildPathable(self, obj:Path, name:keyT):
		"""we pass a Path object as obj, check if that should be a full
		Asset wrapper or not"""
		if Asset.isAssetDir(obj):
			return Asset(name, parent=self)
		elif obj.is_dir():
			return StepDir(parent=self, name=name)
		else:
			return None


defaultAssetData = {
	"tags" : {},
	"created" : None,
	"version" : {"latest" : 0},
	#"uid" : None
}

class AssetFolder(SmartFolder):
	inDir = DiskDescriptor("_in", create=False)
	outDir = DiskDescriptor("_out", create=True)
	vcsDir = DiskDescriptor("_vcs", create=True)
	workDir = DiskDescriptor("_work", create=True)
	assetData = DiskDescriptor("_asset.json", create=True,
	                           default=defaultAssetData)

	createMissingFilesOnInit = False


class Asset(Pathable):
	"""this may also be used for components

	path contains show for now
	"""

	parent : StepDir

	@classmethod
	def tokensFromDirPath(cls, path):
		return Path(str(Path(path)).split(str(WP_ROOT))[-1]).parts[1:]

	def __init__(self, name, parent=None):
		"""parent only used transiently to prepend tokens"""
		#self._path = path
		super().__init__(obj=self, parent=parent, name=name)

		self._diskPath = self.parent.diskPath() / self.name

		self._folder = None
		self._data = None

	def _buildBranchMap(self) ->dict[keyT, Pathable]:
		"""look at top-level folders under this show,
		allowing pathing into rich assets or the medial
		AssetFolder objects
		"""
		children = {}
		for childDir in self.diskPath().glob("*"):
			#log("childDir", childDir)
			if not childDir.is_dir(): continue
			child = self._buildChildPathable(
				childDir, name=childDir.name)
			if child is None: continue
			children[childDir.name] = child
		return children

	def _buildChildPathable(self, obj:Path, name:keyT):
		"""we pass a Path object as obj, check if that should be a full
		Asset wrapper or not"""
		if Asset.isAssetDir(obj):
			return Asset(name, parent=self) # no child dirs below asset
		# elif obj.is_dir():
		# 	return StepDir(parent=self, name=name)
		else:
			return None


	def exists(self)->bool:
		return self.diskPath().exists()


	def showName(self)->str:
		return self.tokens[0]

	def show(self)->Show:
		return Show.get(name=self.showName())

	def smartFolder(self)->AssetFolder:
		if self._folder is None:
			self._folder = AssetFolder(path=self.diskPath())
		return self._folder

	def diskPath(self)->Path:
		"""feels like muddying, maybe asset shouldn't know its
		direct placement on disk, should go through bank to map
		a show path to real disk.

		then again the folders power everything at the moment"""
		return self._diskPath

	@classmethod
	def isAssetDir(cls, path:Path):
		return ( Path(path) / "_asset.json" ).exists()


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

	@classmethod
	def fromDiskPath(cls, path:Path):
		"""from path like 'F:\\wp\\wpcore\\asset\\character\\sourceHuman\\_asset.json'
		return a new asset properly parented to its show and top dirs
		more complicated to support calling a project "wp" under the root folder called "wp"
		:raises KeyError
		"""

		# get show
		show = Show.fromDiskPath(path)
		assert show, f"no show found for asset disk path {path}"
		log("show", show, show.path, show.diskPath(), show.branchMap())
		testPath = Path(path).relative_to(WP_ROOT) # strip root
		tokens = testPath.parts

		assetTokens = tokens[tokens.index(show.name) + 1:]

		assetTokens = [i for i in assetTokens if not "." in i and not i.startswith("_")]
		#log(assetTokens)
		return show.access(show, assetTokens, one=True)

	@classmethod
	def fromPath(cls, path, allowStepDirs=False, default=Sentinel.FailToFind)->Asset:
		#log("from path", path)
		path = toAssetPath(path)
		#log("found path", path)
		try:
			show = Show.fromPath(path)
		except KeyError as e:
			if default is Sentinel.FailToFind:
				raise e
			return default

		#log("show", show, "path", path)
		try:
			result = show.access(show, path=path[1:], default=None)
		except (cls.PathKeyError, KeyError):
			result = None
		if not isinstance(result, Asset):
			if default is Sentinel.FailToFind:
				raise cls.PathKeyError(f"No asset found for path {path}")
			return default
		return result

	@classmethod
	def topAssets(cls, show:Show=None):
		if show:
			return show.topAssets()
		result = []
		for showName, show in Show.showsDict().items():
			result.extend(show.topAssets())
		return result

def toAssetPath(path)->list[str]:
	"""filter given path to start with a show token, strip
	out any file stuff, etc

	this won't work if we allow complex path expressions,
	might need some kind of RE to only strip top level
	"""
	if isinstance(path, str):
		path = path.replace(str(WP_ROOT), "")
		path = list(filter( None, libstr.multiSplit(path, [" ", ",", "/", "\\"],
		                         preserveSepChars=False
		                         )))
	return path

def topLevelAssetFiles(show:Show):
	"""list all paths for top-level assets in show - maybe should
	delegate to a show-based config but for now we chill.
	look for paths matching SHOW / asset / character / human
	"""
	searchDir = show.diskPath() / "*" / "*" / "*" / "_asset.json"
	return glob.glob(str(searchDir))

class AssetRoot(RootDirPathable):
	"""simplifies all kinds of logic to have a persistent root node
	holding shows as first children"""

	def __init__(self):
		super().__init__(path=WP_ROOT, name="")

	def _buildChildPathable(self, obj: Path, name: keyT) -> (DirPathable, None):
		"""we pass a Path object as obj, check if that should be a full
		Asset wrapper or not"""
		if not Show.isValidDir(obj):
			return None
		return Show(name=obj.parts[-1], parent=self)

_root = None
def root()->AssetRoot:
	global _root
	if _root is None:
		_root = AssetRoot()
	return _root

if __name__ == '__main__':
	# for show in Show.shows():
	# 	print(show, topLevelAssetFiles(show))
	# 	show._buildBranchMap()
	p = "F:\\wp\\wp\\asset\\character\\sourceHuman\\_asset.json"
	a = Asset.fromDiskPath(p)
	print("retrieved asset", a)
	# are assets persistent?
	a2 = Asset.fromDiskPath(p)
	log("identical object?", a is a2, a == a2)
	# do we want them to be identical? do we care?

	# get from normal path
	a3 = Asset.fromPath("wp/asset/character/sourceHuman")
	log(a3)
	for i in Asset.topAssets():
		log(i)







