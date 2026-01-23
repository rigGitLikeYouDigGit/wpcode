
from __future__ import annotations
import typing as T
import os, shutil, types, importlib, traceback
from pathlib import Path
from wplib import log


"""package for defining custom wrappers around individual maya node types

this won't directly import any node classes, leave that to the 
catalogue

NEED BASE CLASS for init, either inherit from that in _BASE_ or
put it in the _BASE_ author file

WN() -> random new transform
WN("myTransform") -> new transform, or an existing one if it exists
WN("myTransform", new=1) -> new transform
WN("myTransform", new=0) -> error if not found

WN.Transform -> Transform node wrapper class
WN["Transform"] -> Transform node wrapper class
WN.Transform("myTransform") -> transform node instance wrapping a node named "myTransform"

"""


class NodeClassRetriever:
	"""system to manage deferred lookup of
	node classes at runtime, and to allow the skin-crawling
	looping inheritance structure we're using.

	This gives no typing information at all, use separate
	import statements at compile time
	"""
	genDir = Path(__file__).parent / "gen"
	authorDir = Path(__file__).parent / "author"

	def __init__(self):
		self.nodeClsCache : dict[str, type[WN]] = {}

	def getNodeFile(self, nodeClsName:str) -> Path:
		"""check if given node class name exists in author dir -
		then fall back to gen -
		then raise an error"""
		fileName = nodeClsName + ".py"
		authorPath = self.authorDir / fileName
		if authorPath in self.authorDir.iterdir():
			return authorPath
		genPath = self.genDir / fileName
		if fileName in self.genDir.iterdir():
			return genPath
		#raise FileNotFoundError(nodeClsName)

	authorPackage = "wpm.core.node.author"
	genPackage = "wpm.core.node.gen"

	def getNodeModule(self, nodeClsName:str):
		"""return the module to import for the given node type
		First we look for an author module, and if found, use the class
		defined there.
		Then we fall back to the generated node class
		If we STILL don't find anything, then it's either a plugin node or a misspelling,
		so we only get the base WN type
		"""
		#path = self.getNodeFile(nodeClsName)
		try:
			mod = importlib.import_module(
				"." + nodeClsName,
				package=self.authorPackage
			)
			#log("loading", nodeClsName, "from AUTHOR")
			return mod
		except ModuleNotFoundError as e:
			# print("IMPORT ERROR")
			# traceback.print_exc()
			pass

		try:
			mod = importlib.import_module(
				"." + nodeClsName,
				package=self.genPackage
			)
		except ModuleNotFoundError as e:
			return None
		#log("loading", nodeClsName, "from GEN")
		return mod

	def getNodeCls(self, nodeClsName:str)->type[WN]:
		"""
		there is a bit of weirdness when classes try to import their
		parent types, since those are written with capital first letters,
		but modules and node types have lower first
		:param nodeClsName:
		:return:
		"""
		nodeClsNameLower = nodeClsName[0].lower() + nodeClsName[1:]
		nodeClsNameUpper = nodeClsName[0].upper() + nodeClsName[1:]
		#log("getNodeCls", nodeClsName, nodeClsNameUpper, nodeClsNameLower, self.nodeClsCache)

		found = self.nodeClsCache.get(nodeClsNameLower,
		                              self.nodeClsCache.get(nodeClsNameUpper))
		if found:
			return found
		mod = self.getNodeModule(nodeClsNameLower)
		if mod is None:
			# caveman crutch to account for THDependNode etc
			# the UV prefix casing problem again
			mod = self.getNodeModule(nodeClsNameUpper)
		if mod is None: # no class found anywhere
			# log("No generated OR author node class found for " + nodeClsName + ", using base WN type")
			self.nodeClsCache[nodeClsNameLower] = None
			return None
		cls = getattr(mod, nodeClsNameUpper)
		self.nodeClsCache[nodeClsNameLower] = cls
		return cls

retriever = NodeClassRetriever()

from .base import WN, WNMeta as _WNMeta

_WNMeta.retriever = retriever

if __name__ == '__main__':

	t = WN.AddDoubleLinear





