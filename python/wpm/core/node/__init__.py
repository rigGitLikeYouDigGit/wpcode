
from __future__ import annotations
import typing as T
import os, shutil, types, importlib, traceback
from pathlib import Path


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

	def getNodeModule(self, nodeClsName:str):
		#path = self.getNodeFile(nodeClsName)
		try:
			mod = importlib.import_module(
				"." + nodeClsName.lower(),
				package="wpscratch._nodeoutline.node.author"
			)
			return mod
		except Exception as e:
			print("IMPORT ERROR")
			traceback.print_exc()

		mod = importlib.import_module(
			"." + nodeClsName.lower(),
			package="wpscratch._nodeoutline.node.gen"
		)
		return mod

	def getNodeCls(self, nodeClsName:str)->type[WN]:
		found = self.nodeClsCache.get(nodeClsName)
		if found:
			return found
		mod = self.getNodeModule(nodeClsName)
		cls = getattr(mod, nodeClsName)
		self.nodeClsCache[nodeClsName] = cls
		return cls

retriever = NodeClassRetriever()

from .base import WN


if __name__ == '__main__':

	t = WN.AddDoubleLinear





