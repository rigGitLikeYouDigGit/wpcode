
from __future__ import annotations
import typing as T
import os, shutil, types, importlib, traceback
from pathlib import Path


class NodeClassRetriever:
	"""system to manage deferred lookup of
	node classes at runtime, and to allow the skin-crawling
	looping inheritance structure we're using.

	This gives no typing information at all, use separate
	import statements at compile time
	"""

	def getNodeFile(self, nodeClsName:str) -> Path:
		"""check if given node class name exists in author dir -
		then fall back to gen -
		then raise an error"""
		fileName = nodeClsName + ".py"
		authorPath = authorDir / fileName
		if authorPath in authorDir.iterdir():
			return authorPath
		genPath = genDir / fileName
		if fileName in genDir.iterdir():
			return genPath
		#raise FileNotFoundError(nodeClsName)

	def getNodeModule(self, nodeClsName:str):
		path = self.getNodeFile(nodeClsName)
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

	def getNodeCls(self, nodeClsName:str)->WN:
		mod = self.getNodeModule(nodeClsName)
		return getattr(mod, nodeClsName)

retriever = NodeClassRetriever()


class WNMeta(type):
	"""replace .attr lookup with
	a retrieval for the node class"""

	def __getattr__(self, item):
		return retriever.getNodeCls(item)

class Catalogue:
	pass

if T.TYPE_CHECKING:
	from .gen import *
	from .author import *


class WN(
	Catalogue,
	metaclass=WNMeta):
	"""node base class"""



genDir = Path(__file__).parent / "gen"
authorDir = Path(__file__).parent / "author"





