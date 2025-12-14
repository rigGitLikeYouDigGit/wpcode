from __future__ import annotations
import typing as T

import os
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypedDict, Tuple

try:
	import orjson
	from orjson import loads, dumps as __dumps
	dumps = lambda *args, **kwargs : __dumps(*args,
	                                         option=orjson.OPT_INDENT_2).decode("utf-8")
except (ImportError, ModuleNotFoundError):
	from json import loads, dumps

"""sometimes when saving param dialog scripts, you end up with too 
many slashes and json gets confused"""
safeCharMap = {
	"\t" : "£TAB",
	"\"" : "£DQ",
	"\'" : "£Q"
}
def makeSafeForJson(s:str):
	for k, v in safeCharMap.items():
		s = s.replace(k, v)
	return s
def regenFromJson(s:str):
	for k, v in safeCharMap.items():
		s = s.replace(v, k)
	return s

# def loads(s, *a, **k):
# 	return _loads(regenFromJson(s), *a, **k)
# def dumps(s, *a, **k):
# 	return makeSafeForJson(_dumps(s, *a, **k))

from deepdiff import Delta

#class NodeHeader(NamedTuple):
class NodeHeader(tuple):
	path : str
	scopeType : str
	nodeTypeNS : str
	nodeTypeName : str
	nodeTypeVersion : str
	exactVersionMattersexactVersionMatters : int # easier than bools in json


NodeHeader = lambda *args : args

class NodeDelta(TypedDict):
	"""overall dict representing one step
	of a node's definition"""

	nodes : dict[str, list[NodeHeader]]
	connections : dict
	params : dict[str, Delta]


class ParmNames:
	"""just for easier referring"""

	syncBtn = "syncbtn"

	localEdits = "nodeeditdelta"
	allowEditing = "alloweditingbox"
	liveUpdate = "liveupdatebox"

	endFile = "endfile"

	# folders for added hda parms
	parentHDAParmFolder = "parenthdaparmfolder"
	parentHDAParmFolderLABEL = "Parent HDA parms" # hate
	leafHDAParmFolder = "leafhdaparmfolder"
	leafHDAParmFolderLABEL = "Leaf HDA parms" # how i have come to hate


	# parent folder
	parentDir = "parentfolder"#
	# these are all suffixed "#"
	parentFile = "parentfile"
	parentNodeDelta = "parentnodedelta"
	recursiveParents = "recursiveparentsbox"

	hdaDef = "hdadefid"

	resetToBasesBtn = "resettobasesbtn"
	resetToTextHDABtn = "resettotexthdabtn"

@dataclass
class CachedFile:
	"""load content once,
	then check mtime on file to only reload when changed
	"""
	path : str
	content : T.Any = None
	_mtime : T.Any = None

	def readJson(self):
		"""if no modifications since last mtime, read last loaded
		"""
		path = Path(self.path)
		if self._mtime is not None:
			if os.path.getmtime(str(path)) == self._mtime:
				return self.content
		self._mtime = os.path.getmtime(str(path))
		with path.open("r") as f:
			self.content = loads(f.read())
		return self.content





