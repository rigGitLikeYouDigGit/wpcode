from __future__ import annotations
import typing as T
import sys, os

import copy
from pathlib import Path, PurePath
from dataclasses import dataclass

import orjson

from wplib import inheritance, typelib
from wplib.constant import IMMUTABLE_TYPES, LITERAL_TYPES
#from wplib.serial import serialise

if T.TYPE_CHECKING:
	from wptree import Tree

"""object representing a folder on disk, with
certain files that we may expect or require to find there

motivating example is an asset folder, which should always have certain
attributes
"""

@dataclass
class DiskDescriptor(object):
	# relative path under folder
	path: (str, Path)
	# default contents of this file
	default: (T.Any, str, SmartFolder, Tree) = None
	# should file be created if it's not found when outer smart folder is made
	create : bool = False

	def __get__(self, instance:SmartFolder, owner)->Path:
		return instance.path / self.path


@dataclass
class SmartFolder(object):
	"""use to map out """
	path : Path
	createMissingFilesOnInit : bool = False

	def __post_init__(self):
		self._diskDescriptors : list[DiskDescriptor] = [
			i for i in inheritance.classDescriptors(type(self)).values()
			if isinstance(i, DiskDescriptor)
		]
		if self.createMissingFilesOnInit:
			self.createMissingFiles()


	def createMissingFiles(self):
		for i in self._diskDescriptors:
			relPath = self.path / i.path
			if i.create:
				if relPath.suffix: # it's a file
					relPath.parent.mkdir(parents=True, exist_ok=True)
					relPath.write_text(orjson.dumps(i.default)
					                   )

				else:
					relPath.mkdir(parents=True, exist_ok=True)

	def saveToPath(self, path:(Path, str, T.IO), data:T.Any):
		"""maybe move this to a lib, or allow injection closer to use -
		handle final writing to file.
		if data is just a raw string or primitive value, just dump it
		if it's an object or serialised dict, handle that instead
		"""
		# if an open file stream is not passed in
		if not isinstance(path, T.IO):
			path = open(path, "w+")
		assert path.writable()

		path.write(orjson.dumps(data))








