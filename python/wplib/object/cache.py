from __future__ import annotations

import typing
import typing as T
import os, orjson
from pathlib import Path

from wplib import log

class CacheObj:
	"""tiny object for an explicit access to
	a cached value, with a flag to indicate"""
	def __init__(self):
		self._value = None

	def set(self, value):
		self._value = value

	def get(self):
		return self._value

	def cached(self):
		return self._value is not None

	def clear(self):
		self._value = None

class CacheFile:
	"""file wrapper reloading if it detects file is modified
	us os.stat mtime_ns
	"""

	@staticmethod
	def defaultRead(fd:typing.IO):
		return fd.read()

	def __init__(self, path:Path, loadFn=defaultRead):
		self.path = path
		self.loadFn = loadFn
		self.lastStat : os.stat_result = os.stat(self.path)
		with open(self.path, mode="r") as f:
			self._data = self.loadFn(f)

	def data(self):
		stat = os.stat(self.path)
		if stat.st_mtime_ns > self.lastStat.st_mtime_ns: # reread it
			log("refreshing cached file", self.path)
			with open(self.path, mode="r") as f:
				self._data = self.loadFn(f)
			self.lastStat = stat
		return self._data



