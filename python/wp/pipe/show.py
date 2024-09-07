
from __future__ import annotations
import typing as T
from pathlib import Path
from dataclasses import dataclass

import orjson

from wp import constant

@dataclass
class Show:
	name : str
	prefix : str

	@classmethod
	def get(cls, *args, showDir:Path=None, name=""):
		if args:
			if isinstance(args[0], Show):
				return args[0]
			if isinstance(args[0], str):
				name = args[0]
			if isinstance(args[0], Path):
				showDir = args[0]
		if showDir:
			data = orjson.loads( (showDir / "_show.json").read_text() )
			return Show(name=showDir.name,
			            prefix=data["prefix"]
			            )
		if name:
			return availableShows()[name]
		raise RuntimeError("no options passed to Show.get()")

	def configDict(self)->dict:
		"""maybe there's a good way to automate this"""
		return {"prefix" : self.prefix}

	def makeNewShow(self):
		showDir = constant.WP_ROOT / self.name
		if showDir.is_dir():
			raise RuntimeError("SHOW {} ALREADY EXISTS, STOPPING IMMEDIATELY".format(self.name))
		showDir.mkdir(parents=True, exist_ok=True)
		(showDir / "_show.json").write_text(orjson.dumps(self.configDict()))

	def path(self)->Path:
		return constant.WP_ROOT / self.name


def availableShows()->dict[str, Show]:
	"""check any top-level folders having a "_show.json" file
	"""
	result = {}
	for childDir in constant.WP_ROOT.iterdir():
		if not childDir.is_dir():
			continue
		if not (childDir / "_show.json").exists():
			continue
		result[childDir.name] = Show.get(childDir)
	return result

if __name__ == '__main__':
	print("availableShows")
	print(availableShows())











