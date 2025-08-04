
from __future__ import annotations
import typing as T

import fnmatch
from enum import Enum
from wplib.sentinel import Sentinel
"""
module for dictionary utilities
"""

def defaultUpdate(base:dict, defaults:dict, recursive:bool=False)->dict:
	"""update a dictionary with another dictionary, but only if the key is not present

	returns a new ref to base"""
	for key, value in defaults.items():
		if key not in base:
			base[key] = value
		elif recursive and isinstance(value, dict):
			base[key] = defaultUpdate(base[key], value, recursive=recursive)
	return base

def enumKeyLookup(key:T.Union[Enum, str], data:dict, default=Sentinel.FailToFind):
	"""given a dict with enum keys,
	return value maching either enum object
	or enum value
	if default not given, raises keyError on missing"""

	try:
		result = data[key.value]
		return result
	except KeyError:
		try:
			result = data[key]
			return result
		except KeyError:
			if default is Sentinel.FailToFind:
				raise KeyError(f"key {key} not found in {data}")
			return default

def fnMatchGet(d:dict, pattern:str, returnTies=True):
	"""maybe this should be put in a different "filter" library
	or something -
	return all values (or key, value ties) where the key matches the
	given pattern through fnMatch"""
	matching = fnmatch.filter(d.keys(), pattern)
	if returnTies:
		return [(i, d[i]) for i in matching]
	return [d[i] for i in matching]

