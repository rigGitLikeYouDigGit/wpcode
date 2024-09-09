
from __future__ import annotations
import typing as T

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
	result = Sentinel.FailToFind
	try:
		result = data[key.value]
	except:
		result = data[key]
	if result is Sentinel.FailToFind:
		if default is Sentinel.FailToFind:
			raise KeyError(f"key {key} not found in {data}")
		return default
	return result

