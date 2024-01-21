
from __future__ import annotations
import typing as T

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


