from __future__ import annotations
"""lib functions for asset system - 
find some way to inject control here, if
plugins want to define their own syntax,
tag combining rules etc"""

def tagsToIndexString(tags:dict)->str:
	"""converts tags to string"""
	return " ".join(sorted([f"{k}_{v}" for k, v in tags.items()]))


def tagsToSemicolonString(tags:dict)->str:
	"""converts tags to string"""
	return ";".join(sorted([f"{k}-{v}" for k, v in tags.items()]))


