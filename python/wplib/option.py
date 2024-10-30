from __future__ import annotations
import types, typing as T
import pprint
from wplib import log




""" originally made this file to help defining options,
more flexibly than just enums - 
interoperation between option types and names, list vs dict etc

maybe we can do better but there was some decent stuff here

on second look this just seems super complicated
"""
from dataclasses import dataclass
from enum import Enum

# specific class if you need to define order, tooltip etc
@dataclass
class OptionItem:
	name : str
	value : object = None
	enabled : bool = True
	tooltip : str = ""

optionType = (T.Type[Enum], tuple, set, list, dict, T.List[OptionItem])

# coerce all formats of option to dict
def optionMapFromOptions(options:optionType)->dict:
	"""create string mapping from given options"""
	if isinstance(options, dict):
		return {str(k) : v  for k, v in options.items()}
	elif isinstance(options, type(Enum)):
		return {i.name.split(".", 1)[-1] : i for i in options}
	optionMap = {}
	for i in options:
		if isinstance(i, OptionItem):
			optionMap[i.name] = i
		else:
			optionMap[str(i)] = i
	return optionMap

def optionItemsFromOptions(options:optionType)->list[OptionItem]:
	"""create rich OptionItem objects"""

def optionKeyFromValue(optionValue, optionMap:dict):
	if optionValue in optionMap:
		return optionValue
	try:
		return next(k for k, v in optionMap.items() if v == optionValue or v is optionValue)
	except:
		return None

def optionFromKey(optionKey:str, optionMap:dict):
	"""retrieve original option object from key"""
	return optionMap[optionKey]

