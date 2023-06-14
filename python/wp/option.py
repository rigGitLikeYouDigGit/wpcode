
from __future__ import annotations
""" file dealing with specifying options for widgets and other things
"""
import typing as T
from dataclasses import dataclass
from enum import Enum

from tree.lib.object import TypeNamespace

# specific class if you need to define order, tooltip etc
@dataclass
class OptionItem:
	name : str
	value : object = None
	#enabled : bool = True
	tooltip : str = ""



optionType = (T.Type[Enum], tuple, set, list, dict,
              T.List[OptionItem],
              T.Type[TypeNamespace])

# coerce all formats of option to dict
def optionMapFromOptions(options:optionType)->dict[str, object]:
	"""create string mapping from given options"""
	if isinstance(options, dict):
		return {str(k) : v  for k, v in options.items()}
	elif isinstance(options, type(Enum)):
		return {i.name.split(".", 1)[-1] : i for i in options}
	elif issubclass(options, TypeNamespace):
		return {i.clsNameCamel() : i for i in options}
	optionMap = {}
	for i in options:
		if isinstance(i, OptionItem):
			optionMap[i.name] = i
		else:
			optionMap[str(i)] = i
	return optionMap

def optionItemsFromOptions(options:optionType)->list[OptionItem]:
	"""create rich OptionItem objects from the given input type
	"""
	return [OptionItem(name=k, value=v) for
	        k, v in optionMapFromOptions(options).items()]

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


# filtering options based on user-supplied lambda

@dataclass
class FilteredOptionsData:
	"""class passed in to lambda, to be populated by user if wanted"""
	# parse as tooltips for options allowed, if supplied
	validNotes : dict[OptionItem, str] = None

	# parse as tooltips for options not allowed, if supplied
	invalidNotes : dict[OptionItem, str] = None

optionFilterFnType = T.Callable[[list[OptionItem], FilteredOptionsData], list[OptionItem]]

def optionFilterTemplateFn(options:list[OptionItem], data:FilteredOptionsData)->list[OptionItem]:
	"""filter function for options - define your own as tools require.
	Should return list of options that are valid for this context.

	Populate data to give more details on why some options are valid or invalid.
	"""
	return options

def filterOptions(
		options:list[OptionItem],
		filterFn:optionFilterFnType=optionFilterTemplateFn,
		data:FilteredOptionsData=None)->list[OptionItem]:
	"""filter options based on user-supplied lambda
	optionally pass in data to be populated by user-supplied lambda"""
	if data is None:
		data = FilteredOptionsData()
	validOptions = []
	return filterFn(options, data)


