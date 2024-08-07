
from __future__ import annotations
import typing as T

import inspect, pprint


def getLineLink(file=None, line=None) ->str:
	"""return clickable link to file, to output to log"""
	return f'File "{file}", line {max(line, 1)}'.replace("\\", "/")

def log(*args, file=None, line=None, printMsg=True, vars=True, **kwargs):
	"""print and append link to line, for easier debugging"""
	if file is None:
		file = inspect.stack()[1].filename
	if line is None:
		line = inspect.stack()[1].lineno
	#string = f'File "{file}", line {max(line, 1)}'.replace("\\", "/")
	string = ""
	if vars:
		string += " \nVARS: " +pprint.pformat(inspect.stack()[1].frame.f_locals) + "\n^ "# + " \n^ " + str(inspect.stack()[1].frame.f_globals)
	string += getLineLink(file, line)

	if printMsg:
		print(*args, string, **kwargs)
	return string


def getDefinitionFileLine(obj:T.Any) ->T.Tuple[str, int]:
	"""get file and line number of definition of object"""
	return inspect.getsourcefile(obj), inspect.getsourcelines(obj)[1]


def getDefinitionStrLink(obj:T.Any) ->str:
	"""get link to line number of definition of object"""
	return getLineLink(*getDefinitionFileLine(obj))


