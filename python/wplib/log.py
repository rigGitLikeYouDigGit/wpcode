
from __future__ import annotations
import typing as T

import inspect

def log(*args, file=None, line=None, **kwargs):
	"""print and append link to line, for easier debugging"""
	if file is None:
		file = inspect.stack()[1].filename
	if line is None:
		line = inspect.stack()[1].lineno
	string = f'File "{file}", line {max(line, 1)}'.replace("\\", "/")
	print(*args, string, **kwargs)
	return string
