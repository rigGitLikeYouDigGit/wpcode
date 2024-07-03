
from __future__ import annotations
import typing as T
import importlib

"""init file to be copied into the modified directory"""

# add any extra imports
try:
	{IMPORT_BLOCK}
except NameError:
	pass
