
from __future__ import annotations

import typing as T

import sys, os, subprocess, threading, importlib

from pathlib import Path
import orjson

from wplib import log, WP_PY_RESOURCE_PATH

"""
houdini-specific DCC class
"""

from idem.dcc import DCCProcess

class HoudiniProcess(DCCProcess):
	""""""
	dccName = "houdini"

	@classmethod
	def isThisCurrentDCC(cls) ->bool:
		try:
			import hou
			return True
		except ImportError:
			return False