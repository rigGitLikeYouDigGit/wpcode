
from __future__ import annotations

import typing as T

import sys, os, subprocess, threading, importlib

from pathlib import Path
import orjson

from wplib import log, WP_PY_RESOURCE_PATH

"""
maya-specific DCC class
"""

from idem.dcc import DCC, idemConfig

class Maya(DCC):
	dccName = "maya"

	@classmethod
	def launch(cls,
	           startupFn=None):
		""" command args for maya - salvaged this from
		the original jank version of idem I did years ago,
		took a while to find the exact commas to get maya to just run a file
		"""
		flags = subprocess.CREATE_NEW_CONSOLE
		DETACHED_PROCESS = 0x00000008
		CREATE_NEW_PROCESS_GROUP = 0x00000200
		flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

		# maya script to execute in console
		clientPath = os.sep.join(__file__.split(os.sep)[:-2] +
		                         ["clients", "maya.py"])

		exe = idemConfig["dcc"][cls.dccName]["exe"]
		log("exe", exe)
		startupModulePath = idemConfig["dcc"][cls.dccName]["startup"]
		mod = importlib.import_module(startupModulePath)
		log("imported startup module", mod, type(mod))
		log(mod.__file__, str(mod.__file__))
		cmd = [
			exe,
			"-command",
			"""python("execfile('{}')")""".format(
				str(mod.__file__)
				#os.path.normpath(clientPath)
				# clientPath.replace("\\", "/")
			),
		]

		return subprocess.Popen(
			cmd,
			shell=True,
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
			#creationflags= CREATE_NEW_PROCESS_GROUP
		)

if __name__ == '__main__':

	Maya.launch()
