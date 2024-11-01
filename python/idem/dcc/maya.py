
from __future__ import annotations

import typing as T

import sys, os, subprocess, threading, importlib

from pathlib import Path
import orjson

from wplib import log, WP_PY_RESOURCE_PATH

"""
idem-specific DCC classes shouldn't do too much, just
run the program and pass in functions to execute at startup - 
for now.

maybe it's worth having a single higher class to represent DCC overall,
to start up headless sessions, servers? import, export?
could do.

to state the obvious, dcc domain code should not depend on idem

- associate right dcc launch with class - get exe path

don't care about software versions for now, just get it working

a core dcc package like wpm could define core operations to run on startup
then wpm defines pipeline ops to run on startup
then idem defines random idem stuff to run on startup
then individual assets, shots, etc - 
there's a lot of stuff we could want to stack on certain events for dccs
absolutely NO idea how to do any of that, for now we hardcode a single file

"""


p = Path(__file__)
while p.name != "idem":
	p = p.parent
log("p", p)

idemConfig = orjson.loads((p / "config.json").read_bytes())
log("config", idemConfig)
class DCC:
	dccName = ""
	pass

	@classmethod
	def iconPath(cls)->Path:
		return WP_PY_RESOURCE_PATH / "icon" / (cls.dccName + ".png")


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
