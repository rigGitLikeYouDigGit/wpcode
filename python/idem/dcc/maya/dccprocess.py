
from __future__ import annotations

import typing as T

import sys, os, subprocess, threading, importlib

from pathlib import Path
import orjson

from wplib import log, WP_PY_RESOURCE_PATH

"""
maya-specific DCC class
"""

"""
TODO: 
	- headless mode maybe
	- launch() is now an instance method, with DCCs being instantiated objects - 
		maybe there's still merit in having a class-level description of 
		generally how to launch programs, but
"""

from idem.dcc import DCCProcess

class MayaProcess(DCCProcess):
	""""""
	dccName = "maya"

	def launch(self,
	           #startupFn=None
	           idemParams:dict=None
	           ):
		""" command args for maya - salvaged this from
		the original jank version of idem I did years ago,
		took a while to find the exact commas to get maya to just run a file
		"""
		idemParams = idemParams or {}
		idemConfig = self.getConfig()
		flags = subprocess.CREATE_NEW_CONSOLE
		DETACHED_PROCESS = 0x00000008
		CREATE_NEW_PROCESS_GROUP = 0x00000200
		flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

		# maya script to execute in console
		clientPath = os.sep.join(__file__.split(os.sep)[:-2] +
		                         ["clients", "maya.py"])
		exe = idemConfig["dcc"][self.dccName]["exe"]
		log("exe", exe)
		#startupModulePath = idemConfig["dcc"][self.dccName]["startup"]
		#startupModulePath = str(self.idemStartupFilePath())
		startupModulePath = self.idemStartupFilePath()
		log("startupPath", startupModulePath)


		cmd = [
			exe,
			"-command",
			"""python("exec( open('{}').read() )")""".format(
				str(startupModulePath).replace("\\", "/"),
				#str(mod.__file__)
				#data
				#startupModulePath.read_text()
				#os.path.normpath(clientPath)
				# clientPath.replace("\\", "/")
			),
			"-noAutoloadPlugins",
			#"testVal" # raw test val works?
			#"-idem_params", # dashes freak out maya - raw trailing values work
			#"--idem_params",
			"idemParams::" + orjson.dumps(idemParams).decode("utf-8")
		]

		#log("cmd", cmd)

		# test launching maya in the folder of the current asset - this might
		# completely wreck all its pathing though
		cwd = idemParams.get("launchInDir")

		process = subprocess.Popen(
			cmd,
			shell=True,
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
			#creationflags= CREATE_NEW_PROCESS_GROUP
			cwd=cwd
		)
		self.process = process
		return process

if __name__ == '__main__':

	dcc = MayaProcess("testMayaProcess")
	dcc.launch(idemParams={
		"processName" : "testMaya",
		"portNumber" : 23998
	})
