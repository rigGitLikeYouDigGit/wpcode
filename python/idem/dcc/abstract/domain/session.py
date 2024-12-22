from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import os, sys
from collections import defaultdict
import threading, multiprocessing
from pathlib import Path
from uuid import uuid4

import orjson

import win32pipe

class DCCIdemSession:
	"""domain-side class for access within a dcc,
	handles linking to a bridge idem process,
	declares itself as open if no bridge is found

	session is persistent with whole lifetime of DCC, or
	until this object is cleared - survive through new files, renaming, etc

	files in localsessions are named "akda_maya_myMayaFile" -
		first 4 letters are uuid, and those are all we need to identify,
		the rest are just for ease of inspection
	"""

	dccType = "python" # override with "maya", "houdini", etc

	def __init__(self, sessionTempName="", linkTo=""):
		self.uuid = str(uuid4())[:4]
		self.sessionTempName = sessionTempName

		self.loopThread : threading.Thread = None

		# create starting data files
		self.clear() # get rid of any leftover files
		self.writeSessionFileData(self.liveData())
		os.mkfifo(self.inPipeFilePath())
		os.mkfifo(self.outPipeFilePath())


		if linkTo: # immediately link to a known session
			pass
		pass

	def liveData(self)->dict:
		return {
			"id" : self.uuid,
			"processId" : os.getpid()
		}

	def clear(self):
		"""clear up all data and pipe files for this session"""
		if self.loopThread:
			self.loopThread.join()
		if not self.sessionFileDir().is_dir():
			return
		for i in tuple(self.sessionFileDir().glob(self.uuid + "_*")):
			i.unlink(missing_ok=True)

	def sessionNiceName(self):
		s = f"{self.uuid}_{self.dccType}"
		try:
			if self._sessionFileName():
				s += ("_" + self._sessionFileName())
		except NotImplementedError:
			pass
		return s


	def __del__(self):
		self.clear()
		pass

	@classmethod
	def uuidToFileMap(cls)->dict[str, dict[str, Path]]:
		fileMap = defaultdict(dict)
		for i in cls.sessionFileDir().iterdir():
			uid = i.name.split("_")[0]
			if i.name.endswith("_IN"):
				fileMap[uid]["in"] = i
			elif i.name.endswith("_OUT"):
				fileMap[uid]["out"] = i
			else:
				fileMap[uid]["data"] = i
		return fileMap

	@classmethod
	def uuidToFileData(cls) -> dict[str, dict[str, Path]]:
		fileMap = defaultdict(dict)
		for i in cls.sessionFileDir().iterdir():
			uid = i.name.split("_")[0]
			if i.name.endswith("_IN"):
				fileMap[uid]["in"] = i
			elif i.name.endswith("_OUT"):
				fileMap[uid]["out"] = i
			else:
				fileMap[uid]["data"] = cls.sessionFileData(None, i)
		return fileMap

	def sessionFileData(self, path:Path=None)->dict:
		path = path or self.sessionFilePath()
		if not path.is_file():
			return None
		with open( path, "rb") as f:
			return orjson.loads(f.read())

	def writeSessionFileData(self, data:dict, path:Path=None):
		path = path or self.sessionFilePath()
		log("write session file data", path, data)
		with open(path, "wb") as f:
			f.write(orjson.dumps(data))

	@classmethod
	def sessionFileDir(cls)->Path:
		from idem import getIdemDir
		return getIdemDir() / "localsessions"

	def _sessionFileName(self)->str:
		return f"{self.uuid}_{self.dccType}_{self._getDCCFileName()}"
	def sessionFilePath(self)->Path:
		return self.sessionFileDir() / (self._sessionFileName() + ".json")
	def _getDCCFileName(self)->str:
		"""current file name worked on by this DCC session
		or None if no file name set
		OVERRIDE THIS
		"""
		#raise NotImplementedError
		return self.sessionTempName

	def outPipeFileName(self)->str:
		return f"{self.uuid}_OUT"
	def outPipeFilePath(self)->Path:
		return self.sessionFilePath() / self.outPipeFileName()
	def inPipeFileName(self)->str:
		return f"{self.uuid}_IN"
	def inPipeFilePath(self)->Path:
		return self.sessionFilePath() / self.inPipeFileName()

	def runThreaded(self):
		"""execute the below run() method in a child thread
		"""
		self.loopThread = threading.Thread(target=self.run)
		self.loopThread.start()

	def run(self):
		"""startup method that begins event loop, handles errors
		for now we just hack an event loop here
		"""
		#
		with open(self.inPipeFilePath()) as fifo:
			for line in fifo:
				self.handleMessage(line)

	def handleMessage(self, data):
		print(self.sessionNiceName() + "handle:", data)

