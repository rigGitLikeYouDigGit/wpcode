from __future__ import annotations
import types, typing as T
import pprint
from collections import defaultdict
from pathlib import Path
import sys, os
import json, psutil, time, subprocess, hashlib
import socket
import threading
import queue
from weakref import ref, WeakValueDictionary, WeakSet

from typing import Optional, Any

from wplib.object.callback import CallbackOwner

try:
	import file_hash
except ImportError:
	pass


""" so this watcher isn't designed for high-throughput, but
maybe it's suitable

"""


class FileWatcher:
	"""intended as singleton per interpreter to
	set up events on file changes, and to cache file status to avoid.

	Threading can either update changeMap as a queue, or
	fire event callback directly

	TODO: is it worth a base class for callback function owners?

	"""

	def __init__(
			self,
	        pathMasks: T.Sequence[str],
			callbackFnMap : dict[str | T.Iterable[str],
				T.Callable[[str, dict[str, T.Any]],
				None]] = None,
			triggerCallbacksOnCheck=False,
			initialise=True
				#rootPath: Path=Path(os.getcwd()),
	             ):
		#self.rootPath = rootPath
		if callbackFnMap is None:
			callbackFnMap = {}
		self.callbackFnMap = callbackFnMap
		# using attribute for ease of setting up in thread
		self.triggerCallbacksOnCheck = triggerCallbacksOnCheck

		self.pathMasks = pathMasks

		self._fileStatusCache : dict[str, dict[str, T.Any]] = defaultdict(dict)

		# { individual file : change }
		self._changeMap : dict[str, T.Any] = {}

		# set up initial state
		self.checkAll()

	def _checkAll(self)->defaultdict[str, dict[str, T.Any]]:
		""" run over all tokens, list all,
		build map of statuses"""

	def _compareStateMaps(self, oldMap, newMap)->defaultdict[str, dict[str, T.Any]]:
		""" compare states to get changes"""

	def _checkFileChanges(self):
		"""main tick function check for changes, update cache"""

	def _triggerCallbacks(self):
		"""for each key """

	def _getMatchingFiles(self, exp:str | T.Iterable[str])->list[str]:
		"""get list of files matching the expression, for callback triggering"""

	def checkFileStates(self):
		"""public function to check for changes, can be called by external timer or thread"""
		self._checkFileChanges()
		if self.triggerCallbacksOnCheck:
			if self._changeMap:
				for exp, fn in self.callbackFnMap.items():
					matching = self._getMatchingFiles(exp)
					changes = []
					for k, v in self._changeMap.items():
						pass


	def checkForSavedChanges(self):
		pass


def _getProcessLockData()->tuple:
	return os.getpid(), float(psutil.Process(os.getpid()).create_time())

def _getProcessLockFileData(lockFilePath:str)->tuple:
	data = json.loads(open(lockFilePath).read())
	return data["pid"], data["createTime"]

def _makeLockFile(lockFilePath: str|Path) -> bool:
	"""
	Attempt to create and own the lock file at lockFilePath.
	Returns True if THIS process created it (i.e. should set up the watcher).
	Returns False if another live process already owns it.
	Guaranteed: only one process returns True even under simultaneous calls.

	there's a bit going on here but I trust the machine more for well-travelled
	problems like setting up exclusive file locks
	"""
	Path(lockFilePath).parent.mkdir(parents=True, exist_ok=True)
	lockFilePath = str(lockFilePath)
	currentPid, currentCreateTime = _getProcessLockData()
	# currentPid = os.getpid()
	# currentCreateTime = psutil.Process(currentPid).create_time()

	lockData = {
		"pid": currentPid,
		"createTime": currentCreateTime
	}

	# ── Attempt 1: atomic exclusive create ───────────────────────────────────
	# open with 'x' is a single atomic syscall - only one racing process wins
	try: # x stands for exclusive apparently
		with open(lockFilePath, 'x') as lockFile:
			json.dump(lockData, lockFile)
		return True

	except FileExistsError:
		pass  # another process got there first - but may be stale, check below

	# ── Attempt 2: check if existing lock owner is still alive ───────────────
	# small sleep to let a winning-but-still-writing process finish its write
	time.sleep(0.05)

	try:
		with open(lockFilePath, 'r') as lockFile:
			existingData = json.load(lockFile)

		existingPid = existingData["pid"]
		existingCreateTime = existingData["createTime"]

		# verify pid exists AND matches creation time (guards against pid reuse)
		if psutil.pid_exists(existingPid):
			try:
				existingProcess = psutil.Process(existingPid)
				if abs(existingProcess.create_time() - existingCreateTime) < 1.0:
					# legitimate live owner exists - we are a client, not the watcher
					return False
			except psutil.NoSuchProcess:
				pass  # died in the window between pid_exists and Process() - fall through

		# ── Attempt 3: stale lock - dead owner, try to claim it ──────────────
		# use atomic replace: write new data to temp, then os.replace() over stale lock
		tmpPath = lockFilePath + f".tmp_{currentPid}"
		try:
			with open(tmpPath, 'w') as tmpFile:
				json.dump(lockData, tmpFile)
			os.replace(tmpPath, lockFilePath)
			return True
		except Exception:
			# another process may have claimed it between our stale check and replace
			# treat as lost race - we are a client
			if os.path.exists(tmpPath):
				try:
					os.remove(tmpPath)
				except OSError:
					pass
			return False

	except (FileNotFoundError, json.JSONDecodeError, KeyError):
		# file vanished or was corrupt in our read window
		# another process may have just cleaned it up - treat as lost race
		return False

def getMinimalWindowsEnv():
	childEnv = {
		"SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
		"SYSTEMDRIVE": os.environ.get("SYSTEMDRIVE", "C:"),
		"TEMP": os.environ.get("TEMP", "C:\\Windows\\Temp"),
		"TMP": os.environ.get("TMP", "C:\\Windows\\Temp"),
		"PATH": os.environ.get("PATH", ""),
	}
	return childEnv

def runPythonInDetachedProcess(pyScriptPath,
                               *pyArgs,
                               processEnv:dict=None,
                               pyExePath:str|Path=None,
                               cwd=None
                               )->subprocess.Popen:
	""" honestly this is the first time anyone's ever explained the windows
	process flags properly -
	if processEnv given, override the new process environment with that dict
	if pyExePath given, use that python executable to run the file
		(else the same exe running this calling process)

	The child will outlive this process and any parent console session.

	Args:
		watcherDirPath : absolute path to the watcher working directory
		inheritEnv     : if True, child inherits the calling process's full
		                 environment. if False, child gets a minimal clean env.

	Returns:
		subprocess.Popen handle. Caller can discard this — child is independent.
	"""

	# ── resolve python interpreter ────────────────────────────────────────────
	# use the same interpreter running this code so venv/packages are consistent
	pythonExecutable = pyExePath or sys.executable

	# ── environment setup ────────────────────────────────────────────────────
	childEnv = processEnv if processEnv is not None else os.environ.copy()

	# ── startup info: suppress window, prevent handle inheritance ────────────
	startupInfo = subprocess.STARTUPINFO()
	startupInfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
	startupInfo.wShowWindow = subprocess.SW_HIDE  # no console window visible

	# ── creation flags ───────────────────────────────────────────────────────
	# DETACHED_PROCESS  : child has no console at all — not tied to our session
	# CREATE_NEW_PROCESS_GROUP : child is root of its own signal group —
	#                            ctrl+c / ctrl+break in parent won't reach it
	# note: these two together give maximum independence on Windows.
	# CREATE_BREAKAWAY_FROM_JOB would additionally escape IDE job objects
	# but requires the parent job to allow it — omitted here as it raises
	# PermissionError in environments that disallow it (add try/except if needed)
	creationFlags = (
		subprocess.DETACHED_PROCESS |
		subprocess.CREATE_NEW_PROCESS_GROUP
	)

	# ── spawn ────────────────────────────────────────────────────────────────
	process = subprocess.Popen(
		[pythonExecutable, pyScriptPath, *pyArgs],
		stdin=subprocess.DEVNULL,   # no input — avoids inherited stdin handle
		stdout=subprocess.DEVNULL,  # server logs to file — no pipe to deadlock
		stderr=subprocess.DEVNULL,  # same — avoids any buffer fill scenario
		close_fds=True,             # don't leak open handles into child
		cwd=cwd,         # child working dir is its own folder
		env=childEnv,
		startupinfo=startupInfo,
		creationflags=creationFlags
	)

	# we do not call process.wait() or communicate() — child is independent.
	# the Popen object can be discarded by the caller; child will keep running.
	return process

def makeKillScript(processId:int)->str:
	return f"taskkill /PID {processId} /F"

def getFileHash(filePath, algorithm:str="md5")->str:
	with open(filePath, "rb") as fileHandle:
		digestObject = hashlib.file_digest(fileHandle, algorithm)
	return digestObject.hexdigest()

def fileFingerprint(filePath) -> Optional[tuple]:
	"""
	Cheap tuple of (size, mtime_ns) — use as a fast pre-check before
	falling back to full hashFile() only when this tuple has changed.
	"""
	try:
		stat = os.stat(str(filePath))
		return stat.st_size, stat.st_mtime_ns
	except OSError:
		return None



class EventBroadcaster:
	"""
	Owns the listening socket and the set of connected clients.
	tick() pushes events into each client's queue — sender threads drain them.
	"""

	def __init__(self, watcherDirPath: Path, host: str = "127.0.0.1"):
		self.watcherDirPath = watcherDirPath
		self.host = host

		# one queue per connected client, keyed by the client socket
		# queue is unbounded — a pathological client could grow memory,
		# but at our event rate this is not a real concern
		self._clientQueues: dict[socket.socket, queue.Queue] = {}
		self._clientsLock = threading.Lock()

		self._listenSocket: socket.socket | None = None
		self._shutdownFlag = threading.Event()

	# ── public api ───────────────────────────────────────────────────────────

	def start(self) -> int:
		"""Bind, publish port, start the accept loop. Returns the bound port."""
		self._listenSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self._listenSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self._listenSocket.bind((self.host, 0))  # 0 = OS picks free port
		self._listenSocket.listen(16)

		boundPort = self._listenSocket.getsockname()[1]

		# publish port for clients to discover
		portFilePath = self.watcherDirPath / "port.txt"
		portFilePath.write_text(str(boundPort))

		# accept loop runs in its own thread so tick() never blocks on it
		acceptThread = threading.Thread(
			target=self._acceptLoop,
			name="EventBroadcaster-Accept",
			daemon=True
		)
		acceptThread.start()

		return boundPort

	def broadcast(self, events: dict[str, Any]) -> None:
		if not any(events.values()):
			return
		payload = json.dumps(events).encode("utf-8") + b"\n"

		with self._clientsLock:
			deadClients = []
			for clientSocket, clientQueue in self._clientQueues.items():
				try:
					clientQueue.put_nowait(payload)
				except queue.Full:
					# client is too slow — we drop them rather than leak memory.
					# when the client reconnects it will do a full reload anyway,
					# which matches the "disk save overrides memory" semantics.
					deadClients.append(clientSocket)

			for clientSocket in deadClients:
				self._clientQueues.pop(clientSocket, None)
				try:
					clientSocket.close()  # ← triggers sender thread exit
				except OSError:
					pass

	def shutdown(self) -> None:
		"""Signal all threads to stop and close the listening socket."""
		self._shutdownFlag.set()
		if self._listenSocket is not None:
			try:
				self._listenSocket.close()
			except OSError:
				pass

		# unblock all sender threads by pushing a sentinel onto each queue
		with self._clientsLock:
			for clientQueue in self._clientQueues.values():
				clientQueue.put(None)

	# ── internals ────────────────────────────────────────────────────────────

	def _acceptLoop(self) -> None:
		"""Accept incoming client connections, spawn a sender thread each."""
		while not self._shutdownFlag.is_set():
			try:
				clientSocket, clientAddr = self._listenSocket.accept()
			except OSError:
				# socket was closed — we're shutting down
				break

			clientQueue: queue.Queue = queue.Queue(maxsize=1000)

			with self._clientsLock:
				self._clientQueues[clientSocket] = clientQueue

			senderThread = threading.Thread(
				target=self._senderLoop,
				args=(clientSocket, clientQueue),
				name=f"EventBroadcaster-Sender-{clientAddr[1]}",
				daemon=True
			)
			senderThread.start()

	def _senderLoop(self, clientSocket: socket.socket, clientQueue: queue.Queue) -> None:
		"""Drain this client's queue and push to the socket. Exits on error."""
		try:
			while not self._shutdownFlag.is_set():
				# blocks here until there's an event to send — zero CPU while idle
				payload = clientQueue.get()

				if payload is None:  # shutdown sentinel
					break

				try:
					clientSocket.sendall(payload)
				except (OSError, BrokenPipeError, ConnectionResetError):
					# client disappeared — exit this thread, prune below
					break
		finally:
			with self._clientsLock:
				self._clientQueues.pop(clientSocket, None)
			try:
				clientSocket.close()
			except OSError:
				pass





class FileWatcherServer:
	"""runs in its own process, singleton per local machine -
	set up by any client, persists until no client needs it any longer

	- set up a .lock file

	"""

	# singleton local dir to track watcher state - should be outside any
	# version control
	WATCHER_DIR_PATH = Path("c:/watcher")

	tickSpeed = 1.0 # seconds between file checks

	def __init__(self):
		self.broadcaster = EventBroadcaster(self.WATCHER_DIR_PATH)
		boundPort = self.broadcaster.start()
		# log it so you can see in watcher.log
		#logger.info(f"Event broadcaster listening on port {boundPort}")
		pass

	@classmethod
	def lockFilePath(cls)->Path:
		return cls.WATCHER_DIR_PATH / "lockFile.lock"
	@classmethod
	def userMapPath(cls)->Path:
		return cls.WATCHER_DIR_PATH / "userMap.json"
	@classmethod
	def fileMapPath(cls) -> Path:
		return cls.WATCHER_DIR_PATH / "fileMap.json"
	@classmethod
	def fileHashMapPath(cls) -> Path:
		""" not to be edited, used to track file hashes"""
		return cls.WATCHER_DIR_PATH / "fileHashMap.json"
	@classmethod
	def killScriptPath(cls) -> Path:
		return cls.WATCHER_DIR_PATH / "killFileServer.bat"
	@classmethod
	def logFilePath(cls) -> Path:
		return cls.WATCHER_DIR_PATH / "log.txt"

	@classmethod
	def setupFromClientProcess(cls, callingUserId:tuple):
		"""called from a prospective client process to set up a daemon
		process for the watcher"""
		# clear existing dir
		if cls.WATCHER_DIR_PATH.exists():
			for f in cls.WATCHER_DIR_PATH.iterdir():
				try:
					f.unlink()
				except OSError:
					pass
		cls.WATCHER_DIR_PATH.mkdir(parents=True, exist_ok=True)
		# first map with this process user id
		users = [callingUserId]
		cls.userMapPath().write_text(json.dumps(users))
		cls.fileMapPath().write_text(json.dumps({}))
		# then spawn the server process, which will take over from here
		runPythonInDetachedProcess(
			Path(__file__),
			pyExePath=None,
			cwd=cls.WATCHER_DIR_PATH,
		)


	def tick(self, dt:float)->None:
		"""run over all paths, check against file hashes"""
		if self.fileHashMapPath().exists():
			fileHashMap = json.loads(self.fileHashMapPath().read_text())
		else:
			fileHashMap = {}

		events = {
			"added" : [],
			"removed" : [],
			"changed" : [],
		}
		fileMap = json.loads(self.fileMapPath().read_text())
		for p in fileMap:
			if p not in fileHashMap:
				fileHashMap[p] = (*fileFingerprint(p), getFileHash(p))
				events["added"].append(p)
				continue
			if not Path(p).exists():
				# file deleted
				fileHashMap.pop(p)
				events["removed"].append(p)
				continue
			size, ns, h = fileHashMap[p]
			if fileFingerprint(p) == (size, ns): # no possible change
				continue
			newHash = getFileHash(p)
			if newHash != h:
				# file changed
				fileHashMap[p] = (*fileFingerprint(p), newHash)
				events["changed"].append(p)
		if any(events.values()):
			self.fileHashMapPath().write_text(json.dumps(fileHashMap))
			events["time"] = [time.strftime( '\n%a %Y-%m-%d %H:%M:%S,')]
			self.broadcaster.broadcast(events)




class FileWatcherLiveStatus:
	nonExistent = "nonExistent"
	starting = "starting"
	running = "running"
	stopping = "stopping"

def _getFileWatcherStatus()->str:
	"""check if server lock file exists -
	basically this is to stop races on setup
	"""
	if not FileWatcherServer.userMapPath().exists():
		return FileWatcherLiveStatus.nonExistent
	if not FileWatcherServer.fileMapPath().exists():
		return FileWatcherLiveStatus.starting
	return FileWatcherLiveStatus.running


class FileWatcherClient(CallbackOwner):
	"""singleton per python exe
	multiple threads might use same client - this works with a push model
	where this object can interrupt other execution,
	but if we're listening and polling for received events in a calling loop,
	need a separate object per use - otherwise one listener would drain the
	events of another?

	ok so have a sub-object for listener contained here, vended out as needed -
	those all get copies of incoming events appended to their lists, and it's up
	to the use case to clear out/check content of the lists

	then let this main object hold direct callbacks for push-model
	"""
	def __init__(self,
	             watcherDirPath: Path=FileWatcherServer.WATCHER_DIR_PATH):
		super().__init__()
		port = int((watcherDirPath / "port.txt").read_text().strip())

		self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self._socket.connect(("127.0.0.1", port))

		#self.eventQueue: queue.Queue = queue.Queue()
		self.receiverQueueSet : set[queue.Queue] = WeakSet()

		listenerThread = threading.Thread(
			target=self._listenLoop,
			daemon=True
		)
		listenerThread.start()

	def _listenLoop(self) -> None:
		buffer = b""
		while True:
			try:
				chunk = self._socket.recv(4096)
			except OSError:
				break
			if not chunk:
				break  # server closed connection

			buffer += chunk
			# line-delimited framing: split on newlines, keep any partial line
			events = []
			while b"\n" in buffer:
				line, buffer = buffer.split(b"\n", 1)
				if line:
					events += json.loads(line.decode("utf-8"))
					#self.eventQueue.put(events)
			for i in self.receiverQueueSet:
				i.put(events)
			self.fireCallbacks(events)



SETUP_RACE_TIMEOUT = 10.0

def getWatcher()->FileWatcherClient:
	"""set up server in separate process if it doesn't exist
	"""
	s = _getFileWatcherStatus()
	if s == FileWatcherLiveStatus.nonExistent:
		lockHeld = _makeLockFile(FileWatcherServer.lockFilePath())
		if lockHeld:
			# start building server process
			FileWatcherServer.setupFromClientProcess()
			return FileWatcherClient()

	if s == FileWatcherLiveStatus.running:
		return FileWatcherClient()

	t = 0.0
	while (_getFileWatcherStatus() != FileWatcherLiveStatus.running
	       and t < SETUP_RACE_TIMEOUT):
		time.sleep(0.1)
		t += 0.1
	if t >= SETUP_RACE_TIMEOUT:
		raise TimeoutError("Timed out waiting for file watcher server"
		                   " to start as a result of a race on "
		                   "acquiring it - possible issue preventing "
		                   "it from starting, causing hang with file "
		                   "system etc")
	return FileWatcherClient()


if __name__ == '__main__':
	""" actual watcher server script, runs in separate daemon process
	"""