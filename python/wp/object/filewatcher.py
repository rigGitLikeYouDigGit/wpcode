
from __future__ import annotations
import typing as T

import time, logging, threading, os

from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, FileSystemEventHandler


class ThreadedFileWatcher:
	"""watch for file changes in a separate thread -
	main-thread object remains for control purposes"""

	def __init__(self, pathMasks:list[Path]=()):
		self._pathMasks : list[Path] = list(map(Path, pathMasks))
		self._observer : Observer = Observer()
		self._handlers : list[FileSystemEventHandler] = []

		self.setPathMasks(*pathMasks)

	def setPathMasks(self, *pathMasks:Path):
		"""set paths to watch, can be specific files
		or wildcards, or directories"""
		self._pathMasks = list(pathMasks)
		self._observer.stop()
		self._observer.join()
		self._observer.unschedule_all()

		self._handlers = []
		for i in self._pathMasks:
			handler = FileSystemEventHandler()
			self._observer.schedule(
				handler, str(i), recursive=True)
			self._handlers.append(handler)

	def onFileEvent(self, event):
		"""OVERRIDE any file event"""
		pass


	def start(self):
		"""start watching"""
		self._observer.start()

	def stop(self):
		"""stop watching"""
		self._observer.stop()
		self._observer.join()


	def __del__(self):
		"""cleanup watcher on main object deletion"""
		self.stop()


def main():
	testPath = r"F:\wp\tempest\asset\character_cait___ea2eea74-1991-4bd9-bfc1-8dd9d3c092b2\costume\_out"
	watcher = ThreadedFileWatcher([testPath])
	watcher.start()
	time.sleep(30)
	watcher.stop()
	print("end")

if __name__ == '__main__':
	main()

