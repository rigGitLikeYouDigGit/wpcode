
from __future__ import annotations
import typing as T

import sys, os, threading

from collections import defaultdict

from .workerthread import TaskThread

if T.TYPE_CHECKING:
	from .task import StreamTask
	from .stream import TaskStreamBase

class TaskStreamPool:
	"""Sort task streams into array of queues
	Whenever single stream is served by a thread, and its task executed,
	move that stream to the back of its queue - in this way we ensure that all
	tasks have decent chance of being served if they ever
	receive a task.
	"""

	def __init__(self, nThreads=1, threadMinPeriod=0.1):

		# persistent map of {stream name : stream}
		self._streamMap : dict[str, TaskStreamBase] = {}

		# arrays of task streams, sorted by priority and precedence in queue
		# rebuilt whenever stream attributes change
		self._streamQueue: list[list[TaskStreamBase]] = []

		# array of worker threads
		self._threads : list[TaskThread] = []
		self.threadMinPeriod = threadMinPeriod

		self.running = False
		self.setNThreads(nThreads)


	def start(self):
		"""start all threads running"""
		for i in self._threads:
			i.start()
		self.running = True

	def stop(self):
		for i in self._threads:
			self._stopThread(i)
		self.running = False


	def _stopThread(self, thread:TaskThread):
		thread.shouldPause = True
		thread.join()

	@classmethod
	def getThreadCls(cls)->type[TaskThread]:
		return TaskThread

	def _newThread(self)->TaskThread:
		"""override for any custom thread construction"""
		return TaskThread(pool=self)

	def nThreads(self)->int:
		return len(self._threads)

	def setNThreads(self, nThreads:int):
		"""clear current threads, add new ones equal to nThreads"""
		currentNThreads = self.nThreads()

		# remove excess threads
		for i in range(max(0, currentNThreads - nThreads)):
			threadToRemove : TaskThread = self._threads[-i]
			self._stopThread(threadToRemove)
			self._threads.pop(-1)

		# add missing threads
		for i in range(max(0, nThreads - currentNThreads)):
			newThread = self._newThread()
			self._threads.append(newThread)
			newThread.start()



	def getStreamMap(self)->dict[str, TaskStreamBase]:
		return self._streamMap

	def getStreamQueue(self)->list[list[TaskStreamBase]]:
		return self._streamQueue

	def rebuildStreamQueue(self):
		prioTaskMap = defaultdict(list)
		for name, stream in self.getStreamMap().items():
			prioTaskMap[stream.priority].append(stream)
		queue = [ prioTaskMap[prio] for prio in sorted(prioTaskMap) ]
		self._streamQueue = queue

	def addStream(self, stream:TaskStreamBase):
		self._streamMap[stream.name] = stream
		self.rebuildStreamQueue()

	def removeStreamByName(self, name:str):
		self._streamMap.pop(name)
		self.rebuildStreamQueue()

	# def removeStream(self, stream:TaskStreamBase):
	# 	self.removeStreamByName(stream.name)



	def takeTask(self)->StreamTask:
		"""Main thread-facing function - return the highest-priority
		task in entire pool, remove it from its stream, shuffle that stream to the
		back of its queue"""

		for queue in self.getStreamQueue():
			for stream in tuple(queue):
				task = stream.takeTask()
				if task is not None: # task found, shuffle stream and return task
					queue.append(queue.remove(stream))
					return task
		return None

	def __del__(self):
		"""halt all worked threads"""
		for i in tuple(self._threads):
			self._stopThread(i)
			self._threads.remove(i)
			del i



