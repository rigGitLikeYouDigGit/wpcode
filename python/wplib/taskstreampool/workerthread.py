
from __future__ import annotations
import typing as T

import sys, os, threading, time

if T.TYPE_CHECKING:
	from .main import TaskStreamPool

class TaskThread(threading.Thread):
	"""thread that runs a taskpool"""

	def __init__(self, pool:TaskStreamPool):
		super(TaskThread, self).__init__()
		self.pool = pool
		self.shouldPause = False
		self.shouldDie = False
		self.minPeriod = 0.1 # minimum time in seconds between polls to pool


	def run(self):
		"""main worker function - poll taskpool to get
		higher priority task, then execute it.
		If none are found, sleep for min period.
		then do it again
		"""
		while True:

			# check if thread should exit - don't immediately join from here,
			# just halt iteration
			if self.shouldPause or self.shouldDie:
				break

			# log start time for waiting
			startTime = time.time()

			# check for task, execute it if found
			task = self.pool.takeTask()
			if task is not None:
				task.exec()

			# sleep up to min period
			time.sleep(max(0, self.minPeriod - (time.time() - startTime)))

