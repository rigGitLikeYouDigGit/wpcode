from __future__ import annotations
import typing as T

import sys, os, threading

if T.TYPE_CHECKING:
	from .task import StreamTask


class TaskStreamBase:
	"""Persistent object sitting in TaskPool -
	effectively an incoming stream of tasks.

	Lower priority tasks are attended to quicker.

	Two concrete classes provided:
	- SingleTaskStream: holds only single task, incoming task overrides it
	- MultiTaskStream: holds configurable backlog of tasks

	getTask() returns None if no tasks available, does not affect object
	takeTask() removes task from object and returns it
	"""

	def __init__(self, name:str, priority:int=5):
		self.name = name
		self.priority = priority

	def addTask(self, task: StreamTask):
		raise NotImplementedError()

	# def getTask(self) -> StreamTask:
	# 	"""read task without removing it from stream -
	# 	don't use this."""
	# 	raise NotImplementedError()

	def takeTask(self) -> StreamTask:
		"""read and remove task from stream"""
		raise NotImplementedError()


class SingleTaskStream(TaskStreamBase):
	"""TaskStream that holds only single task, incoming task overrides it"""

	def __init__(self, name:str, priority:int=5):
		super().__init__(name, priority)
		self._task = None

	def addTask(self, task: StreamTask):
		self._task = task

	def takeTask(self) -> StreamTask:
		task = self._task
		self._task = None
		return task



