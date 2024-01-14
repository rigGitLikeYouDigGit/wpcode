from __future__ import annotations
import typing as T


class CacheObj:
	"""tiny object for an explicit access to
	a cached value, with a flag to indicate"""
	def __init__(self):
		self._value = None

	def set(self, value):
		self._value = value

	def get(self):
		return self._value

	def cached(self):
		return self._value is not None

	def clear(self):
		self._value = None


