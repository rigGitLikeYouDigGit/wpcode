from __future__ import annotations
import typing as T

"""tests for any custom annotations or decorators

"""

def toOverride(fn:T.Callable)->T.Callable:
	"""decorator to mark a method as needing to be overridden
	"""
	return fn


