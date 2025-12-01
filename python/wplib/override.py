from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wplib.object import DeepVisitor, DeepVisitOp

"""
back to the old interest, just dictionaries and lists
for now, for json - 

given 2 objects of similar structures,
allow extending and overriding the base.

Use Houdini's env vars as inspiration for syntax?

"""


def override(base, overlay, prevChar="@"):
	"""VERY simple in first pass -
	later try and do function/lambda stuff to allow things like
	["a", @[1:3], "z"]

	"""
	return base

