from __future__ import annotations
import types, typing as T
import pprint

from wpdex import WpDexProxy
from wplib import log


def toStr(x):
	"""TODO: move this to somewhere
	consider the full send - to be robust to different ways
	of representing objects, could define a new hierarchy of
	string-adaptor for each kind of expression syntax, and
	implement each object type bespoke

	"""
	if isinstance(x, WpDexProxy):
		return toStr(x._proxyTarget())
	return str(x)
