from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import requests
"""let's hope this file doesn't get too large"""

def isOnline(url: str) -> bool:
	"""Checks if given url is valid"""
	try:
		return requests.head(url, timeout=3).status_code // 100 == 2
	except Exception:
		return False
