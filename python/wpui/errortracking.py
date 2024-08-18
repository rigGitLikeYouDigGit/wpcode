
from __future__ import annotations
import typing as T

import sys

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.object import SingletonDecorator

"""
totally decoupled system to track errors
in ui - 
when error is detected by singleton manager,
check against registered windows and widgets,
and all their children, to see if they occur
in callstack frames

"""

@SingletonDecorator
class QtErrorTracker:
	"""object sitting off to the side, registering UI
	windows to pass drawn error events.

	TODO:
	"""





