from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtGui, QtWidgets

from idem.ui.sessionstatus import BlinkLight

"""
small label to show name and id of a running process,
with its indicator lights for sending, receiving and process health
"""

from .sessionstatus import ProcessStatusWidget



