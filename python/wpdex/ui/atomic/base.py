from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

import reactivex as rx

from wplib import log
from wplib.object import Signal

from wpdex import WpDexProxy
from wpdex.ui.react import ReactiveWidget


class AtomicWidget(ReactiveWidget):
	"""a base class for atomics like this may be pointless,
	ReactiveWidget has everything we need anyway
	"""
