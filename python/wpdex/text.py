
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log
from wpdex import WpDex

"""
test for an alternate, raw-text view of data - 
we depend on a data structure that can be serialised to a string,
and loaded from it.

by default, just a raw json-esque text editor, 
but what if we could add more useful overlays on top - 
options to move items around, or to edit them in a more structured way?

"""



