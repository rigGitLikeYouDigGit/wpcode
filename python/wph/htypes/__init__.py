from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


"""creating nodes in houdini is a pain if you don't know all the string
type names off by heart, since they don't always map to the
default names you see in the ui,

also it's more common to see them extended with HDAs and packages

try something a bit wacky - rebuild a type hint file every time wph
runs, with every node type listed as an object attribute
"""
