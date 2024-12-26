
from __future__ import annotations
import typing as T, types

import sys, os, subprocess, threading, importlib

from pathlib import Path
import orjson, inspect

"""
overall launcher for asset-focused program sessions

maybe replace with allzpark / rez after we get everything working,
if it can be made to work with the pipeline system

we could go super hard here in setting up plugin interfaces, registering
packages to run any random code on startup -
not yet.

just get it to launch the programs and add the WP code paths

STRUCTURE :
anything in top-level is abstract base classes and/or pure python, that can run
	directly in the standalone Idem program

".dcc" package holds outwards-facing wrapper objects to handle program sessions -
	finding windows, getting processes, startup, and actually sending commands to
	start up an idem process in a dcc session live (if the program can support it)

we define a separate package for each supported dcc, where we specialise the
relevant classes and chimaera nodes
each dcc-specific package may import domain-specific code

TODO: a better name
 ilex
 concourse
 concord (rip)
 wrangle
 wipe
 wot
 tangle
 tara
 tarana
 brunel
 isle?
 isles?


tracking local idem sessions is done through localsessions folder - 
each new system or service creates a new named file there,
and updates it if it connects to a bridge

files also store data to find the process they relate to?

for now just use basic json serialisation for communicating - 
later it might be worth having separate "command pipe" and "data pipe"
to send any random json dict for directions, but the dense mesh stuff
over port?


"""

def getIdemDir()->Path:
	return Path(__file__).parent

def getConfig()->dict:
	p = Path(__file__).parent
	idemConfig = orjson.loads((p / "config.json").read_bytes())
	return idemConfig

from . import adaptor, model, node, ui
from .model import IdemSession, IdemGraph
from .dcc.maya import *

__IDEM__ = None
def getSession()->IdemSession:
	return __IDEM__

