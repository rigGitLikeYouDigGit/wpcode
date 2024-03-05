from __future__ import annotations
import typing as T

"""
tests for generating code, python or otherwise,
based on templates and rules

there are a lot of sophisticated templating libraries out there,
but I couldn't work out how to use them for this

stretch goal: work in harmony with user edits to that
code



motivating case : generating a separate wrapper class
for every type of node in maya, imported lazily as needed
and cached for future use,
while being compatible with normal WPNode system,


3 packages for every generated package:
- ref/ : raw output of generation, all files. never touched by user.
- modified/ : modified files, if any
- gen/ : generated code, except modified files
user manually moves files from gen to modified as needed,
"""

from .main import CodeGenProject