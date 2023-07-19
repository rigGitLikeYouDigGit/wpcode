
from __future__ import annotations
import typing as T

import sys, inspect, functools, types

from dataclasses import dataclass

"""after a great deal of wrestling, this is the system we will use
for referring to types and other elements defined in code.

Nothing crazy, nothing complicated - just do a normal Python namespace
path from the system root.

Do not try and track elements within code when they move or get renamed - 
provide tools and functions to scan over data and update any references
as needed.

Maybe we can in turn try and detect when types move, in commits and so on,
and automatically update references in data files - this is a stretch goal
though.


Here we rely on __qualname__, and will do until we encounter issues.
"""

MODULE_KEY = "module"
CLASS_KEY = "class"
MEMBER_KEY = "member"

MODULE_MEMBER_SEP = ":"

@dataclass
class CodeRefErrorData:
	"""data object holding information about an error,
	 if resolving a code reference fails"""
	codeRefPath : str = None
	modulePath : str = None
	objPath : str = None
	lastFoundParent : object = None
	missingParentMember : str = None

def getCodeRef(obj)->str:
	"""Return a code ref string of the form,
	"module.path:object.path"
	"""
	return MODULE_MEMBER_SEP.join([obj.__module__, obj.__qualname__])

def resolveCodeRef(refStr:str)->T.Any:
	"""resolve a code reference string to the actual object.
	Expects string of the form "module.path:object.path"
	:raises ModuleNotFoundError if the module is not found.
	:raises AttributeError if the object at any level is not found.

	TODO: consider if we need tighter integration with errors - for now
	we pack error data on to the exception object as ".codeRefErrorData"
	"""
	modulePath, objPath = refStr.split(MODULE_MEMBER_SEP)
	try:
		if modulePath in sys.modules:
			module = sys.modules[modulePath]
		else:
			module = __import__(modulePath)
	except ModuleNotFoundError:
		errorData = CodeRefErrorData(
			codeRefPath=refStr,
			modulePath=modulePath)
		exception = ModuleNotFoundError(f"Could not find module {modulePath} for code reference {refStr}")
		exception.codeRefErrorData = errorData
		raise exception
	parentObj = module
	tokens = objPath.split(".")

	for i in range(0, len(tokens)):
		try:
			parentObj = getattr(parentObj, tokens[i])
		except AttributeError:
			errorData = CodeRefErrorData(modulePath=modulePath, objPath=objPath, lastFoundParent=parentObj, missingParentMember=tokens[i])
			exception = AttributeError(f"Could not find attribute {tokens[i]} in object {parentObj} for code reference {refStr}")
			exception.codeRefErrorData = errorData
			raise exception
	return parentObj

