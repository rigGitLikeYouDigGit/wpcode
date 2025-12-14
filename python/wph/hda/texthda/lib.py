from __future__ import annotations
import types, typing as T
import pprint
from collections import defaultdict

from pathlib import Path

import hou


"""in support of convention:

name_version_comment

in file names
"""

def multiSplit(s:str, separators:T.Iterable[str], preserveSepChars=False)->list[str]:
	"""if preserveSepChars, each separator char will be returned as a separate token"""
	#from wplib import log
	tokens = []
	startIndex = 0
	endIndex = -1
	for i in range(len(s)):
		#log("tokens", tokens, i, s[i])
		if s[i] in separators:
			if preserveSepChars:
				tokens.append(s[i])
			tokens.append(s[startIndex:i])
			startIndex = i + 1
	# if not tokens:
	# 	return [s]
	tokens.append(s[startIndex:])
	return tokens


def getVersionFromPath(p)->(int, str) | None:
	"""look for "vxxx" in path, or other space/whitespace-separated tokens that
	evaluate as an int

	prefer longer tokens, in case people want to define "larger" versions of hdas like
	"myFancyHda", "myFancierHda2" etc
	animals

	"""

	tokens = multiSplit( str(p), " _-" )
	if len(tokens) == 1:
		return None, None

	for i in tokens:
		# check for fully numeric token
		if i.isdigit():
			return int(i), i
		# check for "v020" v-leading token
		if i[0] in "vV":
			if i[1:] and i[1:].isdigit():
				return int(i[1:]), i

	return None, None

def getNameVersionCommentFromPath(p):
	p = str(p)
	version, token = getVersionFromPath(p)
	if version is None:
		return None
	parts = p.split(token, 1)
	name = ""
	comment = ""
	if len(parts) == 2:
		name, comment = parts
	else:
		name = parts[0]
	return name, version, comment

def versionDataInDir(p:Path)->dict[str, dict[int, (str, str)]]:
	"""return map of
	{ name : { version : (path, comment) }
	}
	"""
	versionMap = defaultdict(dict)
	for i in p.iterdir():
		data = getNameVersionCommentFromPath(i)
		if data is None:
			versionMap[name][1] = (str(i), "")
			continue
		name, version, comment = data
		versionMap[name][version] = (str(i), comment)
	return versionMap


def updateVersionOptionsWhenPathUpdated(
		pathParm:hou.Parm,
		versionMenuParm:hou.Parm
):
	pass


