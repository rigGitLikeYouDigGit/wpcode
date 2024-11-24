
"""string library"""
from __future__ import print_function, annotations
import typing as T

import string, re, fnmatch

def cap(s:str): # fight me
	"""capitalize first letter of string"""
	return s[0].upper() + s[1:]

def lowFirst(s:str):
	return s[0].lower() + s[1:]

def camelJoin(*parts, lowerFirst=True):
	parts = tuple(filter(None, parts))
	return (lowFirst(parts[0]) if lowerFirst else parts[0]) + "".join(cap(i) for i in parts[1:])

def indicesOf(s:str, char:str)->list[int]:
	"""return indices of char in s"""
	return [i for i, c in enumerate(s) if c == char]

def indicesOfAny(s:str, chars:T.Iterable[str])->list[int]:
	"""return indices of any char in chars"""
	return [i for i, c in enumerate(s) if c in chars]

def splitIndices(s:str, indices:list[int])->list[str]:
	"""split string at given indices"""
	return [s[i:j] for i, j in zip(indices, indices[1:] + [None])] or [s]

def splitAround(s:str, char:str, before=True, after=True)->list[str]:
	"""split string around given char while preserving it-
	if before, add split before each instance of char
	if after, add split after each instance of char
	if both, add split before and after each instance of char

	some strings may be empty - don't shorten list within this function?
	on one hand, we could guarantee exactly how many tokens are returned
	for given number of occurrences
	but then for a double occurrence "//" you'd get an empty string between
	them?
	seems uselesss
	"""
	baseIndices = indicesOf(s, char)
	indices = []
	for index in baseIndices:
		if before:
			indices.append(index)
		if after:
			indices.append(index + 1)
	indices = sorted((indices))
	return list(filter(None,  splitIndices(s, indices)))

def mergeRepeated(s:str, chars:str)->str:
	"""merge blocks of repeated chars in string"""
	res = ""
	for c in s:
		if res and res[-1] == c and c in chars:
			continue
		res += c
	return res

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

def splitBefore(s:str, splitChar:str):
	"""split a string on given char, but do not remove that char from
	split parts"""
	if not splitChar in s[1:]:
		return [s]
	parts = s.split(splitChar)
	if s.startswith(splitChar):
		parts[0] = splitChar + parts[0]
	return [parts[0], *[splitChar + i for i in parts[1:]]]


def sliceFromString(s:str, sliceChar=":")->(slice, None):
	"""returns a slice object from evaluated string -
	this may be better placed somewhere else
	dumb iteration used here since we expect strings of length < 10 chars

	eg from '3:4:1' return that corresponding slice object
	"""
	if not sliceChar in s:
		return None
	args = [None, None, None]
	colonIndices = (-1, *[i for i, char in enumerate(s) if char == sliceChar], len(s))
	for i in range(len(colonIndices) - 1):
		info = s[colonIndices[i] + 1 : colonIndices[i + 1]]
		args[i] = int(info) if info else None
	return slice(*args)

def stringMatches(testStr:str, matchStr, allowPartial=False, regex=False, fnMatch=False):
	"""main function performing string matching -
	allow options for regex, glob, fnmatch etc"""
	if testStr == matchStr:
		return True
	if allowPartial and matchStr in testStr:
		return True
	if regex and re.compile(matchStr).match(testStr):
		return True
	if fnMatch and fnmatch.fnmatch(testStr, matchStr):
		return True
	return

def trailingDigits(s:str)->tuple[str, str]:
	"""for a string message_012
	return ("message_", "012")
	if no trailing digits, second string will be empty
	"""
	if not s:
		return s, ""
	i = 1
	while s[-i].isdigit():
		i+=1
	splitI = -(i-1)
	return s[:splitI], s[splitI:]

def _incrementNameInner(name, currentNames):
	name, digits = trailingDigits(name)
	if digits:
		nDigits = len(digits) # keep same number of characters by zfilling
		val = int(nDigits) + 1
		newDigits = str(val).zfill(nDigits)
		return name + newDigits
	if name[-1] in string.ascii_uppercase:  # ends with capital letter
		if name[-1] == "Z":  # start over
			name += "A"
		else:  # increment with letter, not number
			index = string.ascii_uppercase.find(name[-1])
			name = name[:-1] + string.ascii_uppercase[index + 1]
	else:  # ends with lowerCase letter
		name += "B"
	return name
def incrementName(name, currentNames:T.Iterable[str]=None):
	"""checks if name is already in children, returns valid one"""

	# check if name already taken
	while currentNames and name in currentNames:
		name = _incrementNameInner(name, currentNames)
	return name



if __name__ == '__main__':
	pass

	testS = "/root/trunk/branch/leaf/"
	print(splitAround(testS, "/"))
	testS = "root"
	print(splitAround(testS, "/"))
	testS = "/root//trunk/branch/leaf/"
	print(splitAround(testS, "/"))
	# testS = "d-"
	# print(multiSplit(testS, ("-", "."), before=False))
	# print(multiSplit(testS, ("-", "."), before=True))
	#
	# testS = "3::3"
	# print(sliceFromString(testS))
	#
	# testS = "2:"
	# print(sliceFromString(testS))
	#
	# testS = "2:3:4"
	# print(sliceFromString(testS))
	#
	# testS = "::4"
	# print(sliceFromString(testS))
	#
	# testS = ":4"
	# print(sliceFromString(testS))


