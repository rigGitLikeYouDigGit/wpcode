
"""string library"""
from __future__ import print_function, annotations

import string, re, fnmatch


def cap(s:str): # fight me
	return s[0].upper() + s[1:]

def lowFirst(s:str):
	return s[0].lower() + s[1:]

def camelJoin(*parts, lowerFirst=True):
	parts = tuple(filter(None, parts))
	return (lowFirst(parts[0]) if lowerFirst else parts[0]) + "".join(cap(i) for i in parts[1:])

def multiSplit(s:str, separators:list[str], preserveSepChars=False)->list[str]:
	"""if preserveSepChars, each separator char will be returned as a separate token"""
	tokens = []
	startIndex = 0
	endIndex = -1
	for i in range(len(s)):
		if s[i] in separators:
			if preserveSepChars:
				tokens.append(s[i])
			tokens.append(s[startIndex:i])
			startIndex = i + 1
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

def incrementName(name, currentNames=None):
	"""checks if name is already in children, returns valid one"""
	if currentNames and name not in currentNames:
		return name
	if name[-1].isdigit():  # increment digit like basic bitch
		new = int(name[-1]) + 1
		return name[:-1] + str(new)
	if name[-1] in string.ascii_uppercase:  # ends with capital letter
		if name[-1] == "Z":  # start over
			name += "A"
		else:  # increment with letter, not number
			index = string.ascii_uppercase.find(name[-1])
			name = name[:-1] + string.ascii_uppercase[index + 1]
	else:  # ends with lowerCase letter
		name += "B"

	# check if name already taken
	if currentNames and name in currentNames:
		return incrementName(name, currentNames)
	return name



if __name__ == '__main__':
	# testS = "d-"
	# print(multiSplit(testS, ("-", "."), before=False))
	# print(multiSplit(testS, ("-", "."), before=True))

	testS = "3::3"
	print(sliceFromString(testS))

	testS = "2:"
	print(sliceFromString(testS))

	testS = "2:3:4"
	print(sliceFromString(testS))

	testS = "::4"
	print(sliceFromString(testS))

	testS = ":4"
	print(sliceFromString(testS))


