from __future__ import annotations
import typing as T
import pprint, re



def getBreakCharsInStr(s:str,
allStartChars=(
						 "(", "[", "{", "<",
                      ),
                      allEndChars=(
						")", "]", "}", ">",
						),
                      allSymmetricChars=(
						'"""', "'''", "\'", "\"",
                      ),
					  )->list[tuple[int, str, int]]:
	# first find all occurrences of start and end chars
	breakChars = []
	scanStr : str = s
	for i in allStartChars + allEndChars + allSymmetricChars:
		#print("look for i", i)

		# escape special regex chars
		lookup = "\\" + i if i in ("(", ")", "[", "]", "{", "}") else i

		for m in re.finditer(lookup, scanStr):
			if i in allStartChars:
				flag = 0
			elif i in allEndChars:
				flag = 1
			elif i in allSymmetricChars:
				flag = 2
			breakChars.append((m.start(), i, flag))
			scanStr = scanStr[:m.start()] + "w" * len(i) + scanStr[m.end():]
		#print("scanStr", scanStr)

	breakChars.sort(key=lambda x: x[0])
	return breakChars


def _openNewFrame(frameStack):
	newFrame = []
	frameStack.append(newFrame)
	return newFrame


def _closeFrame(frameStack):
	finishedFrame = frameStack.pop()
	finishedFrame = [i for i in finishedFrame if i]
	frameStack[-1].append(finishedFrame)
	oldFrame = frameStack[-1]
	return oldFrame


def _frameIsNeutral(currentFrame, neutraliseChars):
	if not currentFrame:
		return False
	return currentFrame[0] in neutraliseChars


def getExpParseFrames(s:str,
allStartChars=(
	 "(", "[", "{", "<",
	),
	allEndChars=(
	")", "]", "}", ">",
	),
	allSymmetricChars=(
	'"""', "'''", "\'", "\"",
	),
	neutraliseChars=( # characters that neutralise other markers in them
	'"""', "'''", "\'", "\"",
	)
                      )->list[list[str]]:
	"""get frames of expression syntax -
	returns nested list of lists, each list is a frame.
	First and last entry of each list is bounding character of that frame
	"""

	breakChars = getBreakCharsInStr(s)
	#print("breakChars", breakChars)
	#print(s)

	result = []
	charStack = []
	openIndexStack = []
	openStrIndex = 0
	closeIndex = 0
	# result = []

	frameStack = []
	currentFrame = []
	frameTree = currentFrame
	frameStack.append(frameTree)


	for i, breakChar in enumerate(breakChars):

		if breakChar[2] == 0:
			if _frameIsNeutral(currentFrame, neutraliseChars):
				continue

			# add the string before the break char to the current frame
			currentFrame.append(s[closeIndex:breakChar[0]])
			closeIndex = breakChar[0] + len(breakChar[1])
			# open new frame
			currentFrame = _openNewFrame(frameStack)
			currentFrame.append(breakChar[1])


		if breakChar[2] == 1:
			if _frameIsNeutral(currentFrame, neutraliseChars):
				continue

			# add the string before the break char to the current frame
			currentFrame.append(s[closeIndex:breakChar[0]])
			closeIndex = breakChar[0] + len(breakChar[1])
			# close current frame - add it to previous frame in stack
			currentFrame.append(breakChar[1])
			currentFrame = _closeFrame(frameStack)
		if breakChar[2] == 2: # need to check before and ahead
			if currentFrame[0] == breakChar[1]:
				# add the string before the break char to the current frame
				currentFrame.append(s[closeIndex:breakChar[0]])
				closeIndex = breakChar[0] + len(breakChar[1])
				currentFrame.append(breakChar[1])
				currentFrame = _closeFrame(frameStack)
			else:
				# add the string before the break char to the current frame
				currentFrame.append(s[closeIndex:breakChar[0]])
				closeIndex = breakChar[0] + len(breakChar[1])
				currentFrame = _openNewFrame(frameStack)
				currentFrame.append(breakChar[1])

	#pprint.pprint(frameStack, depth=10, indent=2, compact=False)
	return frameStack


def getBracketContents(s:str, encloseChars="()")->tuple[tuple, str]:
	"""recursively get items contained in outer brackets
	couldn't find a flexible way online so doing it raw
	"""
	stack = 0
	openIndex = 0
	closeIndex = 0
	result = []
	for i, char in enumerate(s):
		if char == encloseChars[0]:
			if stack == 0:
				openIndex = i
			stack += 1
		elif char == encloseChars[1]:
			stack -= 1
			if stack == 0:
				result.append(s[closeIndex:openIndex])
				result.append(getBracketContents(s[ openIndex + 1 : i ], encloseChars))
				closeIndex = i + 1
	result.append(s[closeIndex:])
	#return tuple(filter(None, result))
	return tuple(i for i in result if not (i == ""))


def bracketContentsAreFunction(expBracketContents:tuple[tuple, str]) ->bool:
	"""check if expression is a function
	- does expression start with brackets?
	- does first bracket group precede a colon?
	"""
	if not isinstance(expBracketContents[0], tuple): # flat string
		return False
	if expBracketContents[1][0] != ":": # no colon after first bracket group
		return False
	return True


def restoreBracketContents(expBracketContents:tuple[tuple, str], encloseChars="()")->str:
	"""restore bracket contents to original string"""
	result = ""
	for i, item in enumerate(expBracketContents):
		if isinstance(item, tuple):
			result += encloseChars[0] + restoreBracketContents(item, encloseChars) + encloseChars[1]
		else:
			result += item
	return result


def textDefinesExpFunction(text: str) -> bool:
	"""check if text defines a function
	- does expression start with brackets?
	- does first bracket group precede a colon?
	"""
	return bracketContentsAreFunction(getBracketContents(text.strip()))


def splitTextBodySignature(text: str) -> tuple[str, str]:
	"""split function text into body and signature"""
	bracketParts = getBracketContents(text.strip())
	signature = bracketParts[0]
	body = bracketParts[1:]
	return restoreBracketContents(signature), restoreBracketContents(body)[1:]