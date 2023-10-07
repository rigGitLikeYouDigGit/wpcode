

from __future__ import annotations
import typing as T

import keyword

from pathlib import Path

from wplib.codegen.strtemplate import FunctionTemplate

from maya import cmds


TARGET_PATH = Path(__file__).parent / "cmds.pyi"

flagTypeMap = {
	"string": "str",
	"enum": "str",
	"name" : "str",
	"script" : "str",

	"boolean": "bool",
"on|off": "bool",

	"float": "float",
	"angle": "float",
	"length": "float",
	"time": "float",
	"diameter": "float",
	"distance": "float",

	"int": "int",
	"unsignedint": "int",
	"index": "int",
	"integer": "int",
	"long": "int",
	"short": "int",
	"byte": "int",
	"int64" : "int",
	"string[]": "list[str]",
	"string[...]": "list[str]",
	"float[]": "list[float]",
	"float[...]": "list[float]",
	"floatrange": "list[float]",
	"int[]": "list[int]",
	"int[...]": "list[int]",
	"indexrange": "list[int]",
	"timerange": "list[float]",

"(multi-use)" : "",
"(query" : "",
	"arg" : "",
	"optional)" : "",
	"mandatory)" : "",
	"[" : "",
	"]" : "",
	"(" : "",
	")" : "",

}

keyword

def getDefinitionForHelpItem(
		cmdName:str,
		helpString:str,
		):
	"""
	for xform


Synopsis: xform [flags] [String...]
Flags:
   -q -query
   -a -absolute
  -bb -boundingBox
 -bbi -boundingBoxInvisible
  -cp -centerPivots
 -cpc -centerPivotsOnComponents
 -dph -deletePriorHistory        on|off
  -eu -euler
   -m -matrix                    Float Float Float Float Float Float Float Float Float Float Float Float Float Float Float Float
  -os -objectSpace
   -p -preserve                  on|off
 -piv -pivots                    Length Length Length
 -puv -preserveUV
   -r -relative
  -ra -rotateAxis                Angle Angle Angle
 -rab -reflectionAboutBBox
 -rao -reflectionAboutOrigin
 -rax -reflectionAboutX
 -ray -reflectionAboutY
 -raz -reflectionAboutZ
 -rfl -reflection
 -rft -reflectionTolerance       Float
  -ro -rotation                  Angle Angle Angle
 -roo -rotateOrder               String
  -rp -rotatePivot               Length Length Length
  -rt -rotateTranslation         Length Length Length
   -s -scale                     Float Float Float
  -sh -shear                     Float Float Float
  -sp -scalePivot                Length Length Length
  -st -scaleTranslation          Length Length Length
   -t -translation               Length Length Length
  -wd -worldSpaceDistance
  -ws -worldSpace
 -ztp -zeroTransformPivots


Command Type: Command
	"""

	if helpString.startswith("Quick help is not"):
		return ""

	lines = helpString.split("\n")
	lines = [line.lstrip().rstrip() for line in lines]
	# print("")
	# for line in lines:
	# 	firstTokens = line.split()
	# 	print(firstTokens)
	# 	print("[String...]" in firstTokens)

	if len(lines) < 3:
		return ""

	if not lines[2].startswith("Synopsis:"):
		return ""

	# check for subjects
	firstTokens = lines[2].split()
	args = []
	if "[String...]" in firstTokens:
		args.append(("*subjects", "list[str]"))

	if "Flags:" not in lines:
		template = FunctionTemplate(
			fnName=cmdName,
			fnArgs=args,
			fnKwargs=[],
			fnBody="pass",
			returnType="object",
		)
		return template.resultString()

	flagIndex = lines.index("Flags:")
	kwargs = []
	shortKwargs = []
	cqeKwargs = []
	for line in lines[flagIndex+1:]:
		if not line.startswith("-"):
			break
		tokens = line.split()
		shortFlag = tokens[0][1:]# or "MISSING_SHORT_FLAG"
		longFlag = tokens[1][1:]# or "MISSING_LONG_FLAG"

		if not shortFlag or not longFlag:
			continue

		if keyword.iskeyword(longFlag):
			longFlag = longFlag + "_"
		if keyword.iskeyword(shortFlag):
			shortFlag = shortFlag + "_"


		givenFlagTypes = tokens[2:]
		if givenFlagTypes:
			if len(givenFlagTypes) == 1:
				flagType = flagTypeMap[givenFlagTypes[0].lower()]
			else:
				try:
					flagType = "tuple[ " + ", ".join(
						filter(None, (flagTypeMap[i.lower()] for i in givenFlagTypes))) + " ]"
				except Exception:
					flagType = flagTypeMap[givenFlagTypes[0].lower()]
		else:
			flagType = "bool"

		longKwarg = (longFlag, flagType, 0)
		shortKwarg = (shortFlag, flagType, 0)

		if shortFlag in "qe":
			cqeKwargs.append(longKwarg)
			cqeKwargs.append(shortKwarg)
			continue

		kwargs.append(longKwarg)
		shortKwargs.append(shortKwarg)

	kwargs += cqeKwargs
	kwargs += shortKwargs

	template = FunctionTemplate(
		fnName=cmdName,
		fnArgs=args,
		fnKwargs=kwargs,
		fnBody="pass",
		returnType="object",
	)
	return template.resultString()


def genCmdsCodeHints(targetPath:Path):

	fnNames = cmds.help("*", list=1)
	#fnNames = sorted(fnNames, reverse=1)[:4]

	resultString = ""
	for fnName in fnNames:
		helpString = cmds.help(fnName)
		#resultString += helpString + "\n\n"
		try:
			resultString += getDefinitionForHelpItem(
				fnName,
				helpString
			) + "\n\n"
		except Exception as e:
			print(helpString)
			targetPath.write_text(resultString)
			raise e

	targetPath.write_text(resultString)








