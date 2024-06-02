
from __future__ import annotations
import typing as T
import string, textwrap

"""
TODO: maybe refactor to make ArgT, ArgsKwargsT, TypeT, etc. into classes


"""

# type for a type hint - either a string or a type
typeT = T.Union[str, type]
# type for arguments - either a string or a tuple of (name, type) for hinting
argT = T.Union[str, tuple[str, typeT]]
argsKwargsT = tuple[argT, dict[argT, T.Any]]

indent = lambda s: textwrap.indent(str(s), "\t")

def formatType(t:typeT)->str:
	"""format a type"""
	if isinstance(t, type):
		return t.__name__
	return str(t)

def formatArg(arg:argT, space=False)->str:
	"""format an argument"""
	if isinstance(arg, tuple):
		if space:
			return f"{arg[0]} : {formatType(arg[1])}"
		return f"{arg[0]}:{formatType(arg[1])}"
	return arg

def formatArgsKwargs(argsKwargs:argsKwargsT)->str:
	"""format a list of arguments"""
	argStrs = [formatArg(arg) for arg in argsKwargs[0]]
	kwargStrs = [f"{formatArg(k)}={v}" for k, v in argsKwargs[1].items()]
	return ", ".join(map(str, argStrs + kwargStrs))

def formatLines(thing:T.Any)->str:
	"""small recursive function to format sequences"""
	if isinstance(thing, (list, tuple)):
		return "\n".join(map(formatLines, thing))
	return str(thing)

class StringCodeTemplate:

	def __init__(self,
	             depth:int=0,
	             ):
		self.indentDepth = depth

	def _resultString(self)->str:
		""" OVERRIDE
		return the result string"""
		raise NotImplementedError

	def finalString(self)->str:
		"""return the final string indented"""
		#self.updateChildDepths()
		#return textwrap.indent(self._resultString(), "\t" * (self.indentDepth or 0))
		return self._resultString()

	def updateChildDepths(self):
		"""update the depths of child templates"""
		for child in self.childTemplates():
			child.indentDepth = self.indentDepth + 1
			child.updateChildDepths()

	def childTemplates(self)->T.Iterable[StringCodeTemplate]:
		"""return child templates"""
		return []

	def __str__(self):
		return self.finalString()

class Literal(StringCodeTemplate):
	"""template for a literal string"""

	def __init__(self,
	             literal:str,
	             depth:int=0,
	             ):
		super().__init__(depth=depth)
		self.literal = literal

	def _resultString(self):
		return "\"" + self.literal + "\""

class TextBlock(StringCodeTemplate):
	"""template for a block of text"""

	def __init__(self,
	             text:str,
	             depth:int=0,
	             ):
		super().__init__(depth=depth)
		if isinstance(text, TextBlock):
			self.text = text.text
		else:
			self.text = text

	def _resultString(self):
		return self.text

class IfBlock(StringCodeTemplate):
	"""template for an if block
	lists of (condition, block) pairs are written as
	if condition:
		block
	elif condition:
		block

	with optional else block at end
	"""
	def __init__(self,
	             conditionBlocks:list[[StringCodeTemplate, StringCodeTemplate]],
	             elseBlock:tuple[StringCodeTemplate, StringCodeTemplate]=None,
	             depth:int=0,
	             ):
		super().__init__(depth=depth)
		self.conditionBlocks=conditionBlocks
		self.elseBlock = elseBlock

	def _resultString(self):
		conditionStrs = [
			f"if {str(cond)}:\n" + textwrap.indent(
				formatLines(block), "\t") for cond, block in self.conditionBlocks]
		if self.elseBlock:
			conditionStrs.append(f"else:\n{str(self.elseBlock[1])}")
		return "\n".join(conditionStrs)

	# def updateChildDepths(self):
	# 	print("update depths", self.conditionBlocks, self.elseBlock)
	# 	super().updateChildDepths()
	# 	print("child depths", [i.indentDepth for i in self.childTemplates()])
	def childTemplates(self):
		#print("child templates", self.conditionBlocks, self.elseBlock)
		return [block for cond, block in self.conditionBlocks] + ([self.elseBlock[1]] if self.elseBlock else [])

class ForBlock(StringCodeTemplate):
	"""template for a for block
	"""
	def __init__(self,
	             varAssignStr:str,
	             iterable:StringCodeTemplate,
	             body:list[StringCodeTemplate],
	             depth:int=0,
	             ):
		super().__init__(depth=depth)
		self.varAssignStr = varAssignStr
		self.iterable = iterable
		self.bodyLines = body


	def _resultString(self):
		return f"for {self.varAssignStr} in {self.iterable}:\n" + "\n".join(str(i for i in self.bodyLines))

	def childTemplates(self):
		return [self.bodyLines]

class Import(StringCodeTemplate):
	"""single import line, with optional alias

	TODO: add support for multiple imports on one line
	"""

	def __init__(self,
	             fromModule:str=None,
	             module:str=None,
	             alias:str=None,
	             depth:int=0,
	             ):
		super().__init__(depth=depth)
		self.fromModule = fromModule
		self.module = module
		self.alias = alias

	def _resultString(self):
		if self.fromModule:
			impStr =  f"from {self.fromModule} import {self.module}"
		else:
			impStr = f"import {self.module}"
		if self.alias:
			impStr += f" as {self.alias}"
		return impStr


class Assign(StringCodeTemplate):
	"""template for an assignment"""

	def __init__(self,
	             left:argT,
	             right:StringCodeTemplate=None,
	             depth:int=0,
	             ):
		super().__init__(depth=depth)
		self.left = left
		self.right = right

	def _resultString(self):
		if self.right:
			return f"{formatArg(self.left, space=True)} = {str(self.right)}"
		return f"{formatArg(self.left, space=True)}"

	def childTemplates(self) ->T.Iterable[StringCodeTemplate]:
		#return [self.right]
		return []

class FunctionCallTemplate(StringCodeTemplate):
	"""template for a function call"""

	def __init__(self,
	             fnName:str,
	             fnArgs:argsKwargsT=(),
	             depth:int=0,
	             ):
		super().__init__(depth=depth)
		self.fnName = fnName
		self.fnArgs = fnArgs

	def _resultString(self):
		return f"{self.fnName}({formatArgsKwargs(self.fnArgs)})"

	def childTemplates(self) ->T.Iterable[StringCodeTemplate]:
		return []

class FunctionTemplate(StringCodeTemplate):
	"""template for a function declaration"""

	def __init__(self,
	             fnName,
	             fnArgs:list[argT],
	             fnKwargs:dict[argT, T.Any],
	             fnBody:TextBlock,
	             fnDecorator:(FunctionCallTemplate, str)="",
	             returnType=None,
					depth=0
	             ):
		super().__init__(depth=depth)
		self.fnName = fnName
		self.fnArgs = fnArgs
		self.fnKwargs = fnKwargs
		self.returnType = returnType
		self.fnBody = fnBody
		self.fnDecorator = fnDecorator

	def formatArgs(self):
		"""format the arguments"""
		kwargStrs = [f"{formatArg(k)}={v}" for k, v in self.fnKwargs.items()]
		return " ,\n\t".join([formatArg(arg) for arg in self.fnArgs] + kwargStrs)

	def _resultString(self):
		baseStr = ""
		if self.fnDecorator:
			baseStr += f"@{self.fnDecorator}\n"
		baseStr += f"""def {self.fnName}({self.formatArgs()})->{self.returnType}:\n""" + indent(self.fnBody)
		return baseStr

	def childTemplates(self):
		return [self.fnBody]


class ClassTemplate(StringCodeTemplate):
	"""template for a class declaration

	children?
	maybe have Line object, Assignment object etc

	"""

	def __init__(self,
	             className:str,
	             classBaseClasses:tuple[str],
	             #classAttrs:dict[argT, T.Any]=None,
	             classLines:tuple[StringCodeTemplate]=(),
	             classMethods:tuple[FunctionTemplate]=(),
	             depth:int=0,
	             ):
		super().__init__(depth=depth)
		self.className = className
		self.classBaseClasses = classBaseClasses
		self.classLines = classLines or {}
		self.classMethods = classMethods


	def formatBaseClasses(self):
		"""format the base classes"""
		return ", ".join(self.classBaseClasses)

	def _resultString(self):
		if self.classBaseClasses:
			defStr = f"class {self.className}({self.formatBaseClasses()}):"
		else:
			defStr = f"class {self.className}:"
		lineStr = ""
		methodStr = ""

		if self.classLines:
			lineStr = "\n".join([str(line) for line in self.classLines])
		if self.classMethods:
			methodStr = "\n\n".join([str(method) for method in self.classMethods])
		childBlock = "\n".join(filter(None, [lineStr, methodStr, "pass"]))
		childBlock = indent(childBlock)
		endStr = "\n".join([defStr, childBlock])

		return endStr

	def childTemplates(self):
		return [*self.classLines, *self.classMethods]


