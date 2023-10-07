
from __future__ import annotations
import typing as T



class StringCodeTemplate:

	def resultString(self)->str:
		"""return the result string"""
		raise NotImplementedError

class FunctionTemplate(StringCodeTemplate):

	def __init__(self,
	             fnName,
	             fnArgs,
	             fnKwargs,
	             fnBody,
	             returnType="object",

	             ):
		self.fnName = fnName
		self.fnArgs = fnArgs
		self.fnKwargs = fnKwargs
		self.returnType = returnType
		self.fnBody = fnBody

	def formatArgs(self):
		"""format the arguments"""
		result = ""
		for arg in self.fnArgs:
			if isinstance(arg, tuple):
				result += f"{arg[0]}:{arg[1]}, "
			else:
				result += f"{arg}, "
		return result[:-2]

	def formatKwargs(self):
		"""format the kwargs"""
		result = ""
		for arg in self.fnKwargs:
			if isinstance(arg, tuple):
				if len(arg) == 3 and arg[1]:
					result += f"{arg[0]}:{arg[1]}={arg[2]}, "
				else:
					result += f"{arg[0]}={arg[1]}, "
			else:
				result += f"{arg}, "
		return result[:-2]

	def formatArgsAndKwargs(self):
		"""format the args and kwargs"""
		if self.fnArgs and self.fnKwargs:
			return self.formatArgs() + ", " + self.formatKwargs()
		elif self.fnArgs:
			return self.formatArgs()
		elif self.fnKwargs:
			return self.formatKwargs()
		return ""


	def resultString(self):
		return f"""def {self.fnName}({self.formatArgsAndKwargs()})->{self.returnType}:\n\t{self.fnBody}"""



