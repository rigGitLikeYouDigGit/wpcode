
from __future__ import annotations
import typing as T

from dataclasses import dataclass

from wp.object import ErrorReport

class ValidationError(Exception):
	pass

@dataclass
class ValidationResult:
	"""result of a validation check -
	return success or failure, descriptive error,
	optionally a suggested replacement value"""
	success : bool = True
	errorReport : ErrorReport = None
	suggestion : T.Any = None

class Rule:

	def __init__(self, name:str=""):
		self.name = name or self.__class__.__name__

	def checkInput(self, data)->bool:
		"""override - check input against rule, return
		:raises ValidationError: if input fails rule check
		"""
		raise NotImplementedError

	def getSuggestedValue(self, data)->T.Any:
		"""override - return a suggested value to replace
		invalid input"""
		return None

class RuleSet:

	def __init__(self, rules:list[Rule]):
		self.rules = rules
		self.results = [] # feels weird to give this object state

	def checkInput(self, data)->bool:
		"""check input against all rules, return True if all pass.
		If any fail (or an error is raised), return False.
		Calling code can then inspect self.results for more info"""
		self.results = []
		for rule in self.rules:
			try:
				result = rule.checkInput(data)
				if result == True or result is None:
					self.results.append(ValidationResult(success=True))
			except ValidationError as e: # input failed rule check
				suggestion = rule.getSuggestedValue(data)
				self.results.append(
					ValidationResult(
						success=False,
						errorReport=ErrorReport(
							e, f"Validation error for input {data} \n\t with rule {rule.name} "
							),
						suggestion=suggestion
					                 )
				)
			except Exception as e: # something gone wrong in actual rule code
				self.results.append(
					ValidationResult(success=False,
					errorReport=ErrorReport(
						e, f"Error validating input {data} \n\t with rule {rule.name} "
						)
					                 )
				)
		#print("results", self.results)
		successStatuses = [result.success for result in self.results]
		return all(successStatuses)

